"""
Handoff -- Typed agent-to-agent handoff protocol.

Routes work between agents with guardrail enforcement, audit trails,
timeout management, and chained execution.

Usage:
    from sovrun.core.handoff import HandoffRouter, HandoffRequest, handoff_target

    router = HandoffRouter()

    @handoff_target("summarizer")
    def summarize(context: dict) -> dict:
        return {"summary": context["text"][:100]}

    router.register(summarize)

    result = router.send(HandoffRequest(
        source_agent="planner",
        target_agent="summarizer",
        context={"text": "long document..."},
        task_description="Summarize the document",
    ))

CLI:
    python3 -m sovrun.core.handoff --status    # Show registered agents
    python3 -m sovrun.core.handoff --history    # Show recent handoffs
    python3 -m sovrun.core.handoff --history 50 # Show last 50 handoffs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .resilience import (
    CircuitBreaker,
    TrajectoryLogger,
    safe_path,
    PROJECT_ROOT,
)

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
LOGS_DIR = Path(os.environ.get("SOVRUN_LOGS_DIR", PROJECT_ROOT / "logs"))
HANDOFF_LOG = LOGS_DIR / "handoffs.jsonl"

logger = logging.getLogger("sovrun.handoff")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ts() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Guardrail scopes
# ---------------------------------------------------------------------------

class GuardrailScope(str, Enum):
    """What the target agent is allowed to do."""
    READ_ONLY = "read_only"
    WRITE_ALLOWED = "write_allowed"
    DESTRUCTIVE_ALLOWED = "destructive_allowed"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HandoffRequest:
    """A request to hand work from one agent to another."""
    source_agent: str
    target_agent: str
    context: dict[str, Any]
    task_description: str
    guardrail_scope: GuardrailScope = GuardrailScope.READ_ONLY
    timeout_seconds: float = 120.0
    callback_on_complete: Callable[[Any], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["guardrail_scope"] = self.guardrail_scope.value
        d.pop("callback_on_complete", None)
        return d


@dataclass
class HandoffResult:
    """Result of a completed handoff."""
    success: bool
    result_data: dict[str, Any]
    error: str | None = None
    duration_ms: int = 0
    tokens_used: int = 0
    source_agent: str = ""
    target_agent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# @handoff_target decorator
# ---------------------------------------------------------------------------

_HANDOFF_ATTR = "_handoff_agent_name"


def handoff_target(agent_name: str) -> Callable[..., Any]:
    """Mark a function as a handoff-capable agent endpoint.

    The decorated function receives (context: dict) and returns a dict.
    """
    def decorator(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        setattr(fn, _HANDOFF_ATTR, agent_name)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Guardrail enforcement
# ---------------------------------------------------------------------------

# Actions that require elevated scope
_WRITE_ACTIONS = {"write", "create", "update", "modify", "append", "save"}
_DESTRUCTIVE_ACTIONS = {"delete", "remove", "drop", "kill", "reset", "purge", "destroy"}


def _check_guardrails(request: HandoffRequest) -> str | None:
    """Return an error string if the request violates its guardrail scope, else None."""
    scope = request.guardrail_scope
    ctx = request.context
    action = str(ctx.get("action", "")).lower()

    if scope == GuardrailScope.READ_ONLY:
        if any(w in action for w in _WRITE_ACTIONS):
            return f"BLOCKED: write action '{action}' not allowed under read_only scope"
        if any(w in action for w in _DESTRUCTIVE_ACTIONS):
            return f"BLOCKED: destructive action '{action}' not allowed under read_only scope"

    if scope == GuardrailScope.WRITE_ALLOWED:
        if any(w in action for w in _DESTRUCTIVE_ACTIONS):
            return f"BLOCKED: destructive action '{action}' not allowed under write_allowed scope"

    return None


# ---------------------------------------------------------------------------
# GateKeeper -- action-level approval gates
# ---------------------------------------------------------------------------

GATES_FILE = Path(os.environ.get(
    "SOVRUN_GATES_FILE", PROJECT_ROOT / "config" / "gates.json"))

# Actions blocked by default unless explicitly allowed
_DEFAULT_BLOCKED: list[str] = [
    "delete", "send_email", "spend_money",
    "deploy_production", "external_api_write",
]


class GateKeeper:
    """Block certain action types unless explicitly approved.

    Stores gate configuration in config/gates.json. Actions on the blocked
    list are rejected by check_gate() unless removed or overridden.

    Usage:
        gk = GateKeeper()
        gk.check_gate("read_file")        # True (allowed)
        gk.check_gate("delete")           # False (blocked by default)
        gk.add_gate("nuke_db")            # add custom blocked action
        gk.remove_gate("send_email")      # unblock an action
    """

    def __init__(self, gates_file: Path | None = None) -> None:
        self._path = gates_file or GATES_FILE
        self._blocked: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load gates from disk, merging with defaults."""
        self._blocked = set(_DEFAULT_BLOCKED)
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                stored = data.get("blocked_actions", [])
                if isinstance(stored, list):
                    self._blocked = set(stored)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        """Persist gate config to disk."""
        _ensure_dir(self._path.parent)
        data = {"blocked_actions": sorted(self._blocked)}
        self._path.write_text(json.dumps(data, indent=2) + "\n")

    def add_gate(self, action: str, requires_approval: bool = True) -> None:
        """Add an action to the blocked list."""
        if requires_approval:
            self._blocked.add(action.lower())
        else:
            self._blocked.discard(action.lower())
        self._save()

    def remove_gate(self, action: str) -> None:
        """Remove an action from the blocked list (allow it)."""
        self._blocked.discard(action.lower())
        self._save()

    def check_gate(self, action: str) -> bool:
        """Return True if the action is ALLOWED, False if blocked."""
        return action.lower() not in self._blocked

    def list_gates(self) -> list[str]:
        """Return sorted list of currently blocked actions."""
        return sorted(self._blocked)

    def status(self) -> str:
        """Formatted status string."""
        gates = self.list_gates()
        lines = [f"\n=== GateKeeper ({len(gates)} blocked actions) ===\n"]
        for g in gates:
            lines.append(f"  BLOCKED: {g}")
        if not gates:
            lines.append("  (no gates configured -- all actions allowed)")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HandoffRouter
# ---------------------------------------------------------------------------

class HandoffRouter:
    """Routes handoff requests to registered agent endpoints.

    Thread-safe registry. Logs every handoff to JSONL audit trail.
    Supports sync and async (callback) execution with timeout enforcement.
    """

    def __init__(self, log_path: Path | None = None,
                 gate_keeper: GateKeeper | None = None) -> None:
        self._registry: dict[str, Callable[..., dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._log_path = log_path or HANDOFF_LOG
        self._trajectory = TrajectoryLogger("handoff_router")
        self._cb = CircuitBreaker(name="handoff", failure_threshold=5,
                                  recovery_timeout=120, window=600)
        self._gate = gate_keeper or GateKeeper()
        _ensure_dir(self._log_path.parent)

    # -- Registry -----------------------------------------------------------

    def register(self, fn: Callable[..., dict[str, Any]],
                 name: str | None = None) -> None:
        """Register a callable as a handoff target.

        If fn was decorated with @handoff_target, its name is used
        unless overridden by the name parameter.
        """
        agent_name = name or getattr(fn, _HANDOFF_ATTR, None)
        if not agent_name:
            raise ValueError(
                f"No agent name for {fn.__name__}. "
                "Use @handoff_target or pass name= explicitly."
            )
        with self._lock:
            self._registry[agent_name] = fn
            logger.info("registered handoff target: %s", agent_name)

    def unregister(self, name: str) -> bool:
        """Remove an agent from the registry. Returns True if it existed."""
        with self._lock:
            return self._registry.pop(name, None) is not None

    def list_agents(self) -> list[str]:
        """Return sorted list of registered agent names."""
        with self._lock:
            return sorted(self._registry.keys())

    # -- Audit trail --------------------------------------------------------

    def _log_handoff(self, request: HandoffRequest, result: HandoffResult) -> None:
        """Append handoff record to JSONL audit trail."""
        entry = {
            "ts": _now_iso(),
            "source": request.source_agent,
            "target": request.target_agent,
            "task": request.task_description,
            "scope": request.guardrail_scope.value,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "tokens_used": result.tokens_used,
            "error": result.error,
        }
        try:
            log_path = safe_path(self._log_path)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except (ValueError, OSError) as exc:
            logger.warning("failed to write handoff log: %s", exc)

    def _read_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent handoff log entries."""
        if not self._log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return []
        return entries[-limit:]

    # -- Execution ----------------------------------------------------------

    def _execute_with_timeout(self, fn: Callable[..., dict[str, Any]],
                              context: dict[str, Any],
                              timeout: float) -> HandoffResult:
        """Run fn(context) in a thread with timeout enforcement."""
        result_holder: list[HandoffResult] = []
        exception_holder: list[Exception] = []

        def _run() -> None:
            try:
                data = fn(context)
                if not isinstance(data, dict):
                    data = {"result": data}
                result_holder.append(HandoffResult(
                    success=True, result_data=data,
                ))
            except Exception as exc:
                exception_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        start = _ts()
        thread.start()
        thread.join(timeout=timeout)
        duration_ms = int((_ts() - start) * 1000)

        if thread.is_alive():
            return HandoffResult(
                success=False,
                result_data={},
                error=f"Timeout after {timeout}s",
                duration_ms=duration_ms,
            )

        if exception_holder:
            return HandoffResult(
                success=False,
                result_data={},
                error=str(exception_holder[0]),
                duration_ms=duration_ms,
            )

        if result_holder:
            r = result_holder[0]
            r.duration_ms = duration_ms
            return r

        return HandoffResult(
            success=False,
            result_data={},
            error="No result produced",
            duration_ms=duration_ms,
        )

    def send(self, request: HandoffRequest) -> HandoffResult:
        """Send a handoff request synchronously. Blocks until complete or timeout."""
        # Guardrail check
        violation = _check_guardrails(request)
        if violation:
            result = HandoffResult(
                success=False, result_data={}, error=violation,
                source_agent=request.source_agent,
                target_agent=request.target_agent,
            )
            self._log_handoff(request, result)
            return result

        # GateKeeper check -- block destructive actions unless approved
        action = str(request.context.get("action", "")).lower()
        if action and not self._gate.check_gate(action):
            result = HandoffResult(
                success=False, result_data={},
                error=f"GATE BLOCKED: action '{action}' requires approval",
                source_agent=request.source_agent,
                target_agent=request.target_agent,
            )
            self._log_handoff(request, result)
            return result

        # Lookup target
        with self._lock:
            fn = self._registry.get(request.target_agent)
        if fn is None:
            result = HandoffResult(
                success=False, result_data={},
                error=f"Unknown target agent: {request.target_agent}",
                source_agent=request.source_agent,
                target_agent=request.target_agent,
            )
            self._log_handoff(request, result)
            return result

        # Execute with circuit breaker + timeout
        start = self._trajectory.log_attempt(
            "handoff", source=request.source_agent,
            target=request.target_agent,
        )
        result = HandoffResult(success=False, result_data={}, error="not executed")
        try:
            with self._cb:
                result = self._execute_with_timeout(
                    fn, request.context, request.timeout_seconds,
                )
        except Exception as exc:
            result = HandoffResult(
                success=False, result_data={},
                error=str(exc),
            )

        result.source_agent = request.source_agent
        result.target_agent = request.target_agent

        # Log
        if result.success:
            self._trajectory.log_success("handoff", start=start,
                                         target=request.target_agent)
        else:
            self._trajectory.log_failure("handoff", error=result.error or "",
                                         start=start,
                                         target=request.target_agent)
        self._log_handoff(request, result)

        # Callback (fire-and-forget in background thread)
        if request.callback_on_complete is not None:
            cb = request.callback_on_complete
            threading.Thread(
                target=lambda: cb(result), daemon=True,
            ).start()

        return result

    def send_async(self, request: HandoffRequest,
                   callback: Callable[[HandoffResult], None] | None = None) -> None:
        """Send a handoff request asynchronously.

        If callback is provided, it overrides request.callback_on_complete.
        The handoff runs in a background thread.
        """
        if callback is not None:
            request.callback_on_complete = callback
        threading.Thread(
            target=lambda: self.send(request), daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# HandoffChain
# ---------------------------------------------------------------------------

class HandoffChain:
    """Chain multiple handoffs: A -> B -> C, each receiving previous result.

    Each step gets the previous result merged into its context under
    the key 'previous_result'.
    """

    def __init__(self, router: HandoffRouter) -> None:
        self._router = router
        self._steps: list[HandoffRequest] = []

    def add(self, request: HandoffRequest) -> HandoffChain:
        """Add a step to the chain. Returns self for fluent API."""
        self._steps.append(request)
        return self

    def run(self) -> list[HandoffResult]:
        """Execute the chain sequentially. Stops on first failure."""
        results: list[HandoffResult] = []
        previous_data: dict[str, Any] = {}

        for step in self._steps:
            # Inject previous result into context
            if previous_data:
                step.context["previous_result"] = previous_data

            result = self._router.send(step)
            results.append(result)

            if not result.success:
                logger.warning(
                    "chain stopped at %s -> %s: %s",
                    step.source_agent, step.target_agent, result.error,
                )
                break

            previous_data = result.result_data

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _show_status(router: HandoffRouter) -> None:
    agents = router.list_agents()
    print(f"\n=== Handoff Router Status ===\n")
    print(f"Registered agents: {len(agents)}")
    if agents:
        for name in agents:
            print(f"  - {name}")
    else:
        print("  (none)")
    print()


def _show_history(limit: int = 20) -> None:
    """Show recent handoff history from the audit log."""
    if not HANDOFF_LOG.exists():
        print("\nNo handoff history found.\n")
        return

    entries: list[dict[str, Any]] = []
    with open(HANDOFF_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entries = entries[-limit:]

    print(f"\n=== Handoff History (last {len(entries)}) ===\n")
    if not entries:
        print("  (empty)")
    for e in entries:
        status = "OK" if e.get("success") else "FAIL"
        dur = e.get("duration_ms", 0)
        src = e.get("source", "?")
        tgt = e.get("target", "?")
        task = e.get("task", "")[:60]
        err = e.get("error", "")
        ts = e.get("ts", e.get("timestamp", ""))[:19]
        line = f"  [{ts}] {status:4s} {dur:6d}ms  {src} -> {tgt}  {task}"
        if err:
            line += f"  ERR: {err[:40]}"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handoff -- agent-to-agent handoff protocol",
    )
    parser.add_argument("--status", action="store_true",
                        help="Show registered agents")
    parser.add_argument("--history", nargs="?", const=20, type=int,
                        metavar="N", help="Show last N handoffs (default 20)")
    parser.add_argument("--gates", action="store_true",
                        help="Show configured action gates")
    args = parser.parse_args()

    if not any([args.status, args.history is not None, args.gates]):
        parser.print_help()
        return

    if args.status:
        router = HandoffRouter()
        _show_status(router)

    if args.history is not None:
        _show_history(limit=args.history)

    if args.gates:
        gk = GateKeeper()
        print(gk.status())


if __name__ == "__main__":
    main()
