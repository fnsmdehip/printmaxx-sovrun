#!/usr/bin/env python3
"""
Orchestration -- Formal execution patterns for autonomous agent pipelines.

Provides composable orchestration primitives:
- AgentStep: unit of work with dependencies, timeout, retry
- SequentialPipeline: ordered step execution with checkpointing
- ParallelExecutor: concurrent step execution respecting dependency graph
- LoopRunner: repeated execution until condition met
- DAGOrchestrator: full DAG-based execution with parallel fanout, checkpointing, resume

Inspired by Google ADK's SequentialAgent/ParallelAgent/LoopAgent but built
for real autonomous systems that crash, resume, and run overnight.

Stdlib only. No external dependencies.

Usage:
    from sovrun.core.orchestration import DAGOrchestrator, AgentStep, step

    # Functional API
    @step(name="fetch_data", timeout_seconds=30)
    def fetch_data():
        return {"rows": 100}

    @step(name="process", depends_on=["fetch_data"])
    def process(prev_result):
        return {"processed": prev_result["rows"]}

    dag = DAGOrchestrator(steps=[fetch_data, process])
    dag.run()

    # CLI
    sovrun-orchestrate --status
    sovrun-orchestrate --resume
    sovrun-orchestrate --visualize
    sovrun-orchestrate --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

STATE_DIR = Path(os.environ.get("SOVRUN_STATE_DIR", PROJECT_ROOT / "state"))
LOGS_DIR = Path(os.environ.get("SOVRUN_LOGS_DIR", PROJECT_ROOT / "logs"))

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = STATE_DIR / "orchestration_checkpoint.json"
AUDIT_LOG = LOGS_DIR / "orchestration.jsonl"


def safe_path(target: str | Path) -> Path:
    """Verify path is within project root."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} outside {root}")
    return resolved


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ts() -> float:
    return time.time()


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [ORCH] [{level}] {msg}")


# ---------------------------------------------------------------------------
# Step status and data
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class AgentStep:
    """A single unit of work in an orchestration pipeline.

    Args:
        name: unique identifier for this step
        fn: callable to execute. Receives a dict of completed step results
            keyed by step name. Returns any value.
        depends_on: names of steps that must complete before this one runs
        timeout_seconds: max execution time (0 = no limit)
        retry_count: how many times to retry on failure (0 = no retry)
        result: stored result after execution
        status: current execution status
        error: error message if failed
        duration_ms: execution duration in milliseconds
    """
    name: str
    fn: Callable[..., Any] | None = None
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 0
    retry_count: int = 0
    result: Any = None
    status: StepStatus = StepStatus.PENDING
    error: str = ""
    duration_ms: int = 0

    def reset(self) -> None:
        """Reset step to pending state."""
        self.result = None
        self.status = StepStatus.PENDING
        self.error = ""
        self.duration_ms = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize step state (without fn) for checkpointing."""
        return {
            "name": self.name,
            "depends_on": self.depends_on,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "error": self.error,
            "duration_ms": self.duration_ms,
            # result serialized best-effort
            "result": _safe_serialize(self.result),
        }


def _safe_serialize(obj: Any) -> Any:
    """Best-effort JSON serialization of step results."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    return str(obj)


# ---------------------------------------------------------------------------
# @step decorator
# ---------------------------------------------------------------------------

def step(name: str, depends_on: list[str] | None = None,
         timeout_seconds: int = 0, retry_count: int = 0) -> Callable:
    """Decorator that marks a function as an orchestration step.

    The decorated function becomes an AgentStep instance with .fn set
    to the original function. Pass it directly to DAGOrchestrator.

    Usage:
        @step(name="fetch", timeout_seconds=30)
        def fetch(results):
            return requests.get(url).json()
    """
    def decorator(fn: Callable) -> AgentStep:
        return AgentStep(
            name=name,
            fn=fn,
            depends_on=depends_on or [],
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
        )
    return decorator


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

_audit_lock = threading.Lock()


def _audit(event: str, step_name: str = "", **extra: Any) -> None:
    """Append an entry to the JSONL audit log."""
    entry = {
        "ts": _now_iso(),
        "event": event,
        "step": step_name,
        **{k: _safe_serialize(v) for k, v in extra.items()},
    }
    with _audit_lock:
        try:
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Checkpoint (atomic writes)
# ---------------------------------------------------------------------------

def _save_checkpoint(steps: list[AgentStep], meta: dict[str, Any] | None = None) -> None:
    """Atomic checkpoint write: write to tmp file then rename."""
    data = {
        "ts": _now_iso(),
        "meta": meta or {},
        "steps": {s.name: s.to_dict() for s in steps},
    }
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(CHECKPOINT_FILE.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(CHECKPOINT_FILE))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_checkpoint() -> dict[str, Any] | None:
    """Load checkpoint if it exists and is valid JSON."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        return json.loads(CHECKPOINT_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _execute_step(s: AgentStep, results: dict[str, Any],
                  timeout: int = 0) -> None:
    """Execute a single step with retry and timeout.

    Mutates the step in-place: sets status, result, error, duration_ms.
    """
    if s.fn is None:
        s.status = StepStatus.SKIPPED
        s.error = "no callable"
        return

    attempts = max(1, s.retry_count + 1)
    effective_timeout = timeout or s.timeout_seconds or 0

    for attempt in range(attempts):
        s.status = StepStatus.RUNNING
        start = _ts()
        try:
            if effective_timeout > 0:
                result_container: list[Any] = []
                exc_container: list[BaseException] = []

                def _run() -> None:
                    try:
                        if s.fn is None:
                            raise ValueError(f"Step '{s.name}' has no callable")
                        result_container.append(s.fn(results))
                    except Exception as e:
                        exc_container.append(e)

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                t.join(timeout=effective_timeout)
                if t.is_alive():
                    raise TimeoutError(
                        f"step '{s.name}' exceeded {effective_timeout}s timeout")
                if exc_container:
                    raise exc_container[0]
                s.result = result_container[0] if result_container else None
            else:
                s.result = s.fn(results)

            s.status = StepStatus.COMPLETE
            s.duration_ms = int((_ts() - start) * 1000)
            s.error = ""
            return

        except Exception as e:
            s.duration_ms = int((_ts() - start) * 1000)
            s.error = f"{type(e).__name__}: {e}"
            if attempt < attempts - 1:
                backoff = min(2 ** attempt, 30)
                log(f"Retry {attempt + 1}/{attempts} for '{s.name}': {e} "
                    f"(backoff {backoff}s)", "WARN")
                time.sleep(backoff)

    s.status = StepStatus.FAILED


# ---------------------------------------------------------------------------
# SequentialPipeline
# ---------------------------------------------------------------------------

class SequentialPipeline:
    """Execute steps in order, passing each result to the next.

    Args:
        steps: ordered list of AgentSteps
        fail_fast: if True, stop on first failure. If False, continue and
                   skip steps that depend on failed steps.
    """

    def __init__(self, steps: list[AgentStep],
                 fail_fast: bool = True) -> None:
        self.steps = steps
        self.fail_fast = fail_fast

    def run(self, initial_results: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all steps sequentially. Returns dict of step_name -> result."""
        results: dict[str, Any] = dict(initial_results or {})
        _audit("sequential_start", step_name="",
               steps=[s.name for s in self.steps])

        for s in self.steps:
            log(f"Sequential: running '{s.name}'")
            _execute_step(s, results)
            _save_checkpoint(self.steps)

            if s.status == StepStatus.COMPLETE:
                results[s.name] = s.result
                _audit("step_complete", s.name, duration_ms=s.duration_ms)
            else:
                _audit("step_failed", s.name, error=s.error)
                if self.fail_fast:
                    log(f"Sequential: fail_fast on '{s.name}': {s.error}",
                        "ERROR")
                    break
                log(f"Sequential: '{s.name}' failed, continuing: {s.error}",
                    "WARN")

        _audit("sequential_end",
               completed=sum(1 for s in self.steps
                             if s.status == StepStatus.COMPLETE),
               total=len(self.steps))
        return results


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Execute independent steps concurrently using threads.

    Respects dependency graph: only runs steps whose dependencies
    are all COMPLETE. Waits until all steps finish or overall timeout.

    Args:
        steps: list of AgentSteps (may have depends_on)
        max_workers: thread pool size
        timeout_seconds: overall timeout (0 = no limit)
    """

    def __init__(self, steps: list[AgentStep], max_workers: int = 4,
                 timeout_seconds: int = 0) -> None:
        self.steps = steps
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds

    def run(self, initial_results: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run steps respecting dependencies. Returns step_name -> result."""
        results: dict[str, Any] = dict(initial_results or {})
        by_name: dict[str, AgentStep] = {s.name: s for s in self.steps}
        lock = threading.Lock()
        deadline = _ts() + self.timeout_seconds if self.timeout_seconds else 0

        _audit("parallel_start",
               steps=[s.name for s in self.steps],
               max_workers=self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: dict[str, Future] = {}
            submitted: set[str] = set()

            def _deps_met(s: AgentStep) -> bool:
                for dep in s.depends_on:
                    if dep in by_name and by_name[dep].status != StepStatus.COMPLETE:
                        return False
                    # Dep not in our step list = assume external, check results
                    if dep not in by_name and dep not in results:
                        return False
                return True

            def _deps_failed(s: AgentStep) -> bool:
                for dep in s.depends_on:
                    if dep in by_name and by_name[dep].status == StepStatus.FAILED:
                        return True
                return False

            def _run_step_and_collect(s: AgentStep) -> None:
                """Execute step then immediately update shared results."""
                # Take a snapshot of results for this step's input
                with lock:
                    results_snapshot = dict(results)
                _execute_step(s, results_snapshot)
                if s.status == StepStatus.COMPLETE:
                    with lock:
                        results[s.name] = s.result

            def _submit_ready() -> int:
                count = 0
                for s in self.steps:
                    if s.name in submitted:
                        continue
                    if s.status != StepStatus.PENDING:
                        continue
                    if _deps_failed(s):
                        s.status = StepStatus.SKIPPED
                        s.error = "dependency failed"
                        _audit("step_skipped", s.name, error=s.error)
                        submitted.add(s.name)
                        continue
                    if _deps_met(s):
                        log(f"Parallel: submitting '{s.name}'")
                        fut = pool.submit(_run_step_and_collect, s)
                        futures[s.name] = fut
                        submitted.add(s.name)
                        count += 1
                return count

            _submit_ready()

            while len(submitted) < len(self.steps):
                if deadline and _ts() > deadline:
                    log("Parallel: overall timeout reached", "WARN")
                    for s in self.steps:
                        if s.status == StepStatus.PENDING:
                            s.status = StepStatus.SKIPPED
                            s.error = "overall timeout"
                    break

                # Wait for any future to complete
                time.sleep(0.1)

                # Collect completed futures
                for name, fut in list(futures.items()):
                    if fut.done():
                        s = by_name[name]
                        if s.status == StepStatus.COMPLETE:
                            _audit("step_complete", s.name,
                                   duration_ms=s.duration_ms)
                        else:
                            _audit("step_failed", s.name, error=s.error)
                        _save_checkpoint(self.steps)
                        del futures[name]

                _submit_ready()

            # Wait for remaining futures
            for name, fut in list(futures.items()):
                remaining = (deadline - _ts()) if deadline else None
                try:
                    fut.result(timeout=max(0.1, remaining) if remaining else None)
                except Exception:
                    pass
                _save_checkpoint(self.steps)

        _audit("parallel_end",
               completed=sum(1 for s in self.steps
                             if s.status == StepStatus.COMPLETE),
               total=len(self.steps))
        return results


# ---------------------------------------------------------------------------
# LoopRunner
# ---------------------------------------------------------------------------

class LoopRunner:
    """Execute a step repeatedly until a condition is met.

    Args:
        step_to_run: the AgentStep to execute each iteration
        max_iterations: hard cap on iterations (0 = unlimited)
        stop_condition: callable(result) -> bool; stops when True
        backoff_seconds: delay between iterations
    """

    def __init__(self, step_to_run: AgentStep, max_iterations: int = 10,
                 stop_condition: Callable[[Any], bool] | None = None,
                 backoff_seconds: float = 1.0) -> None:
        self.step = step_to_run
        self.max_iterations = max_iterations
        self.stop_condition = stop_condition or (lambda _: False)
        self.backoff_seconds = backoff_seconds

    def run(self, initial_results: dict[str, Any] | None = None) -> list[Any]:
        """Run the step in a loop. Returns list of results from each iteration."""
        results_list: list[Any] = []
        results: dict[str, Any] = dict(initial_results or {})

        _audit("loop_start", self.step.name,
               max_iterations=self.max_iterations)

        iteration = 0
        while True:
            if self.max_iterations and iteration >= self.max_iterations:
                log(f"Loop: '{self.step.name}' hit max iterations "
                    f"({self.max_iterations})")
                break

            iteration += 1
            self.step.reset()
            log(f"Loop: '{self.step.name}' iteration {iteration}")
            _execute_step(self.step, results)

            if self.step.status == StepStatus.COMPLETE:
                results_list.append(self.step.result)
                results[self.step.name] = self.step.result
                _audit("loop_iteration", self.step.name,
                       iteration=iteration, duration_ms=self.step.duration_ms)

                if self.stop_condition(self.step.result):
                    log(f"Loop: '{self.step.name}' stop condition met "
                        f"at iteration {iteration}")
                    break
            else:
                _audit("loop_iteration_failed", self.step.name,
                       iteration=iteration, error=self.step.error)
                log(f"Loop: '{self.step.name}' failed at iteration "
                    f"{iteration}: {self.step.error}", "WARN")
                break

            if self.backoff_seconds > 0 and (
                    not self.max_iterations or iteration < self.max_iterations):
                time.sleep(self.backoff_seconds)

        _audit("loop_end", self.step.name,
               iterations=iteration, results_count=len(results_list))
        return results_list


# ---------------------------------------------------------------------------
# DAGOrchestrator
# ---------------------------------------------------------------------------

class DAGOrchestrator:
    """Full DAG-based orchestrator.

    Takes a list of AgentSteps with dependency declarations. Builds a DAG,
    validates it (no cycles), and executes steps respecting dependencies
    with parallel fanout where possible.

    Features:
        - Topological sort for execution order validation
        - Parallel execution of independent steps via ParallelExecutor
        - Checkpoint after each step (atomic JSON write)
        - Resume from checkpoint on crash
        - Timeout per step and overall
        - JSONL audit trail
        - ASCII DAG visualization

    Args:
        steps: list of AgentSteps (or @step-decorated functions)
        max_workers: thread pool size for parallel execution
        timeout_seconds: overall DAG timeout (0 = no limit)
        checkpoint_file: override default checkpoint path
    """

    def __init__(self, steps: list[AgentStep], max_workers: int = 4,
                 timeout_seconds: int = 0,
                 checkpoint_file: Path | None = None) -> None:
        self.steps = steps
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self._checkpoint_file = checkpoint_file or CHECKPOINT_FILE
        self._by_name: dict[str, AgentStep] = {s.name: s for s in steps}
        self._validate()

    def _validate(self) -> None:
        """Validate DAG: check for missing deps and cycles."""
        names = set(self._by_name.keys())
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in names:
                    raise ValueError(
                        f"Step '{s.name}' depends on '{dep}' which "
                        f"does not exist in the step list")
        # Cycle detection via topological sort
        self._topo_sort()

    def _topo_sort(self) -> list[str]:
        """Kahn's algorithm for topological sort. Raises on cycle."""
        in_degree: dict[str, int] = {s.name: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.name: [] for s in self.steps}

        for s in self.steps:
            for dep in s.depends_on:
                adjacency[dep].append(s.name)
                in_degree[s.name] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.steps):
            remaining = set(self._by_name.keys()) - set(order)
            raise ValueError(f"Cycle detected involving: {remaining}")

        return order

    def _restore_from_checkpoint(self) -> bool:
        """Restore step states from checkpoint. Returns True if restored."""
        ckpt = _load_checkpoint()
        if not ckpt:
            return False

        step_data = ckpt.get("steps", {})
        restored = 0
        for s in self.steps:
            if s.name in step_data:
                saved = step_data[s.name]
                saved_status = saved.get("status", "PENDING")
                if saved_status == "COMPLETE":
                    s.status = StepStatus.COMPLETE
                    s.result = saved.get("result")
                    s.error = saved.get("error", "")
                    s.duration_ms = saved.get("duration_ms", 0)
                    restored += 1

        if restored:
            log(f"Restored {restored} completed steps from checkpoint")
            _audit("checkpoint_restored", restored=restored,
                   checkpoint_ts=ckpt.get("ts", ""))
        return restored > 0

    def run(self, resume: bool = False) -> dict[str, Any]:
        """Execute the full DAG.

        Args:
            resume: if True, restore completed steps from checkpoint
                    and skip them.

        Returns:
            dict of step_name -> result for all completed steps.
        """
        if resume:
            self._restore_from_checkpoint()
        else:
            for s in self.steps:
                s.reset()

        results: dict[str, Any] = {}
        # Pre-populate results from already-completed steps
        for s in self.steps:
            if s.status == StepStatus.COMPLETE:
                results[s.name] = s.result

        _audit("dag_start",
               steps=[s.name for s in self.steps],
               resume=resume,
               pre_completed=len(results))

        deadline = _ts() + self.timeout_seconds if self.timeout_seconds else 0

        # Use ParallelExecutor which already handles dependency graph
        pending = [s for s in self.steps if s.status == StepStatus.PENDING]
        if pending:
            executor = ParallelExecutor(
                steps=pending,
                max_workers=self.max_workers,
                timeout_seconds=self.timeout_seconds,
            )
            new_results = executor.run(initial_results=results)
            results.update(new_results)

        _save_checkpoint(self.steps, meta={"mode": "dag", "resume": resume})

        completed = sum(1 for s in self.steps
                        if s.status == StepStatus.COMPLETE)
        failed = sum(1 for s in self.steps
                     if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps
                      if s.status == StepStatus.SKIPPED)

        _audit("dag_end", completed=completed, failed=failed,
               skipped=skipped, total=len(self.steps))

        log(f"DAG complete: {completed} done, {failed} failed, "
            f"{skipped} skipped out of {len(self.steps)}")

        return results

    def status(self) -> str:
        """Return formatted status string for all steps."""
        lines = ["\n=== DAG Orchestration Status ===\n"]

        # Try loading checkpoint
        ckpt = _load_checkpoint()
        if ckpt:
            lines.append(f"Last checkpoint: {ckpt.get('ts', '?')}")
            step_data = ckpt.get("steps", {})
            lines.append(f"Steps in checkpoint: {len(step_data)}\n")

            status_icons = {
                "COMPLETE": "+", "FAILED": "X",
                "RUNNING": ">", "SKIPPED": "-", "PENDING": ".",
            }

            for name, data in step_data.items():
                st = data.get("status", "PENDING")
                icon = status_icons.get(st, "?")
                dur = data.get("duration_ms", 0)
                err = data.get("error", "")
                dep = data.get("depends_on", [])
                dep_str = f" (after: {', '.join(dep)})" if dep else ""
                dur_str = f" [{dur}ms]" if dur else ""
                err_str = f" ERR: {err}" if err else ""
                lines.append(f"  [{icon}] {name}{dep_str}{dur_str}{err_str}")
        else:
            lines.append("No checkpoint found.")

        lines.append("")
        return "\n".join(lines)

    def visualize(self) -> str:
        """ASCII art DAG visualization showing dependencies and status."""
        order = self._topo_sort()
        lines = ["\n=== DAG Visualization ===\n"]

        # Load checkpoint for status info
        ckpt = _load_checkpoint()
        step_statuses: dict[str, str] = {}
        if ckpt:
            for name, data in ckpt.get("steps", {}).items():
                step_statuses[name] = data.get("status", "PENDING")

        # Compute depth (longest path from root) for indentation
        depths: dict[str, int] = {}
        for name in order:
            s = self._by_name[name]
            if not s.depends_on:
                depths[name] = 0
            else:
                depths[name] = max(depths.get(d, 0) for d in s.depends_on) + 1

        max_depth = max(depths.values()) if depths else 0

        status_markers = {
            "COMPLETE": "[+]", "FAILED": "[X]",
            "RUNNING": "[>]", "SKIPPED": "[-]", "PENDING": "[ ]",
        }

        # Group by depth
        for depth in range(max_depth + 1):
            at_depth = [n for n in order if depths[n] == depth]
            indent = "  " * depth
            connector = "|-- " if depth > 0 else ""

            for name in at_depth:
                s = self._by_name[name]
                st = step_statuses.get(name, s.status.value
                                       if isinstance(s.status, StepStatus)
                                       else s.status)
                marker = status_markers.get(st, "[?]")
                dep_str = ""
                if s.depends_on:
                    dep_str = f" <- {', '.join(s.depends_on)}"
                lines.append(f"{indent}{connector}{marker} {name}{dep_str}")

        lines.append("")
        lines.append("Legend: [+] complete  [X] failed  [>] running  "
                      "[-] skipped  [ ] pending")
        lines.append("")
        return "\n".join(lines)

    def dry_run(self) -> str:
        """Show execution plan without running anything."""
        order = self._topo_sort()
        lines = ["\n=== Dry Run: Execution Plan ===\n"]

        # Group by wave (steps at same depth can run in parallel)
        depths: dict[str, int] = {}
        for name in order:
            s = self._by_name[name]
            if not s.depends_on:
                depths[name] = 0
            else:
                depths[name] = max(depths.get(d, 0) for d in s.depends_on) + 1

        max_depth = max(depths.values()) if depths else 0

        for wave in range(max_depth + 1):
            at_wave = [n for n in order if depths[n] == wave]
            parallel = len(at_wave) > 1
            mode = "PARALLEL" if parallel else "SEQUENTIAL"
            lines.append(f"Wave {wave + 1} ({mode}):")
            for name in at_wave:
                s = self._by_name[name]
                timeout_str = (f", timeout={s.timeout_seconds}s"
                               if s.timeout_seconds else "")
                retry_str = (f", retries={s.retry_count}"
                             if s.retry_count else "")
                dep_str = (f", after=[{', '.join(s.depends_on)}]"
                           if s.depends_on else "")
                lines.append(f"  - {name}{dep_str}{timeout_str}{retry_str}")
            lines.append("")

        lines.append(f"Total: {len(self.steps)} steps in "
                      f"{max_depth + 1} waves")
        lines.append(f"Max parallel: {self.max_workers} workers")
        if self.timeout_seconds:
            lines.append(f"Overall timeout: {self.timeout_seconds}s")
        lines.append("")
        return "\n".join(lines)

    def export_mermaid(self) -> str:
        """Generate Mermaid flowchart syntax for the DAG.

        Color-codes nodes by status:
            green=COMPLETE, yellow=RUNNING, gray=PENDING,
            red=FAILED, strikethrough=SKIPPED
        """
        order = self._topo_sort()

        # Load checkpoint for live status
        ckpt = _load_checkpoint()
        step_statuses: dict[str, str] = {}
        if ckpt:
            for name, data in ckpt.get("steps", {}).items():
                step_statuses[name] = data.get("status", "PENDING")

        style_map: dict[str, str] = {
            "COMPLETE": "fill:#22c55e,color:#fff",
            "RUNNING": "fill:#eab308,color:#000",
            "PENDING": "fill:#6b7280,color:#fff",
            "FAILED": "fill:#ef4444,color:#fff",
            "SKIPPED": "fill:#9ca3af,color:#fff,stroke-dasharray:5",
        }

        lines: list[str] = ["graph TD"]

        # Define nodes
        for name in order:
            s = self._by_name[name]
            st = step_statuses.get(name, s.status.value
                                   if isinstance(s.status, StepStatus)
                                   else str(s.status))
            label = f"{name}"
            if s.timeout_seconds:
                label += f"\\n({s.timeout_seconds}s)"
            # Use rounded rectangle for all nodes
            lines.append(f"    {name}[\"{label}\"]")

        # Define edges
        for name in order:
            s = self._by_name[name]
            for dep in s.depends_on:
                lines.append(f"    {dep} --> {name}")

        # Apply styles
        for name in order:
            st = step_statuses.get(
                name,
                self._by_name[name].status.value
                if isinstance(self._by_name[name].status, StepStatus)
                else str(self._by_name[name].status),
            )
            style = style_map.get(st, style_map["PENDING"])
            lines.append(f"    style {name} {style}")

        return "\n".join(lines)

    def export_html_graph(self) -> str:
        """Generate a standalone HTML file with Mermaid DAG visualization.

        Returns the file path of the generated HTML.
        """
        mermaid_src = self.export_mermaid()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DAG Visualization</title>
<style>
  body {{ font-family: 'SF Mono', monospace; background: #0a0a0a; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 18px; margin-bottom: 16px; }}
  .mermaid {{ background: #151515; border-radius: 8px; padding: 24px; }}
  .legend {{ margin-top: 16px; font-size: 13px; color: #888; }}
  .legend span {{ margin-right: 16px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>DAG Orchestration Graph</h1>
<div class="mermaid">
{mermaid_src}
</div>
<div class="legend">
  <span><span class="dot" style="background:#22c55e;"></span>Complete</span>
  <span><span class="dot" style="background:#eab308;"></span>Running</span>
  <span><span class="dot" style="background:#6b7280;"></span>Pending</span>
  <span><span class="dot" style="background:#ef4444;"></span>Failed</span>
  <span><span class="dot" style="background:#9ca3af;"></span>Skipped</span>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
</script>
</body>
</html>"""

        html_path = safe_path(STATE_DIR / "dag_graph.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w") as f:
            f.write(html_content)

        return str(html_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestration -- DAG-based step execution with "
                    "checkpointing and resume")
    parser.add_argument("--status", action="store_true",
                        help="Show current DAG state from checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--visualize", action="store_true",
                        help="ASCII art DAG visualization")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show execution plan without running")
    parser.add_argument("--mermaid", action="store_true",
                        help="Output Mermaid flowchart syntax for the DAG")
    args = parser.parse_args()

    if not any([args.status, args.resume, args.visualize, args.dry_run,
                args.mermaid]):
        parser.print_help()
        return

    # For CLI status/visualize, reconstruct steps from checkpoint
    ckpt = _load_checkpoint()
    if not ckpt and not args.dry_run:
        print("No checkpoint found. Run a DAG first to generate state.")
        return

    if ckpt:
        # Reconstruct steps from checkpoint (no callables)
        steps = []
        for name, data in ckpt.get("steps", {}).items():
            s = AgentStep(
                name=name,
                depends_on=data.get("depends_on", []),
                timeout_seconds=data.get("timeout_seconds", 0),
                retry_count=data.get("retry_count", 0),
            )
            s.status = StepStatus(data.get("status", "PENDING"))
            s.result = data.get("result")
            s.error = data.get("error", "")
            s.duration_ms = data.get("duration_ms", 0)
            steps.append(s)

        dag = DAGOrchestrator(steps=steps)

        if args.status:
            print(dag.status())
        if args.visualize:
            print(dag.visualize())
        if args.dry_run:
            print(dag.dry_run())
        if args.mermaid:
            print(dag.export_mermaid())
            html_path = dag.export_html_graph()
            print(f"\nHTML graph written to: {html_path}")
        if args.resume:
            print("Resume requires callable steps. Use DAGOrchestrator "
                  "programmatically with resume=True.")
    else:
        print("No checkpoint data available.")


if __name__ == "__main__":
    main()
