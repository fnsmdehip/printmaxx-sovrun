#!/usr/bin/env python3
"""
Durable -- Crash recovery with deterministic replay for agent executions.

Each agent execution step is logged atomically to a JSONL replay log.
On crash, replay from last completed step without re-executing completed work.

Inspired by Temporal/Vercel DurableAgent patterns but built for
agents that crash at 3 AM and need to resume at 3:01 AM.

Usage:
    from sovrun.core.durable import DurableExecution

    exe = DurableExecution("my-pipeline")

    def fetch_data(prev):
        # prev is initial_input (None by default) for the first step
        return {"rows": 100}

    def process(prev):
        return {"processed": prev["rows"]}

    result = exe.execute([
        ("fetch", fetch_data),
        ("process", process),
    ])
    # result == {"fetch": {"rows": 100}, "process": {"processed": 100}}

CLI:
    sovrun-durable --status                # Show all executions
    sovrun-durable --replay EXECUTION_ID   # Show replay data
    sovrun-durable --clean 7               # Remove logs older than 7 days
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))
LOGS_DIR = Path(os.environ.get("SOVRUN_LOGS_DIR", PROJECT_ROOT / "logs"))
REPLAY_DIR = Path(os.environ.get("SOVRUN_REPLAY_DIR", LOGS_DIR / "replay"))

logger = logging.getLogger("sovrun.durable")


def safe_path(target: str | Path) -> Path:
    """Verify path is within project root. Raises ValueError if not."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} outside {root}")
    return resolved


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ts() -> float:
    return time.time()


def _hash_input(obj: Any) -> str:
    """Deterministic hash of a step's input for replay matching."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _safe_serialize(obj: Any) -> Any:
    """Best-effort JSON serialization."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    return str(obj)


# ---------------------------------------------------------------------------
# StepLog
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    """Record of a single completed execution step."""
    step_id: str
    step_name: str
    input_hash: str
    output: Any
    timestamp: str = ""
    duration_ms: int = 0
    tokens_used: int = 0
    status: str = "complete"
    error: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output"] = _safe_serialize(d["output"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepLog:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DurableExecution
# ---------------------------------------------------------------------------

class DurableExecution:
    """Wraps a sequence of callables with deterministic replay on crash recovery.

    Each step result is logged atomically to a JSONL file. On re-run,
    completed steps are loaded from the replay log and skipped.

    Args:
        execution_id: unique identifier for this execution run.
        replay_dir: override default replay log directory.
    """

    def __init__(self, execution_id: str,
                 replay_dir: Path | None = None) -> None:
        self.execution_id = execution_id
        self._dir = replay_dir or REPLAY_DIR
        _ensure_dir(self._dir)
        self._log_path = safe_path(self._dir / f"{execution_id}.jsonl")
        self._completed: dict[str, StepLog] = {}
        self._load_replay()

    def _load_replay(self) -> None:
        """Load previously completed steps from the replay log."""
        if not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("status") == "complete":
                            step = StepLog.from_dict(entry)
                            self._completed[step.step_id] = step
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

        if self._completed:
            logger.info("replay: loaded %d completed steps for %s",
                        len(self._completed), self.execution_id)

    def _append_log(self, step_log: StepLog) -> None:
        """Append a step to the JSONL replay log.

        Uses write-then-flush to minimize partial-write risk without
        the overhead of temp-file-then-copy.
        """
        entry = json.dumps(step_log.to_dict(), default=str) + "\n"
        _ensure_dir(self._log_path.parent)
        with open(self._log_path, "a") as f:
            f.write(entry)
            f.flush()
            os.fsync(f.fileno())

    def replay(self) -> dict[str, StepLog]:
        """Return all completed steps from the replay log without re-executing."""
        self._load_replay()
        return dict(self._completed)

    def execute(self, steps: list[tuple[str, Callable[..., Any]]],
                initial_input: Any = None) -> dict[str, Any]:
        """Run steps sequentially, skipping already-completed ones.

        Each step callable receives the previous step's output (or
        initial_input for the first step). Returns a dict of
        step_name -> output for all steps.

        Args:
            steps: list of (step_name, callable) tuples.
            initial_input: input passed to the first step.

        Returns:
            dict mapping step_name to its output.
        """
        results: dict[str, Any] = {}
        prev_output: Any = initial_input

        # Pre-populate results from completed steps, stopping at
        # the first missing or invalidated step to avoid using stale
        # outputs from later steps that may need re-execution.
        for step_name, _ in steps:
            step_id = f"{self.execution_id}:{step_name}"
            if step_id in self._completed:
                input_hash = _hash_input(prev_output)
                cached = self._completed[step_id]
                if cached.input_hash == input_hash:
                    prev_output = cached.output
                    results[step_name] = prev_output
                else:
                    break
            else:
                break

        for step_name, fn in steps:
            step_id = f"{self.execution_id}:{step_name}"

            # Skip if already completed with matching input
            input_hash = _hash_input(prev_output)
            if step_id in self._completed:
                cached = self._completed[step_id]
                if cached.input_hash == input_hash:
                    logger.info("replay: skipping completed step '%s'",
                                step_name)
                    prev_output = cached.output
                    results[step_name] = prev_output
                    continue
                else:
                    logger.info(
                        "replay: input changed for '%s', re-executing",
                        step_name)

            # Execute the step
            start = _ts()
            try:
                output = fn(prev_output)
                duration_ms = int((_ts() - start) * 1000)

                step_log = StepLog(
                    step_id=step_id,
                    step_name=step_name,
                    input_hash=input_hash,
                    output=output,
                    duration_ms=duration_ms,
                    status="complete",
                )
                self._append_log(step_log)
                self._completed[step_id] = step_log
                prev_output = output
                results[step_name] = output
                logger.info("durable: step '%s' complete (%dms)",
                            step_name, duration_ms)

            except Exception as exc:
                duration_ms = int((_ts() - start) * 1000)
                step_log = StepLog(
                    step_id=step_id,
                    step_name=step_name,
                    input_hash=input_hash,
                    output=None,
                    duration_ms=duration_ms,
                    status="failed",
                    error=str(exc),
                )
                self._append_log(step_log)
                logger.error("durable: step '%s' failed: %s",
                             step_name, exc)
                raise

        return results

    def status(self) -> dict[str, Any]:
        """Return execution status summary."""
        completed = len(self._completed)
        total_duration = sum(s.duration_ms for s in self._completed.values())
        total_tokens = sum(s.tokens_used for s in self._completed.values())
        return {
            "execution_id": self.execution_id,
            "completed_steps": completed,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "log_file": str(self._log_path),
            "steps": {
                sid: {
                    "name": s.step_name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "timestamp": s.timestamp,
                }
                for sid, s in self._completed.items()
            },
        }


# ---------------------------------------------------------------------------
# Index: list all executions
# ---------------------------------------------------------------------------

def list_executions(replay_dir: Path | None = None) -> list[dict[str, Any]]:
    """List all execution replay logs with basic stats."""
    d = replay_dir or REPLAY_DIR
    if not d.exists():
        return []

    results: list[dict[str, Any]] = []
    for log_file in sorted(d.glob("*.jsonl")):
        execution_id = log_file.stem
        step_count = 0
        last_ts = ""
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("status") == "complete":
                            step_count += 1
                            last_ts = entry.get("timestamp", last_ts)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

        size_kb = log_file.stat().st_size // 1024
        results.append({
            "execution_id": execution_id,
            "completed_steps": step_count,
            "last_activity": last_ts,
            "log_size_kb": size_kb,
            "log_file": str(log_file),
        })

    return results


def clean_old_logs(days: int, replay_dir: Path | None = None) -> int:
    """Remove replay logs older than N days. Returns count of removed files."""
    d = replay_dir or REPLAY_DIR
    if not d.exists():
        return 0

    cutoff = _ts() - (days * 86400)
    removed = 0

    for log_file in d.glob("*.jsonl"):
        try:
            safe_path(log_file)
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                removed += 1
                logger.info("cleaned old replay log: %s", log_file.name)
        except (OSError, ValueError):
            continue

    return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Durable -- crash recovery with deterministic replay")
    parser.add_argument("--status", action="store_true",
                        help="Show all executions")
    parser.add_argument("--replay", type=str, metavar="ID",
                        help="Show replay data for an execution")
    parser.add_argument("--clean", type=int, metavar="DAYS",
                        help="Remove replay logs older than N days")
    args = parser.parse_args()

    if not any([args.status, args.replay, args.clean is not None]):
        parser.print_help()
        return

    if args.status:
        executions = list_executions()
        print(f"\n=== Durable Executions ({len(executions)}) ===\n")
        if not executions:
            print("  No executions found.")
        for ex in executions:
            print(f"  {ex['execution_id']}")
            print(f"    Steps: {ex['completed_steps']}  "
                  f"Size: {ex['log_size_kb']}KB  "
                  f"Last: {ex['last_activity'][:19]}")
        print()

    if args.replay:
        exe = DurableExecution(args.replay)
        info = exe.status()
        print(f"\n=== Replay: {info['execution_id']} ===\n")
        print(f"Completed steps: {info['completed_steps']}")
        print(f"Total duration: {info['total_duration_ms']}ms")
        print(f"Log file: {info['log_file']}")
        print()
        for sid, step_info in info["steps"].items():
            print(f"  [{step_info['status'].upper():8s}] "
                  f"{step_info['name']}  "
                  f"{step_info['duration_ms']}ms  "
                  f"{step_info['timestamp'][:19]}")
        print()

    if args.clean is not None:
        removed = clean_old_logs(args.clean)
        print(f"Cleaned {removed} replay logs older than {args.clean} days.")


if __name__ == "__main__":
    main()
