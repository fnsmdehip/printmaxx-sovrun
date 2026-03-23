#!/usr/bin/env python3
"""
Session Briefing -- Fast briefing at session start.

Reads existing files (no LLM calls) to produce a concise briefing:
1. System changes since last session (git diff summary)
2. Actionable queue from task tracker
3. Agent output summaries
4. Recently updated state files

Must finish in < 30 seconds. No API calls. Pure file reading.

Usage:
    python3 session_briefing.py              # Print briefing to stdout
    python3 session_briefing.py --save       # Also save to output/
    python3 session_briefing.py --json       # JSON output
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

LOGS_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Input files (configure per your project structure)
TASK_TRACKER = Path(os.environ.get(
    "SOVRUN_TASK_TRACKER", PROJECT_ROOT / "data" / "tasks.md"))
AGENT_REPORTS_DIR = Path(os.environ.get(
    "SOVRUN_REPORTS_DIR", PROJECT_ROOT / "output" / "reports"))

OUTPUT_FILE = OUTPUT_DIR / "session_briefing.md"
BRIEFING_STATE = STATE_DIR / "session_briefing_state.json"
LOG_FILE = LOGS_DIR / "session_briefing.log"


def safe_path(target: str | Path) -> Path:
    """Verify path is within project root."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} is outside project root {root}")
    return resolved


def log_to_file(msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def read_file_safe(path: Path, max_lines: int = 0) -> str:
    """Read a file safely, return empty string on failure."""
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_lines > 0:
            lines = text.split("\n")
            return "\n".join(lines[:max_lines])
        return text
    except Exception:
        return ""


def load_state() -> dict[str, Any]:
    """Load last session briefing state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if BRIEFING_STATE.exists():
        try:
            return json.loads(BRIEFING_STATE.read_text())
        except Exception:
            pass
    return {"last_briefing": None, "last_session_ts": None}


def save_state(state: dict[str, Any]) -> None:
    """Save session briefing state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BRIEFING_STATE, "w") as f:
        json.dump(state, f, indent=2)


def get_last_session_ts(state: dict[str, Any]) -> datetime:
    """Get timestamp of last session, default 24h ago."""
    ts_str = state.get("last_session_ts")
    if ts_str:
        try:
            return datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            pass
    return datetime.now() - timedelta(hours=24)


def changes_since_last_session(since: datetime) -> str:
    """Summarize git changes since last session."""
    lines: list[str] = []
    lines.append("## Changes Since Last Session")
    lines.append("")

    try:
        since_str = since.strftime("%Y-%m-%d %H:%M")
        result = subprocess.run(
            ["git", "log", f"--since={since_str}", "--oneline", "--no-merges", "-30"],
            capture_output=True, text=True, timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            commits = result.stdout.strip().split("\n")
            lines.append(f"### Commits ({len(commits)})")
            for c in commits[:15]:
                lines.append(f"- {c}")
            if len(commits) > 15:
                lines.append(f"- ... and {len(commits) - 15} more")
            lines.append("")
        else:
            lines.append("No git commits since last session.")
            lines.append("")
    except Exception as e:
        lines.append(f"Git check failed: {e}")
        lines.append("")

    return "\n".join(lines)


def recent_reports(since: datetime) -> str:
    """List recent agent reports."""
    lines: list[str] = []

    if not AGENT_REPORTS_DIR.exists():
        return ""

    recent: list[str] = []
    try:
        for f in sorted(AGENT_REPORTS_DIR.iterdir(), reverse=True):
            if f.is_file():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime >= since:
                    recent.append(f.name)
    except Exception:
        return ""

    if recent:
        lines.append(f"## Agent Reports ({len(recent)} since last session)")
        lines.append("")
        for r in recent[:15]:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)


def actionable_queue() -> str:
    """Extract actionable items from task tracker."""
    lines: list[str] = []
    lines.append("## Actionable Queue")
    lines.append("")

    tracker = read_file_safe(TASK_TRACKER)
    if not tracker:
        lines.append("No task tracker found.")
        lines.append("")
        return "\n".join(lines)

    tracker_lines = tracker.split("\n")
    tasks: list[str] = []

    for tline in tracker_lines[:100]:
        tline_stripped = tline.strip()
        if tline_stripped.startswith("- ") or tline_stripped.startswith("### "):
            tasks.append(tline_stripped)

    if tasks:
        for t in tasks[:15]:
            lines.append(t)
        lines.append("")
    else:
        lines.append("No pending tasks found.")
        lines.append("")

    return "\n".join(lines)


def build_briefing(save: bool = False, as_json: bool = False) -> str:
    """Build the full session briefing."""
    state = load_state()
    since = get_last_session_ts(state)
    now = datetime.now()

    hours_since_last = (now - since).total_seconds() / 3600

    sections: list[str] = []
    sections.append(f"# SESSION BRIEFING -- {now.strftime('%Y-%m-%d %H:%M')}")
    sections.append(f"Last session: {since.strftime('%Y-%m-%d %H:%M')} ({hours_since_last:.1f}h ago)")
    sections.append("")
    sections.append("---")
    sections.append("")

    sections.append(changes_since_last_session(since))

    reports = recent_reports(since)
    if reports:
        sections.append(reports)

    sections.append(actionable_queue())

    briefing = "\n".join(sections)

    state["last_session_ts"] = now.isoformat()
    state["last_briefing"] = now.isoformat()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(briefing, encoding="utf-8")
        save_state(state)

    if as_json:
        result: dict[str, Any] = {
            "generated": now.isoformat(),
            "last_session": since.isoformat(),
            "hours_since_last": round(hours_since_last, 1),
            "briefing": briefing,
        }
        return json.dumps(result, indent=2)

    return briefing


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Session briefing generator")
    parser.add_argument("--save", action="store_true", help="Save to output/session_briefing.md")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    briefing = build_briefing(save=args.save, as_json=args.json)
    print(briefing)

    log_to_file(f"Briefing generated, save={args.save}")


if __name__ == "__main__":
    main()
