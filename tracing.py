#!/usr/bin/env python3
"""
Tracing -- Agent observability with structured event collection.

Collects events for every agent action: LLM calls, tool use, handoffs,
errors, decisions. Exports timelines as JSON or standalone HTML files.

Usage:
    from sovrun.core.tracing import Tracer, TraceEvent

    tracer = Tracer()
    tracer.start_session("planner")

    tracer.record(TraceEvent(
        event_type="llm_call",
        agent_name="planner",
        input_summary="Analyze task",
        output_summary="Plan generated",
        tokens_in=500,
        tokens_out=200,
    ))

    tracer.end_session()
    html = tracer.export_html(tracer.current_trace_id)

CLI:
    sovrun-trace --trace ID       # Show trace details
    sovrun-trace --agents         # Per-agent summary
    sovrun-trace --cost           # Cost report
    sovrun-trace --html ID        # Generate HTML timeline
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))
LOGS_DIR = Path(os.environ.get("SOVRUN_LOGS_DIR", PROJECT_ROOT / "logs"))
TRACES_DIR = Path(os.environ.get("SOVRUN_TRACES_DIR", LOGS_DIR / "traces"))

# Default cost rates ($/1K tokens). Override via Tracer(cost_rates={...}).
# These are approximate -- update for your provider's current pricing.
DEFAULT_COST_RATES: dict[str, dict[str, float]] = {
    "opus": {"input": 0.015, "output": 0.075},
    "sonnet": {"input": 0.003, "output": 0.015},
    "haiku": {"input": 0.00025, "output": 0.00125},
    "default": {"input": 0.003, "output": 0.015},
}

logger = logging.getLogger("sovrun.tracing")


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


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """A single recorded event in an agent trace."""
    event_type: str  # llm_call, tool_use, handoff, error, decision, custom
    agent_name: str
    timestamp: str = ""
    duration_ms: int = 0
    input_summary: str = ""
    output_summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    event_id: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _now_iso()
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# TraceSession
# ---------------------------------------------------------------------------

@dataclass
class TraceSession:
    """Groups events for a single agent session."""
    trace_id: str
    agent_name: str
    started_at: str = ""
    ended_at: str = ""
    events: list[TraceEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = _now_iso()

    @property
    def total_tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.events)

    @property
    def total_tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.events)

    @property
    def total_duration_ms(self) -> int:
        return sum(e.duration_ms for e in self.events)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "event_count": self.event_count,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_duration_ms": self.total_duration_ms,
            "event_types": _count_by(self.events, "event_type"),
        }


def _count_by(events: list[TraceEvent], attr: str) -> dict[str, int]:
    """Count events by a given attribute."""
    counts: dict[str, int] = {}
    for e in events:
        key = getattr(e, attr, "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Collects and persists trace events for agent observability.

    Events are written to JSONL files in logs/traces/, one file per trace.

    Args:
        traces_dir: override default traces directory.
        cost_rates: override default $/1K token cost rates per model.
    """

    def __init__(self, traces_dir: Path | None = None,
                 cost_rates: dict[str, dict[str, float]] | None = None) -> None:
        self._dir = traces_dir or TRACES_DIR
        _ensure_dir(self._dir)
        self._cost_rates = cost_rates or DEFAULT_COST_RATES
        self._sessions: dict[str, TraceSession] = {}
        self._current_trace_id: str = ""

    @property
    def current_trace_id(self) -> str:
        return self._current_trace_id

    def start_session(self, agent_name: str,
                      trace_id: str | None = None) -> str:
        """Start a new trace session. Returns the trace_id."""
        tid = trace_id or uuid.uuid4().hex[:16]
        self._current_trace_id = tid
        session = TraceSession(trace_id=tid, agent_name=agent_name)
        self._sessions[tid] = session
        logger.info("trace session started: %s (%s)", tid, agent_name)
        return tid

    def end_session(self, trace_id: str | None = None) -> TraceSession | None:
        """End a trace session and return its data."""
        tid = trace_id or self._current_trace_id
        session = self._sessions.get(tid)
        if session:
            session.ended_at = _now_iso()
            logger.info("trace session ended: %s (%d events)",
                        tid, session.event_count)
        return session

    def record(self, event: TraceEvent) -> None:
        """Record a trace event to the current session and persist to disk."""
        tid = event.trace_id or self._current_trace_id
        if not tid:
            logger.warning("record() called with no trace_id and no active "
                           "session -- event dropped for %s", event.agent_name)
            return
        event.trace_id = tid

        # Add to in-memory session if active
        if tid in self._sessions:
            self._sessions[tid].events.append(event)

        # Persist to JSONL
        self._write_event(tid, event)

    def _write_event(self, trace_id: str, event: TraceEvent) -> None:
        """Append event to the trace JSONL file."""
        log_path = safe_path(self._dir / f"{trace_id}.jsonl")
        entry = event.to_dict()
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            logger.warning("failed to write trace event: %s", exc)

    def _load_events(self, trace_id: str) -> list[TraceEvent]:
        """Load all events for a trace from disk."""
        log_path = self._dir / f"{trace_id}.jsonl"
        if not log_path.exists():
            return []
        events: list[TraceEvent] = []
        try:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(TraceEvent.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass
        return events

    def _list_trace_ids(self) -> list[str]:
        """List all trace IDs from disk."""
        if not self._dir.exists():
            return []
        return sorted(
            f.stem for f in self._dir.glob("*.jsonl")
        )

    # -- Analysis -----------------------------------------------------------

    def export_timeline(self, trace_id: str) -> list[dict[str, Any]]:
        """Return JSON timeline of events for a trace."""
        events = self._load_events(trace_id)
        return [e.to_dict() for e in events]

    def token_attribution(self,
                          trace_id: str | None = None) -> dict[str, dict[str, int]]:
        """Per-agent token usage breakdown.

        If trace_id is given, scoped to that trace.
        Otherwise aggregates across all traces.
        """
        if trace_id:
            events = self._load_events(trace_id)
        else:
            events = []
            for tid in self._list_trace_ids():
                events.extend(self._load_events(tid))

        attribution: dict[str, dict[str, int]] = {}
        for e in events:
            agent = e.agent_name or "unknown"
            if agent not in attribution:
                attribution[agent] = {"tokens_in": 0, "tokens_out": 0,
                                      "total": 0, "events": 0}
            attribution[agent]["tokens_in"] += e.tokens_in
            attribution[agent]["tokens_out"] += e.tokens_out
            attribution[agent]["total"] += e.tokens_in + e.tokens_out
            attribution[agent]["events"] += 1

        return attribution

    def cost_report(self,
                    trace_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Estimated cost by agent and model.

        Uses configurable $/1K token rates.
        """
        if trace_id:
            events = self._load_events(trace_id)
        else:
            events = []
            for tid in self._list_trace_ids():
                events.extend(self._load_events(tid))

        costs: dict[str, dict[str, float]] = {}
        for e in events:
            agent = e.agent_name or "unknown"
            model = e.model or "default"

            # Find matching rate
            rates = self._cost_rates.get(model, self._cost_rates["default"])

            cost_in = (e.tokens_in / 1000) * rates["input"]
            cost_out = (e.tokens_out / 1000) * rates["output"]

            if agent not in costs:
                costs[agent] = {"input_cost": 0.0, "output_cost": 0.0,
                                "total_cost": 0.0, "tokens_in": 0,
                                "tokens_out": 0}
            costs[agent]["input_cost"] += cost_in
            costs[agent]["output_cost"] += cost_out
            costs[agent]["total_cost"] += cost_in + cost_out
            costs[agent]["tokens_in"] += e.tokens_in
            costs[agent]["tokens_out"] += e.tokens_out

        return costs

    # -- HTML export --------------------------------------------------------

    def export_html(self, trace_id: str) -> str:
        """Generate a standalone HTML file with visual timeline.

        Returns the file path of the generated HTML.
        """
        events = self._load_events(trace_id)
        if not events:
            return ""

        # Build timeline data -- escape closing script tags to prevent XSS
        # if trace data contains user-controlled strings
        timeline_data = json.dumps(
            [e.to_dict() for e in events], default=str, indent=2
        ).replace("</", "<\\/")

        # Token totals
        total_in = sum(e.tokens_in for e in events)
        total_out = sum(e.tokens_out for e in events)
        total_dur = sum(e.duration_ms for e in events)

        # Agent colors
        agents = sorted(set(e.agent_name for e in events))
        colors = [
            "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
            "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
        ]
        agent_colors = {a: colors[i % len(colors)] for i, a in enumerate(agents)}

        # Event type icons
        type_icons = {
            "llm_call": "&#9641;",   # square
            "tool_use": "&#9654;",   # triangle
            "handoff": "&#10132;",   # arrow
            "error": "&#10006;",     # X
            "decision": "&#9733;",   # star
            "custom": "&#9679;",     # circle
        }

        # Build event rows
        event_rows = []
        for e in events:
            color = agent_colors.get(e.agent_name, "#666")
            icon = type_icons.get(e.event_type, "&#9679;")
            ts_short = e.timestamp[11:19] if len(e.timestamp) > 19 else e.timestamp
            in_summary = html_lib.escape(e.input_summary[:80]) if e.input_summary else ""
            out_summary = html_lib.escape(e.output_summary[:80]) if e.output_summary else ""

            event_rows.append(f"""
            <div class="event" style="border-left: 4px solid {color};">
                <div class="event-header">
                    <span class="icon">{icon}</span>
                    <span class="type">{html_lib.escape(e.event_type)}</span>
                    <span class="agent" style="color: {color};">{html_lib.escape(e.agent_name)}</span>
                    <span class="time">{ts_short}</span>
                    <span class="duration">{e.duration_ms}ms</span>
                    <span class="tokens">{e.tokens_in}+{e.tokens_out} tok</span>
                </div>
                <div class="event-body">
                    {f'<div class="input">IN: {in_summary}</div>' if in_summary else ''}
                    {f'<div class="output">OUT: {out_summary}</div>' if out_summary else ''}
                </div>
            </div>""")

        events_html = "\n".join(event_rows)

        # Agent legend
        legend_items = "".join(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{c};"></span>'
            f'{html_lib.escape(a)}</span>'
            for a, c in agent_colors.items()
        )

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace: {html_lib.escape(trace_id)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 18px; margin-bottom: 8px; color: #fff; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
  .meta span {{ margin-right: 16px; }}
  .legend {{ margin-bottom: 20px; display: flex; gap: 16px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .timeline {{ display: flex; flex-direction: column; gap: 4px; }}
  .event {{ background: #151515; border-radius: 4px; padding: 10px 14px; }}
  .event-header {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}
  .icon {{ font-size: 14px; }}
  .type {{ font-weight: 600; min-width: 80px; }}
  .agent {{ min-width: 100px; font-weight: 500; }}
  .time {{ color: #888; }}
  .duration {{ color: #4ade80; min-width: 60px; text-align: right; }}
  .tokens {{ color: #888; font-size: 12px; }}
  .event-body {{ margin-top: 6px; font-size: 12px; color: #aaa; }}
  .input {{ margin-bottom: 2px; }}
  .output {{ color: #6ee7b7; }}
</style>
</head>
<body>
<h1>Trace: {html_lib.escape(trace_id)}</h1>
<div class="meta">
  <span>Events: {len(events)}</span>
  <span>Tokens: {total_in} in / {total_out} out</span>
  <span>Duration: {total_dur}ms</span>
</div>
<div class="legend">{legend_items}</div>
<div class="timeline">
{events_html}
</div>
<script>
const traceData = {timeline_data};
console.log('Trace data loaded:', traceData.length, 'events');
</script>
</body>
</html>"""

        # Write HTML file
        html_path = safe_path(self._dir / f"{trace_id}.html")
        with open(html_path, "w") as f:
            f.write(page)

        logger.info("exported HTML timeline: %s", html_path)
        return str(html_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _show_trace(trace_id: str) -> None:
    tracer = Tracer()
    events = tracer._load_events(trace_id)
    if not events:
        print(f"\nNo trace found for ID: {trace_id}\n")
        return

    total_in = sum(e.tokens_in for e in events)
    total_out = sum(e.tokens_out for e in events)
    total_dur = sum(e.duration_ms for e in events)

    print(f"\n=== Trace: {trace_id} ===\n")
    print(f"Events: {len(events)}  Tokens: {total_in} in / {total_out} out  "
          f"Duration: {total_dur}ms\n")

    for e in events:
        ts = e.timestamp[11:19] if len(e.timestamp) > 19 else e.timestamp
        print(f"  [{ts}] {e.event_type:10s} {e.agent_name:15s} "
              f"{e.duration_ms:6d}ms  {e.tokens_in}+{e.tokens_out} tok")
        if e.input_summary:
            print(f"           IN:  {e.input_summary[:80]}")
        if e.output_summary:
            print(f"           OUT: {e.output_summary[:80]}")
    print()


def _show_agents() -> None:
    tracer = Tracer()
    attribution = tracer.token_attribution()
    if not attribution:
        print("\nNo trace data found.\n")
        return

    print(f"\n=== Agent Token Attribution ===\n")
    print(f"  {'Agent':20s} {'In':>10s} {'Out':>10s} {'Total':>10s} {'Events':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for agent, data in sorted(attribution.items(),
                                key=lambda x: x[1]["total"], reverse=True):
        print(f"  {agent:20s} {data['tokens_in']:>10,} "
              f"{data['tokens_out']:>10,} {data['total']:>10,} "
              f"{data['events']:>8d}")
    print()


def _show_cost() -> None:
    tracer = Tracer()
    costs = tracer.cost_report()
    if not costs:
        print("\nNo trace data found.\n")
        return

    total_cost = sum(c["total_cost"] for c in costs.values())

    print(f"\n=== Cost Report (estimated) ===\n")
    print(f"  {'Agent':20s} {'In Cost':>10s} {'Out Cost':>10s} "
          f"{'Total':>10s} {'Tokens':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for agent, data in sorted(costs.items(),
                                key=lambda x: x[1]["total_cost"], reverse=True):
        tok_total = data["tokens_in"] + data["tokens_out"]
        print(f"  {agent:20s} ${data['input_cost']:>8.4f} "
              f"${data['output_cost']:>8.4f} ${data['total_cost']:>8.4f} "
              f"{tok_total:>11,}")
    print(f"\n  Total estimated cost: ${total_cost:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracing -- agent observability and cost tracking")
    parser.add_argument("--trace", type=str, metavar="ID",
                        help="Show trace details")
    parser.add_argument("--agents", action="store_true",
                        help="Per-agent token summary")
    parser.add_argument("--cost", action="store_true",
                        help="Estimated cost report")
    parser.add_argument("--html", type=str, metavar="ID",
                        help="Generate HTML timeline for a trace")
    args = parser.parse_args()

    if not any([args.trace, args.agents, args.cost, args.html]):
        parser.print_help()
        return

    if args.trace:
        _show_trace(args.trace)

    if args.agents:
        _show_agents()

    if args.cost:
        _show_cost()

    if args.html:
        tracer = Tracer()
        path = tracer.export_html(args.html)
        if path:
            print(f"HTML timeline written to: {path}")
        else:
            print(f"No events found for trace: {args.html}")


if __name__ == "__main__":
    main()
