#!/usr/bin/env python3
"""
Decision Engine -- Closed-loop autonomous decision agent.

Replaces the "scan data, log to CSV, nothing happens" pattern with:
scan > analyze > decide > act > log > learn

Each cycle:
1. Reads data sources (CSVs, logs, output directories)
2. Scores opportunities using configurable thresholds
3. Takes action (within safety limits)
4. Logs every decision with reasoning for audit trail
5. Updates progress trackers

The engine can run rule-based decisions for simple threshold checks
and delegate to an LLM for nuanced judgment calls.

Usage:
    python3 decision_engine.py --cycle          # Run one decision cycle
    python3 decision_engine.py --daemon         # Run continuously
    python3 decision_engine.py --status         # Show pipeline status
    python3 decision_engine.py --dry-run        # Show what would happen without acting
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DECISION_LOG = LOGS_DIR / "decision_engine.log"
DECISION_LEDGER = PROJECT_ROOT / "output" / "decisions.csv"
STATE_FILE = PROJECT_ROOT / "state" / "decision_engine_state.json"


def safe_path(target: str | Path) -> Path:
    """Verify path is within project root."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} is outside project root {root}")
    return resolved


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [DECISION] [{level}] {msg}"
    print(line)
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_LOG, "a") as f:
        f.write(line + "\n")


def log_decision(source, action, reasoning, outcome="PENDING"):
    """Append to decisions ledger for full audit trail."""
    DECISION_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "action": action,
        "reasoning": reasoning,
        "outcome": outcome,
    }
    exists = DECISION_LEDGER.exists()
    with open(DECISION_LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Decision scoring
# ---------------------------------------------------------------------------

class DecisionRule:
    """A single decision rule with threshold-based scoring."""

    def __init__(self, name: str, source_check, threshold: float,
                 action: str, reasoning_template: str):
        self.name = name
        self.source_check = source_check  # callable() -> score
        self.threshold = threshold
        self.action = action
        self.reasoning_template = reasoning_template

    def evaluate(self) -> dict | None:
        """Evaluate this rule. Returns decision dict if threshold met, None otherwise."""
        try:
            score = self.source_check()
        except Exception as e:
            log(f"Rule {self.name} check failed: {e}", "WARN")
            return None

        if score >= self.threshold:
            return {
                "rule": self.name,
                "score": score,
                "threshold": self.threshold,
                "action": self.action,
                "reasoning": self.reasoning_template.format(score=score),
            }
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Users register their own rules here
RULES: list[DecisionRule] = []


def register_rule(rule: DecisionRule):
    """Register a decision rule with the engine."""
    RULES.append(rule)


def load_state() -> dict:
    """Load engine state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "last_cycle": None,
        "total_cycles": 0,
        "total_decisions": 0,
        "total_actions": 0,
    }


def save_state(state: dict):
    """Save engine state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run_cycle(dry_run: bool = False) -> list[dict]:
    """Run one decision cycle through all registered rules."""
    state = load_state()
    decisions = []

    log(f"Decision cycle starting. {len(RULES)} rules registered.")

    for rule in RULES:
        result = rule.evaluate()
        if result:
            decisions.append(result)
            if dry_run:
                log(f"DRY-RUN [{rule.name}]: {result['action']} (score: {result['score']})")
            else:
                log(f"EXECUTING [{rule.name}]: {result['action']} (score: {result['score']})")
                log_decision(rule.name, result["action"], result["reasoning"])

    state["last_cycle"] = datetime.now().isoformat()
    state["total_cycles"] = state.get("total_cycles", 0) + 1
    state["total_decisions"] = state.get("total_decisions", 0) + len(decisions)
    if not dry_run:
        state["total_actions"] = state.get("total_actions", 0) + len(decisions)
    save_state(state)

    log(f"Cycle complete. {len(decisions)} decisions made.")
    return decisions


def run_daemon(interval_minutes: int = 30):
    """Run the engine continuously."""
    log(f"Decision engine daemon starting. Interval: {interval_minutes}min")
    while True:
        try:
            run_cycle()
        except Exception as e:
            log(f"Cycle error: {e}", "ERROR")
        time.sleep(interval_minutes * 60)


def show_status():
    """Show engine status."""
    state = load_state()
    print("\n=== Decision Engine Status ===\n")
    print(f"Last cycle:       {state.get('last_cycle', 'never')}")
    print(f"Total cycles:     {state.get('total_cycles', 0)}")
    print(f"Total decisions:  {state.get('total_decisions', 0)}")
    print(f"Total actions:    {state.get('total_actions', 0)}")
    print(f"Rules registered: {len(RULES)}")

    if DECISION_LEDGER.exists():
        try:
            with open(DECISION_LEDGER) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"Audit trail:      {len(rows)} entries in {DECISION_LEDGER.name}")
            if rows:
                last = rows[-1]
                print(f"Last decision:    [{last.get('source', '?')}] {last.get('action', '?')[:80]}")
        except Exception:
            pass

    print()


def main():
    parser = argparse.ArgumentParser(description="Decision Engine -- closed-loop autonomous decisions")
    parser.add_argument("--cycle", action="store_true", help="Run one decision cycle")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    parser.add_argument("--dry-run", action="store_true", help="Show without acting")
    parser.add_argument("--interval", type=int, default=30, help="Daemon interval in minutes")
    args = parser.parse_args()

    if not any([args.cycle, args.daemon, args.status]):
        parser.print_help()
        return

    if args.cycle or args.dry_run:
        run_cycle(dry_run=args.dry_run)
    elif args.daemon:
        run_daemon(interval_minutes=args.interval)
    elif args.status:
        show_status()


if __name__ == "__main__":
    main()
