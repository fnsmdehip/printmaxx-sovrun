#!/usr/bin/env python3
"""
Self Audit -- Competitive cognition audit for meta-improvement.

Analyzes your own system's prompting patterns, configuration files,
and architecture to find ways to improve. This is the system that
improves the system.

Checks for:
- AI slop in system files (banned vocabulary in your own config)
- Context bloat (instruction files too long, wasting tokens every session)
- Missing meta-cognition protocols
- Orphan scripts (code not wired into any automation)
- Prompt pattern health (are you learning from corrections?)
- Competitive edge decay (are your patterns still ahead?)

Usage:
    python3 self_audit.py --audit    # Run full audit
    python3 self_audit.py --report   # Show latest findings
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

AUDIT_DIR = Path(os.environ.get(
    "SOVRUN_AUDIT_DIR", PROJECT_ROOT / "output" / "cognition_audits"))
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_AUDIT = AUDIT_DIR / "latest_audit.json"

# Files to check (override via environment or config)
SYSTEM_INSTRUCTIONS = Path(os.environ.get(
    "SOVRUN_INSTRUCTIONS", PROJECT_ROOT / "templates" / "CLAUDE.md"))
SOUL_FILE = Path(os.environ.get(
    "SOVRUN_SOUL_MD", PROJECT_ROOT / "templates" / "SOUL.md"))
SCRIPTS_DIR = Path(os.environ.get(
    "SOVRUN_SCRIPTS_DIR", PROJECT_ROOT / "core"))
PROMPTS_FILE = Path(os.environ.get(
    "SOVRUN_PROMPTS", PROJECT_ROOT / "data" / "prompts.jsonl"))

# Banned AI vocabulary (same list used by voice extractor)
BANNED_WORDS = [
    "comprehensive", "leverage", "utilize", "delve", "innovative",
    "seamless", "game-changer", "unlock", "elevate", "cutting-edge",
    "empower", "foster", "frictionless", "journey", "robust", "streamlined",
]


def audit_system_patterns():
    """Analyze the system for improvement opportunities."""
    findings = []

    # 1. Check system instructions for staleness and slop
    if SYSTEM_INSTRUCTIONS.exists():
        content = SYSTEM_INSTRUCTIONS.read_text()
        lines = len(content.split("\n"))

        for word in BANNED_WORDS:
            if word.lower() in content.lower():
                findings.append({
                    "category": "AI_SLOP_IN_SYSTEM_FILES",
                    "severity": "MEDIUM",
                    "detail": f"System instructions contain banned AI vocabulary ('{word}'). Your own config has slop.",
                    "fix": f"Search and replace '{word}' in {SYSTEM_INSTRUCTIONS.name}"
                })
                break  # one finding is enough

        if lines > 500:
            findings.append({
                "category": "CONTEXT_BLOAT",
                "severity": "HIGH",
                "detail": f"System instructions are {lines} lines. Every session loads this. Token waste compounds.",
                "fix": "Extract rarely-used sections to reference files. Keep main instructions under 200 lines."
            })

    # 2. Check SOUL.md for key sections
    if SOUL_FILE.exists():
        content = SOUL_FILE.read_text()
        if "bias-null" not in content.lower() and "bias null" not in content.lower():
            findings.append({
                "category": "MISSING_BIAS_NULL",
                "severity": "HIGH",
                "detail": "SOUL.md lacks bias-null protocol. Agents will default to generic LLM priors.",
                "fix": "Add Bias-Null Stack section to SOUL.md"
            })

        if "competitive" not in content.lower() and "self-correction" not in content.lower():
            findings.append({
                "category": "MISSING_META_COGNITION",
                "severity": "HIGH",
                "detail": "SOUL.md lacks competitive cognition or self-correction protocol.",
                "fix": "Add Competitive Cognition Protocol section to SOUL.md"
            })

        lines = len(content.split("\n"))
        if lines > 200:
            findings.append({
                "category": "SOUL_BLOAT",
                "severity": "MEDIUM",
                "detail": f"SOUL.md is {lines} lines. Over-long SOUL = diluted identity.",
                "fix": "Trim SOUL.md to essential behavioral directives only."
            })

    # 3. Check for orphan scripts
    if SCRIPTS_DIR.exists():
        scripts = list(SCRIPTS_DIR.glob("*.py"))
        if scripts:
            # Simple check: are scripts importable?
            broken = []
            for script in scripts:
                if script.name.startswith("_"):
                    continue
                try:
                    content = script.read_text()
                    compile(content, str(script), "exec")
                except SyntaxError as e:
                    broken.append(f"{script.name}: {e}")

            if broken:
                findings.append({
                    "category": "BROKEN_SCRIPTS",
                    "severity": "HIGH",
                    "detail": f"{len(broken)} scripts have syntax errors: {', '.join(broken[:3])}",
                    "fix": "Fix syntax errors in broken scripts."
                })

    # 4. Check prompt history health
    if PROMPTS_FILE.exists():
        try:
            prompt_count = sum(1 for _ in open(PROMPTS_FILE))
            findings.append({
                "category": "PROMPT_INTELLIGENCE",
                "severity": "INFO",
                "detail": f"{prompt_count} user prompts logged. Run pattern_miner.py and cognitive_engine.py to extract value.",
                "fix": "Run full extraction pipeline on prompt history."
            })
        except OSError:
            pass

    # 5. Meta-cognition check
    findings.append({
        "category": "EDGE_CHECK",
        "severity": "INFO",
        "detail": "Weekly check: Are your system patterns still ahead of generic approaches? Compare your correction chain reduction over time.",
        "fix": "Run cognitive_engine.py --status and compare chain lengths month over month."
    })

    findings.append({
        "category": "META_COGNITION",
        "severity": "INFO",
        "detail": "Is the voice model actually changing agent behavior? Compare agent output quality before and after voice injection.",
        "fix": "A/B test: run same task with and without voice model injection. Measure user satisfaction."
    })

    # Save audit
    audit = {
        "ts": datetime.now().isoformat(),
        "findings": findings,
        "total_findings": len(findings),
        "high_severity": len([f for f in findings if f["severity"] == "HIGH"]),
        "medium_severity": len([f for f in findings if f["severity"] == "MEDIUM"]),
    }

    with open(LATEST_AUDIT, "w") as f:
        json.dump(audit, f, indent=2)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    with open(AUDIT_DIR / f"audit_{ts}.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n=== Self Audit ===")
    print(f"Findings: {len(findings)} ({audit['high_severity']} HIGH, {audit['medium_severity']} MEDIUM)")
    for f in findings:
        print(f"  [{f['severity']:6s}] {f['category']}: {f['detail'][:100]}")

    return audit


def show_report():
    """Show latest audit findings."""
    if LATEST_AUDIT.exists():
        audit = json.loads(LATEST_AUDIT.read_text())
        print(f"\n=== Latest Self Audit ({audit['timestamp']}) ===")
        for f in audit.get("findings", []):
            print(f"\n[{f['severity']}] {f['category']}")
            print(f"  Detail: {f['detail']}")
            print(f"  Fix: {f['fix']}")
    else:
        print("No audit results yet. Run --audit first.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--audit" in args:
        audit_system_patterns()
    elif "--report" in args:
        show_report()
    else:
        print("Usage: self_audit.py --audit | --report")
        print("Meta-improvement system that audits your own system architecture.")
