#!/usr/bin/env python3
"""
Pattern Miner -- Extracts meta-rules from user prompt history.

Reads prompt JSONL files and mines for: recurring patterns, correction sequences,
escalation triggers, effective prompting styles, and transferable meta-rules.

Also provides keyword search: before executing a task, find similar past prompts
and what approach worked.

Usage:
    python3 pattern_miner.py --mine              # Extract all patterns
    python3 pattern_miner.py --similar "QUERY"   # Find similar past prompts
    python3 pattern_miner.py --corrections        # Find correction sequences
    python3 pattern_miner.py --escalations        # Find escalation patterns
    python3 pattern_miner.py --effective          # Find satisfaction triggers
    python3 pattern_miner.py --rules              # Generate meta-rules from patterns
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

PROMPTS_FILE = Path(os.environ.get(
    "SOVRUN_PROMPTS", PROJECT_ROOT / "data" / "prompts.jsonl"))
OUTPUT_DIR = Path(os.environ.get(
    "SOVRUN_PATTERNS_DIR", PROJECT_ROOT / "output" / "prompt_intelligence"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PATTERNS_FILE = OUTPUT_DIR / "extracted_patterns.json"
RULES_FILE = OUTPUT_DIR / "mined_meta_rules.md"

# Markers of user pushback / correction (gold: they reveal where the system fails)
CORRECTION_MARKERS = [
    "no not", "don't do that", "that's not what", "bruh", "lazy", "wrong",
    "i said", "not that", "why did you", "stop", "you're just",
    "didn't ask", "too much", "too generic", "not good",
    "basic", "default", "actually think", "critically",
    "not just", "deeper", "above and beyond",
    "surprise me", "proactive", "use ur best", "best judgment",
]

# Markers of satisfaction / effective prompts
SATISFACTION_MARKERS = [
    "perfect", "exactly", "yes that", "love it", "fire",
    "this is what", "appreciate", "nice", "great", "solid", "nailed",
]

# Markers of escalation (user pushing for more depth)
ESCALATION_MARKERS = [
    "like why", "but why", "what about", "also",
    "think bigger", "think harder",
    "surprise me", "above and beyond",
    "i feel like", "shouldn't we", "what if",
]


def load_prompts():
    """Load all user prompts."""
    prompts = []
    if PROMPTS_FILE.exists():
        for line in PROMPTS_FILE.read_text().strip().split("\n"):
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return prompts


def find_corrections(prompts):
    """Find prompts where user corrected/pushed back on the system."""
    corrections = []
    for p in prompts:
        text = p.get("prompt", "").lower()
        triggered = [m for m in CORRECTION_MARKERS if m in text]
        if triggered:
            corrections.append({
                "timestamp": p.get("ts", ""),
                "prompt": p.get("prompt", "")[:500],
                "markers": triggered,
                "marker_count": len(triggered),
            })
    corrections.sort(key=lambda x: x["marker_count"], reverse=True)
    return corrections


def find_escalations(prompts):
    """Find prompts where user pushed for deeper thinking."""
    escalations = []
    for p in prompts:
        text = p.get("prompt", "").lower()
        triggered = [m for m in ESCALATION_MARKERS if m in text]
        if triggered:
            escalations.append({
                "timestamp": p.get("ts", ""),
                "prompt": p.get("prompt", "")[:500],
                "markers": triggered,
            })
    return escalations


def find_satisfactions(prompts):
    """Find prompts where user expressed satisfaction."""
    satisfactions = []
    for p in prompts:
        text = p.get("prompt", "").lower()
        triggered = [m for m in SATISFACTION_MARKERS if m in text]
        if triggered:
            satisfactions.append({
                "timestamp": p.get("ts", ""),
                "prompt": p.get("prompt", "")[:500],
                "markers": triggered,
            })
    return satisfactions


def extract_keyword_patterns(prompts):
    """Find recurring themes and keywords across prompts."""
    all_words = Counter()
    bigrams = Counter()
    for p in prompts:
        text = p.get("prompt", "").lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)
        all_words.update(words)
        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1

    common = {"that", "this", "with", "from", "have", "like", "just", "what",
              "make", "sure", "also", "want", "need", "think", "about", "them",
              "their", "they", "would", "could", "should", "every", "some",
              "other", "then", "when", "into", "more", "been", "will", "your",
              "does", "know", "good", "best", "dont"}
    filtered = {k: v for k, v in all_words.items() if k not in common and v >= 3}
    return dict(Counter(filtered).most_common(50)), dict(bigrams.most_common(30))


def find_similar_prompts(query, prompts, top_n=5):
    """Find prompts similar to a query using keyword overlap."""
    query_words = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
    scored = []
    for p in prompts:
        text = p.get("prompt", "")
        prompt_words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
        overlap = len(query_words & prompt_words)
        if overlap >= 2:
            scored.append({
                "timestamp": p.get("ts", ""),
                "prompt": text[:300],
                "overlap": overlap,
                "shared_words": list(query_words & prompt_words)[:10],
            })
    scored.sort(key=lambda x: x["overlap"], reverse=True)
    return scored[:top_n]


def generate_meta_rules(corrections, escalations, satisfactions):
    """Generate meta-rules from pattern analysis."""
    rules = []

    correction_themes = Counter()
    for c in corrections:
        for m in c["markers"]:
            correction_themes[m] += 1

    top_corrections = correction_themes.most_common(10)
    for marker, count in top_corrections:
        if marker in ("lazy", "basic", "default"):
            rules.append(f"ANTI-LAZY (triggered {count}x): User pushes back when output defaults to generic solutions instead of critically analyzing for best.")
        elif marker in ("bruh", "wrong", "no not"):
            rules.append(f"DIRECT-CORRECTION (triggered {count}x): User gives direct corrections. Extract the rule and never repeat the mistake.")
        elif marker in ("surprise me", "above and beyond", "proactive"):
            rules.append(f"DEPTH-DEMAND (triggered {count}x): User expects output BEYOND the literal ask. Find adjacent improvements and non-obvious angles.")

    if len(escalations) > 5:
        rules.append(f"ESCALATION-PATTERN: User escalated {len(escalations)} times across sessions. Default response should be at the depth of what the user typically escalates TO.")

    if satisfactions:
        rules.append(f"SATISFACTION-TRIGGERS: User expressed satisfaction {len(satisfactions)} times. Common context: when system executes autonomously, when output exceeds expectations.")

    return rules


def mine_all():
    """Run full pattern mining pipeline."""
    print("\n=== Pattern Miner ===\n")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    corrections = find_corrections(prompts)
    escalations = find_escalations(prompts)
    satisfactions = find_satisfactions(prompts)
    keywords, bigrams = extract_keyword_patterns(prompts)

    print(f"Corrections (user pushback): {len(corrections)}")
    print(f"Escalations (depth demands): {len(escalations)}")
    print(f"Satisfactions: {len(satisfactions)}")

    rules = generate_meta_rules(corrections, escalations, satisfactions)

    patterns = {
        "mined_at": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "correction_count": len(corrections),
        "escalation_count": len(escalations),
        "satisfaction_count": len(satisfactions),
        "top_corrections": corrections[:20],
        "top_escalations": escalations[:15],
        "satisfactions": satisfactions[:10],
        "top_keywords": dict(list(keywords.items())[:30]),
        "top_bigrams": dict(list(bigrams.items())[:20]),
        "meta_rules": rules,
    }
    with open(PATTERNS_FILE, "w") as f:
        json.dump(patterns, f, indent=2)

    rules_md = f"# Mined Meta-Rules\n\n"
    rules_md += f"Source: {len(prompts)} prompts analyzed\n\n"
    for i, rule in enumerate(rules, 1):
        rules_md += f"{i}. {rule}\n\n"

    rules_md += "\n## Top Correction Triggers\n\n"
    for c in corrections[:10]:
        rules_md += f"- [{c['timestamp'][:10]}] Markers: {', '.join(c['markers'][:3])}\n"

    rules_md += "\n## Top Escalation Triggers\n\n"
    for e in escalations[:10]:
        rules_md += f"- [{e['timestamp'][:10]}] Markers: {', '.join(e['markers'][:3])}\n"

    with open(RULES_FILE, "w") as f:
        f.write(rules_md)

    print(f"\nPatterns saved: {PATTERNS_FILE}")
    print(f"Rules saved: {RULES_FILE}")
    print(f"\nMeta-rules extracted: {len(rules)}")
    for r in rules:
        print(f"  - {r[:100]}...")

    return patterns


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--mine" in args:
        mine_all()
    elif "--similar" in args:
        idx = args.index("--similar")
        query = " ".join(args[idx + 1:])
        prompts = load_prompts()
        similar = find_similar_prompts(query, prompts)
        if similar:
            print(f"\n=== Similar Past Prompts ===\n")
            for i, s in enumerate(similar, 1):
                print(f"{i}. [{s['timestamp'][:10]}] (overlap: {s['overlap']})")
                print(f"   Shared: {', '.join(s['shared_words'][:5])}")
                print(f"   Prompt: {s['prompt'][:150]}...")
                print()
        else:
            print("No similar prompts found.")
    elif "--corrections" in args:
        prompts = load_prompts()
        corrections = find_corrections(prompts)
        print(f"\n=== Correction Patterns ({len(corrections)} found) ===\n")
        for c in corrections[:15]:
            print(f"[{c['timestamp'][:10]}] Markers: {', '.join(c['markers'][:4])}")
            print(f"  \"{c['prompt'][:200]}...\"")
            print()
    elif "--escalations" in args:
        prompts = load_prompts()
        escalations = find_escalations(prompts)
        print(f"\n=== Escalation Patterns ({len(escalations)} found) ===\n")
        for e in escalations[:15]:
            print(f"[{e['timestamp'][:10]}] {', '.join(e['markers'][:3])}")
            print(f"  \"{e['prompt'][:200]}...\"")
            print()
    elif "--effective" in args:
        prompts = load_prompts()
        sats = find_satisfactions(prompts)
        print(f"\n=== Satisfaction Triggers ({len(sats)} found) ===\n")
        for s in sats[:10]:
            print(f"[{s['timestamp'][:10]}] {', '.join(s['markers'][:3])}")
            print(f"  \"{s['prompt'][:200]}...\"")
            print()
    elif "--rules" in args:
        mine_all()
    else:
        print("Pattern Miner -- extracts meta-rules from prompt history")
        print()
        print("  --mine              Full pattern extraction")
        print("  --similar QUERY     Find similar past prompts")
        print("  --corrections       Show user pushback patterns")
        print("  --escalations       Show depth-demand patterns")
        print("  --effective         Show satisfaction triggers")
        print("  --rules             Generate meta-rules")
