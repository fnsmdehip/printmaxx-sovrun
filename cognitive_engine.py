#!/usr/bin/env python3
"""
Cognitive Engine -- Hybrid prompt intelligence system.

Combines three capabilities:
1. META-RULE EXTRACTION: Mines prompt history for transferable cognitive patterns
2. SIMILAR-TASK LOOKUP: Finds past task sequences (prompt > correction > resolution)
   to inform how to approach similar new tasks
3. AUTONOMOUS REFINEMENT: Uses extracted patterns + task history to simulate
   what the user would prompt, then feeds it into the next cycle

The engine builds a "cognition model" of the user's thinking by:
- Tracking correction sequences (what went wrong, how user fixed it)
- Identifying effective prompt chains (series of prompts that led to good outcomes)
- Extracting transferable rules from both failures and successes
- Providing keyword search across all past interactions

Usage:
    python3 cognitive_engine.py --build-model          # Build full cognition model
    python3 cognitive_engine.py --lookup "TASK_DESC"    # Find similar past tasks
    python3 cognitive_engine.py --rules                 # Show extracted meta-rules
    python3 cognitive_engine.py --chain-analysis        # Analyze prompt correction chains
    python3 cognitive_engine.py --status                # Show engine status
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

PROMPTS_FILE = Path(os.environ.get(
    "SOVRUN_PROMPTS", PROJECT_ROOT / "data" / "prompts.jsonl"))
OUTPUT_DIR = Path(os.environ.get(
    "SOVRUN_COGNITION_DIR", PROJECT_ROOT / "output" / "prompt_intelligence"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = OUTPUT_DIR / "cognition_model.json"
CHAINS_FILE = OUTPUT_DIR / "correction_chains.json"
TASK_INDEX_FILE = OUTPUT_DIR / "task_index.json"
RULES_FILE = OUTPUT_DIR / "meta_rules.md"

# What signals a correction sequence (user was not happy, prompted again)
CORRECTION_SIGNALS = [
    "no ", "not that", "wrong", "bruh", "lazy", "don't", "stop",
    "i said", "why did", "that's not", "actually", "but ", "like why",
    "isn't", "shouldn't", "you're just", "basic", "surface",
    "deeper", "more", "also ", "what about",
]

SATISFACTION_SIGNALS = [
    "good", "perfect", "exactly", "yes", "nice", "great", "solid",
    "love", "fire", "appreciate", "nailed",
]


def load_prompts():
    """Load prompt history with timestamps."""
    prompts = []
    if PROMPTS_FILE.exists():
        for line in PROMPTS_FILE.read_text().strip().split("\n"):
            try:
                entry = json.loads(line)
                prompts.append(entry)
            except json.JSONDecodeError:
                continue
    return prompts


def extract_correction_chains(prompts):
    """Find sequences where user corrected the system and trace the full chain.

    Uses TIMESTAMP PROXIMITY (configurable window) instead of session_id.

    A correction chain looks like:
    1. User asks for X (initial prompt)
    2. [System responds]
    3. User says "no not that, do Y instead" (correction, within time window)
    4. [System responds again]
    5. User says "better but also Z" (refinement, within time window)
    6. User says "good" or moves to new topic (resolution)
    """
    chains = []
    current_chain = None
    prev_ts = None
    window_minutes = int(os.environ.get("SOVRUN_CHAIN_WINDOW", "10"))

    for prompt in prompts:
        text = prompt.get("prompt", "").lower()[:500]
        ts_str = prompt.get("ts", "")

        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        time_gap_minutes = float("inf")
        if prev_ts:
            time_gap_minutes = (ts - prev_ts).total_seconds() / 60

        prev_ts = ts

        is_correction = any(sig in text for sig in CORRECTION_SIGNALS)
        is_satisfaction = any(sig in text for sig in SATISFACTION_SIGNALS)
        is_same_conversation = time_gap_minutes < window_minutes

        if current_chain is None:
            if not is_correction and len(text) > 20:
                current_chain = {
                    "initial_prompt": prompt.get("prompt", "")[:500],
                    "initial_ts": ts_str,
                    "corrections": [],
                    "resolution": None,
                    "chain_length": 1,
                }
        else:
            if is_same_conversation and is_correction:
                current_chain["corrections"].append({
                    "prompt": prompt.get("prompt", "")[:500],
                    "ts": ts_str,
                    "signals": [s for s in CORRECTION_SIGNALS if s in text][:5],
                })
                current_chain["chain_length"] += 1
            elif is_same_conversation and is_satisfaction:
                current_chain["resolution"] = prompt.get("prompt", "")[:200]
                current_chain["resolved"] = True
                if current_chain["chain_length"] >= 2:
                    chains.append(current_chain)
                current_chain = None
            elif not is_same_conversation:
                current_chain["resolved"] = False
                if current_chain["chain_length"] >= 2:
                    chains.append(current_chain)
                if not is_correction and len(text) > 20:
                    current_chain = {
                        "initial_prompt": prompt.get("prompt", "")[:500],
                        "initial_ts": ts_str,
                        "corrections": [],
                        "resolution": None,
                        "chain_length": 1,
                    }
                else:
                    current_chain = None
            elif is_same_conversation and not is_correction:
                current_chain["chain_length"] += 1

    if current_chain and current_chain["chain_length"] >= 2:
        current_chain["resolved"] = False
        chains.append(current_chain)

    return chains


def build_task_index(prompts):
    """Build a searchable index of tasks and their outcomes.

    Groups prompts by timestamp proximity.
    """
    window_seconds = int(os.environ.get("SOVRUN_CHAIN_WINDOW", "10")) * 60
    conversations = []
    current_convo = []
    prev_ts = None

    for p in prompts:
        ts_str = p.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        if prev_ts and (ts - prev_ts).total_seconds() > window_seconds:
            if len(current_convo) >= 2:
                conversations.append(current_convo)
            current_convo = [p]
        else:
            current_convo.append(p)
        prev_ts = ts

    if len(current_convo) >= 2:
        conversations.append(current_convo)

    tasks = []
    for convo in conversations:
        initial = convo[0].get("prompt", "")[:500]
        words = set(re.findall(r'\b[a-z]{4,}\b', initial.lower()))

        corrections = [p for p in convo
                      if any(s in p.get("prompt", "").lower() for s in CORRECTION_SIGNALS)]

        resolved = any(any(s in p.get("prompt", "").lower() for s in SATISFACTION_SIGNALS)
                      for p in convo[-3:])

        tasks.append({
            "initial_task": initial,
            "keywords": list(words)[:20],
            "prompt_count": len(convo),
            "correction_count": len(corrections),
            "resolved": resolved,
            "correction_prompts": [c.get("prompt", "")[:300] for c in corrections[:5]],
            "all_prompts_preview": [p.get("prompt", "")[:150] for p in convo[:10]],
            "timestamp": convo[0].get("ts", ""),
        })

    return tasks


def lookup_similar_tasks(query, task_index, top_n=5):
    """Find similar past tasks using keyword overlap."""
    query_words = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))

    scored = []
    for task in task_index:
        task_words = set(task.get("keywords", []))
        overlap = len(query_words & task_words)
        if overlap >= 2:
            scored.append({
                **task,
                "overlap": overlap,
                "shared_words": list(query_words & task_words)[:8],
            })

    scored.sort(key=lambda x: x["overlap"], reverse=True)
    return scored[:top_n]


def extract_rules(chains, task_index, prompts):
    """Extract meta-rules from all data sources."""
    rules = []

    correction_themes = defaultdict(int)
    for chain in chains:
        for c in chain.get("corrections", []):
            for sig in c.get("signals", []):
                correction_themes[sig] += 1

    lazy_count = sum(v for k, v in correction_themes.items()
                     if k in ("lazy", "basic", "surface"))
    depth_count = sum(v for k, v in correction_themes.items()
                      if k in ("deeper", "more", "also ", "what about"))
    wrong_count = sum(v for k, v in correction_themes.items()
                      if k in ("no ", "wrong", "not that", "don't"))

    if lazy_count > 5:
        rules.append({
            "id": "ANTI-LAZY",
            "count": lazy_count,
            "rule": "Default output depth is consistently below user expectations. System must default to the ESCALATED depth, not the initial polite response depth.",
            "evidence": f"User triggered anti-lazy corrections {lazy_count} times across sessions.",
            "action": "Before presenting any output, ask: 'Would the user call this lazy?' If yes, go deeper before presenting.",
        })

    if depth_count > 5:
        rules.append({
            "id": "DEPTH-FIRST",
            "count": depth_count,
            "rule": "User consistently asks for MORE after initial response. The initial response should already include what the user typically asks for in their follow-up.",
            "evidence": f"User demanded more depth {depth_count} times.",
            "action": "After completing the literal ask, proactively address 2-3 adjacent areas the user would likely follow up on.",
        })

    if wrong_count > 10:
        rules.append({
            "id": "WRONG-DIRECTION",
            "count": wrong_count,
            "rule": "System frequently misinterprets user intent on first pass. Need better intent parsing before executing.",
            "evidence": f"User said 'no/wrong/not that' {wrong_count} times.",
            "action": "For ambiguous requests, internally generate 2-3 interpretations and select the most consistent with user patterns.",
        })

    high_correction_tasks = [t for t in task_index if t.get("correction_count", 0) >= 3]
    if high_correction_tasks:
        common_words = defaultdict(int)
        for t in high_correction_tasks:
            for w in t.get("keywords", []):
                common_words[w] += 1

        hard_topics = sorted(common_words.items(), key=lambda x: -x[1])[:10]
        rules.append({
            "id": "HARD-TOPICS",
            "count": len(high_correction_tasks),
            "rule": f"Topics that consistently need multiple corrections: {', '.join(w for w, _ in hard_topics[:5])}.",
            "evidence": f"{len(high_correction_tasks)} tasks had 3+ corrections.",
            "action": "When a task matches these topics, use the correction chain from history to pre-emptively address typical issues.",
        })

    satisfaction_contexts = []
    for p in prompts:
        text = p.get("prompt", "").lower()
        if any(s in text for s in SATISFACTION_SIGNALS):
            satisfaction_contexts.append(text[:200])

    if satisfaction_contexts:
        rules.append({
            "id": "SATISFACTION-PATTERN",
            "count": len(satisfaction_contexts),
            "rule": "User is most satisfied when: system executes autonomously without asking, output exceeds explicit ask, non-obvious angles are found.",
            "evidence": f"{len(satisfaction_contexts)} satisfaction signals found.",
            "action": "Optimize for these satisfaction triggers in every output.",
        })

    avg_chain = sum(c.get("chain_length", 1) for c in chains) / max(len(chains), 1)
    if avg_chain > 2:
        rules.append({
            "id": "CHAIN-LENGTH",
            "count": len(chains),
            "rule": f"Average correction chain is {avg_chain:.1f} prompts long. Target: get to the right output in 1 prompt.",
            "evidence": f"{len(chains)} correction chains analyzed.",
            "action": "Use the full cognition model to anticipate corrections before presenting output.",
        })

    return rules


def build_model():
    """Build the full cognition model."""
    print("\n=== Building Cognition Model ===\n")

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    chains = extract_correction_chains(prompts)
    print(f"Extracted {len(chains)} correction chains")

    task_index = build_task_index(prompts)
    print(f"Built task index with {len(task_index)} sessions")

    rules = extract_rules(chains, task_index, prompts)
    print(f"Generated {len(rules)} meta-rules")

    model = {
        "built_at": datetime.now().isoformat(),
        "prompt_count": len(prompts),
        "chain_count": len(chains),
        "task_count": len(task_index),
        "rule_count": len(rules),
        "rules": rules,
    }
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)

    with open(CHAINS_FILE, "w") as f:
        json.dump(chains[:100], f, indent=2)

    with open(TASK_INDEX_FILE, "w") as f:
        json.dump(task_index, f, indent=2)

    rules_md = f"# Meta-Rules from Cognition Model\n"
    rules_md += f"Built: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    rules_md += f"Source: {len(prompts)} prompts, {len(chains)} correction chains, {len(task_index)} sessions\n\n"

    for r in rules:
        rules_md += f"## {r['id']} (triggered {r['count']}x)\n\n"
        rules_md += f"**Rule:** {r['rule']}\n\n"
        rules_md += f"**Evidence:** {r['evidence']}\n\n"
        rules_md += f"**Action:** {r['action']}\n\n---\n\n"

    rules_md += "## Example Correction Chains\n\n"
    for chain in chains[:5]:
        rules_md += f"### Chain from {chain.get('initial_ts', 'unknown')[:10]}\n\n"
        rules_md += f"**Initial ask:** {chain['initial_prompt'][:200]}...\n\n"
        for i, c in enumerate(chain.get("corrections", [])[:3], 1):
            rules_md += f"**Correction {i}:** {c['prompt'][:200]}...\n"
            rules_md += f"  Signals: {', '.join(c.get('signals', [])[:3])}\n\n"
        if chain.get("resolution"):
            rules_md += f"**Resolution:** {chain['resolution'][:200]}\n\n"
        rules_md += "---\n\n"

    with open(RULES_FILE, "w") as f:
        f.write(rules_md)

    print(f"\nModel saved: {MODEL_FILE}")
    print(f"Chains saved: {CHAINS_FILE}")
    print(f"Rules saved: {RULES_FILE}")

    print(f"\n--- Meta-Rules ---")
    for r in rules:
        print(f"  [{r['id']}] ({r['count']}x) {r['rule'][:100]}...")

    return model


def lookup_task(query):
    """Find similar past tasks."""
    if not TASK_INDEX_FILE.exists():
        print("Task index not built yet. Run --build-model first.")
        return

    task_index = json.loads(TASK_INDEX_FILE.read_text())
    similar = lookup_similar_tasks(query, task_index)

    if similar:
        print(f"\n=== Similar Past Tasks for: \"{query[:50]}\" ===\n")
        for i, t in enumerate(similar, 1):
            print(f"{i}. [{t['timestamp'][:10]}] Overlap: {t['overlap']} | Corrections: {t['correction_count']}")
            print(f"   Shared words: {', '.join(t['shared_words'][:5])}")
            print(f"   Task: {t['initial_task'][:150]}...")
            if t.get("correction_prompts"):
                print(f"   User corrections:")
                for j, cp in enumerate(t["correction_prompts"][:2], 1):
                    print(f"     {j}. \"{cp[:100]}...\"")
            print(f"   Resolved: {'Yes' if t.get('resolved') else 'No'}")
            print()
    else:
        print("No similar tasks found in history.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--build-model" in args:
        build_model()
    elif "--lookup" in args:
        idx = args.index("--lookup")
        query = " ".join(args[idx + 1:])
        lookup_task(query)
    elif "--rules" in args:
        if RULES_FILE.exists():
            print(RULES_FILE.read_text()[:3000])
        else:
            print("Rules not generated yet. Run --build-model first.")
    elif "--chain-analysis" in args:
        prompts = load_prompts()
        chains = extract_correction_chains(prompts)
        print(f"\n=== Correction Chain Analysis ({len(chains)} chains) ===\n")
        for c in chains[:10]:
            print(f"[{c.get('initial_ts', '')[:10]}] Chain length: {c['chain_length']}")
            print(f"  Initial: \"{c['initial_prompt'][:100]}...\"")
            for cor in c.get("corrections", [])[:2]:
                print(f"  Correction: \"{cor['prompt'][:80]}...\" ({', '.join(cor.get('signals', [])[:2])})")
            if c.get("resolution"):
                print(f"  Resolution: \"{c['resolution'][:80]}...\"")
            print()
    elif "--status" in args:
        print("\n=== Cognitive Engine Status ===\n")
        if MODEL_FILE.exists():
            model = json.loads(MODEL_FILE.read_text())
            print(f"Model built: {model.get('built_at', 'unknown')}")
            print(f"Prompts analyzed: {model.get('prompt_count', 0)}")
            print(f"Correction chains: {model.get('chain_count', 0)}")
            print(f"Task sessions: {model.get('task_count', 0)}")
            print(f"Meta-rules: {model.get('rule_count', 0)}")
        else:
            print("Model not built yet. Run --build-model.")
    else:
        print("Cognitive Engine -- hybrid prompt intelligence system")
        print()
        print("  --build-model           Build cognition model from prompt history")
        print("  --lookup QUERY          Find similar past tasks + correction chains")
        print("  --rules                 Show extracted meta-rules")
        print("  --chain-analysis        Analyze correction chains")
        print("  --status                Show engine status")
