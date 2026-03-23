#!/usr/bin/env python3
"""
Procedural Memory -- Skill document system for autonomous agents.

Agents learn by doing. This module captures successful task completions as
reusable skill documents, indexes them with SQLite FTS5 for sub-second recall,
and consolidates conversation history into new skills automatically.

The core loop: DO -> CAPTURE -> INDEX -> RECALL -> DO BETTER.

When an agent faces a new task, it queries procedural memory first. If a
matching skill exists, the agent gets solution steps and context injected
into its prompt -- no correction chain needed.

Usage:
    python3 -m sovrun.core.procedural_memory --capture "task" --result "outcome"
    python3 -m sovrun.core.procedural_memory --recall "query"
    python3 -m sovrun.core.procedural_memory --consolidate
    python3 -m sovrun.core.procedural_memory --stats
    python3 -m sovrun.core.procedural_memory --inject "query"
    python3 -m sovrun.core.procedural_memory --prune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

SKILLS_DB = Path(os.environ.get(
    "SOVRUN_SKILLS_DB", PROJECT_ROOT / "data" / "skills.db"))
CONVERSATION_FILE = Path(os.environ.get(
    "SOVRUN_CONVERSATIONS", PROJECT_ROOT / "data" / "conversations.jsonl"))
AUDIT_LOG = Path(os.environ.get(
    "SOVRUN_SKILLS_AUDIT", PROJECT_ROOT / "logs" / "skills_audit.jsonl"))

LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def safe_path(target: str | Path) -> Path:
    """Verify path is within project root. Raises ValueError if not."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} is outside project root {root}")
    return resolved


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [MEMORY] [{level}] {msg}")


def audit(action: str, details: dict[str, Any]) -> None:
    """Append an entry to the skills audit trail."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        **details,
    }
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# SkillDocument
# ---------------------------------------------------------------------------

@dataclass
class SkillDocument:
    """A single learned skill captured from a successful task completion."""

    skill_id: str
    title: str
    problem_description: str
    solution_steps: list[str]
    context_tags: list[str]
    success_count: int = 1
    last_used: str = ""
    created_at: str = ""
    source_session: str = ""
    confidence: float = 0.5

    def __post_init__(self) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        if not self.last_used:
            self.last_used = now
        if not self.skill_id:
            raw = f"{self.title}:{self.problem_description}:{now}"
            self.skill_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def steps_text(self) -> str:
        """Flat text of solution steps for indexing."""
        return " | ".join(self.solution_steps)

    def tags_text(self) -> str:
        """Flat text of context tags for indexing."""
        return " ".join(self.context_tags)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillDocument:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Satisfaction / success detection for consolidation
# ---------------------------------------------------------------------------

SATISFACTION_SIGNALS = [
    "perfect", "exactly", "yes that", "ship it", "love it", "fire",
    "nice", "great", "solid", "nailed", "good", "appreciate",
]

MULTI_STEP_SIGNALS = [
    "step 1", "step 2", "first,", "second,", "then ", "finally,",
    "next,", "after that", "1.", "2.", "3.",
]


def _looks_satisfied(text: str) -> bool:
    """Check if text contains satisfaction signals."""
    lower = text.lower()[:500]
    return any(sig in lower for sig in SATISFACTION_SIGNALS)


def _looks_multi_step(text: str) -> bool:
    """Check if text contains multi-step task indicators."""
    lower = text.lower()[:2000]
    hits = sum(1 for sig in MULTI_STEP_SIGNALS if sig in lower)
    return hits >= 2


def _extract_steps_from_text(text: str) -> list[str]:
    """Pull numbered or sequential steps from assistant text."""
    steps = []

    # Try numbered list extraction (1. ... 2. ... etc)
    numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)', text, re.MULTILINE)
    if len(numbered) >= 2:
        return [s.strip()[:200] for s in numbered[:10]]

    # Try bullet extraction
    bullets = re.findall(r'^\s*[-*]\s+(.+)', text, re.MULTILINE)
    if len(bullets) >= 2:
        return [s.strip()[:200] for s in bullets[:10]]

    # Fallback: split into sentences and take first few meaningful ones
    sentences = re.split(r'[.!?\n]', text)
    for sent in sentences:
        sent = sent.strip()
        if 15 < len(sent) < 300:
            steps.append(sent)
        if len(steps) >= 5:
            break

    return steps


def _extract_tags(text: str) -> list[str]:
    """Extract context tags from text via keyword extraction."""
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stopwords = {
        "that", "this", "with", "from", "have", "like", "just", "what",
        "make", "sure", "also", "want", "need", "think", "about", "them",
        "their", "they", "would", "could", "should", "every", "some",
        "other", "then", "when", "into", "more", "been", "will", "your",
        "does", "know", "good", "best", "here", "there", "very",
        "only", "same", "such", "than", "each", "both",
    }
    from collections import Counter
    counts = Counter(w for w in words if w not in stopwords)
    return [w for w, _ in counts.most_common(8)]


def _generate_title(task: str) -> str:
    """Generate a concise title from a task description."""
    # Take first sentence or first 80 chars
    first_sentence = re.split(r'[.!?\n]', task)[0].strip()
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:77] + "..."
    return first_sentence


# ---------------------------------------------------------------------------
# ProceduralMemory -- main class
# ---------------------------------------------------------------------------

class ProceduralMemory:
    """Skill document store with FTS5 full-text search."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or SKILLS_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                problem_description TEXT NOT NULL,
                solution_steps TEXT NOT NULL,
                context_tags TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                last_used TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_session TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                skill_id,
                title,
                problem_description,
                solution_steps,
                context_tags,
                content=skills,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
                INSERT INTO skills_fts(rowid, skill_id, title, problem_description, solution_steps, context_tags)
                VALUES (new.rowid, new.skill_id, new.title, new.problem_description, new.solution_steps, new.context_tags);
            END;

            CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
                INSERT INTO skills_fts(skills_fts, rowid, skill_id, title, problem_description, solution_steps, context_tags)
                VALUES ('delete', old.rowid, old.skill_id, old.title, old.problem_description, old.solution_steps, old.context_tags);
            END;

            CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
                INSERT INTO skills_fts(skills_fts, rowid, skill_id, title, problem_description, solution_steps, context_tags)
                VALUES ('delete', old.rowid, old.skill_id, old.title, old.problem_description, old.solution_steps, old.context_tags);
                INSERT INTO skills_fts(rowid, skill_id, title, problem_description, solution_steps, context_tags)
                VALUES (new.rowid, new.skill_id, new.title, new.problem_description, new.solution_steps, new.context_tags);
            END;
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------
    # capture
    # -------------------------------------------------------------------

    def capture(self, task: str, result: str, success: bool = True,
                source_session: str = "", confidence: float | None = None) -> SkillDocument | None:
        """Capture a completed task as a skill document.

        Only captures successful completions. Extracts solution steps from
        the result text and indexes everything for later recall.
        """
        if not success:
            audit("capture_skipped", {"reason": "not_successful", "task": task[:200]})
            return None

        if len(task.strip()) < 10:
            audit("capture_skipped", {"reason": "task_too_short", "task": task})
            return None

        title = _generate_title(task)
        steps = _extract_steps_from_text(result)
        if not steps:
            steps = [result[:300]]
        tags = _extract_tags(task + " " + result)

        conf = confidence if confidence is not None else (0.6 if len(steps) >= 3 else 0.4)

        doc = SkillDocument(
            skill_id="",
            title=title,
            problem_description=task[:1000],
            solution_steps=steps,
            context_tags=tags,
            source_session=source_session,
            confidence=conf,
        )

        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO skills
                   (skill_id, title, problem_description, solution_steps,
                    context_tags, success_count, last_used, created_at,
                    source_session, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc.skill_id, doc.title, doc.problem_description,
                 doc.steps_text(), doc.tags_text(),
                 doc.success_count, doc.last_used, doc.created_at,
                 doc.source_session, doc.confidence),
            )
            conn.commit()
        except sqlite3.Error as e:
            log(f"Failed to insert skill: {e}", "ERROR")
            return None

        audit("captured", {"skill_id": doc.skill_id, "title": doc.title})
        log(f"Captured skill: {doc.title} ({doc.skill_id})")
        return doc

    # -------------------------------------------------------------------
    # recall
    # -------------------------------------------------------------------

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Full-text search across all skill documents.

        Returns ranked matches with BM25 scoring.
        """
        if not query.strip():
            return []

        conn = self._connect()

        # Sanitize query for FTS5: remove special chars, wrap tokens in quotes
        tokens = re.findall(r'\w+', query)
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        try:
            rows = conn.execute(
                """SELECT s.*, rank
                   FROM skills_fts f
                   JOIN skills s ON s.rowid = f.rowid
                   WHERE skills_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log(f"FTS query failed: {e}", "WARN")
            return []

        results = []
        for row in rows:
            results.append({
                "skill_id": row["skill_id"],
                "title": row["title"],
                "problem_description": row["problem_description"],
                "solution_steps": row["solution_steps"].split(" | "),
                "context_tags": row["context_tags"].split(),
                "success_count": row["success_count"],
                "confidence": row["confidence"],
                "last_used": row["last_used"],
                "created_at": row["created_at"],
                "rank": row["rank"],
            })

        if results:
            audit("recall", {"query": query[:200], "results": len(results)})

        return results

    # -------------------------------------------------------------------
    # consolidate
    # -------------------------------------------------------------------

    def consolidate(self, conversation_file: Path | None = None) -> int:
        """Scan conversation logs, extract successful problem-solution pairs,
        generate new skill documents automatically.

        Looks for episodes where:
        - User expressed satisfaction (perfect, exactly, ship it)
        - Assistant completed a multi-step task
        """
        conv_file = conversation_file or CONVERSATION_FILE
        if not conv_file.exists():
            log(f"No conversation file at {conv_file}", "WARN")
            return 0

        entries = []
        with open(conv_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not entries:
            log("No conversation entries found")
            return 0

        log(f"Scanning {len(entries)} conversation entries for skill candidates...")

        # Group into windows of user-assistant pairs
        captured = 0
        window: list[dict] = []
        prev_ts: str = ""

        for entry in entries:
            ts = entry.get("ts", "")
            role = entry.get("role", "")

            # New window if time gap > 10 min
            gap_minutes = float("inf")
            if prev_ts and ts:
                try:
                    t1 = datetime.fromisoformat(prev_ts)
                    t2 = datetime.fromisoformat(ts)
                    gap_minutes = abs((t2 - t1).total_seconds()) / 60
                except (ValueError, TypeError):
                    pass

            if gap_minutes > 10 and window:
                captured += self._process_window(window)
                window = []

            window.append(entry)
            prev_ts = ts

        if window:
            captured += self._process_window(window)

        log(f"Consolidation complete: {captured} new skills captured")
        audit("consolidate", {"entries_scanned": len(entries), "skills_captured": captured})
        return captured

    def _process_window(self, window: list[dict]) -> int:
        """Process a conversation window for skill extraction."""
        user_msgs = [e for e in window if e.get("role") in ("user", "human")]
        asst_msgs = [e for e in window if e.get("role") == "assistant"]

        if not user_msgs or not asst_msgs:
            return 0

        # Check if user expressed satisfaction in last few messages
        last_user_msgs = user_msgs[-3:]
        satisfied = any(
            _looks_satisfied(m.get("content", ""))
            for m in last_user_msgs
        )

        # Check if assistant produced multi-step output
        multi_step = any(
            _looks_multi_step(m.get("content", ""))
            for m in asst_msgs
        )

        if not (satisfied or multi_step):
            return 0

        # Extract the problem (first user message) and solution (last assistant message with steps)
        problem = user_msgs[0].get("content", "")[:1000]
        if not problem or len(problem.strip()) < 10:
            return 0

        # Find the best assistant response (longest with steps)
        best_response = ""
        for m in reversed(asst_msgs):
            content = m.get("content", "")
            if _looks_multi_step(content) or len(content) > len(best_response):
                best_response = content
                if _looks_multi_step(content):
                    break

        if not best_response:
            return 0

        # Check for duplicate (same problem description hash)
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
        conn = self._connect()
        existing = conn.execute(
            "SELECT skill_id FROM skills WHERE skill_id = ?",
            (problem_hash,),
        ).fetchone()

        if existing:
            return 0

        session_id = window[0].get("session_id", "")
        conf = 0.7 if satisfied else 0.5

        doc = self.capture(
            task=problem,
            result=best_response[:3000],
            success=True,
            source_session=session_id,
            confidence=conf,
        )

        return 1 if doc else 0

    # -------------------------------------------------------------------
    # improve
    # -------------------------------------------------------------------

    def improve(self, skill_id: str, new_outcome: str) -> bool:
        """Update a skill doc based on new usage.

        Increments success_count, updates last_used, and optionally
        refines solution_steps if the new outcome has better steps.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM skills WHERE skill_id = ?", (skill_id,)
        ).fetchone()

        if not row:
            log(f"Skill {skill_id} not found", "WARN")
            return False

        new_steps = _extract_steps_from_text(new_outcome)
        existing_steps = row["solution_steps"]

        # If new outcome has more detailed steps, use those
        if new_steps and len(new_steps) > len(existing_steps.split(" | ")):
            steps_text = " | ".join(new_steps)
        else:
            steps_text = existing_steps

        new_count = row["success_count"] + 1
        new_confidence = min(1.0, row["confidence"] + 0.05)
        now = datetime.now().isoformat(timespec="seconds")

        conn.execute(
            """UPDATE skills
               SET success_count = ?, last_used = ?, confidence = ?, solution_steps = ?
               WHERE skill_id = ?""",
            (new_count, now, new_confidence, steps_text, skill_id),
        )
        conn.commit()

        audit("improved", {
            "skill_id": skill_id,
            "new_count": new_count,
            "new_confidence": new_confidence,
        })
        log(f"Improved skill {skill_id}: count={new_count}, confidence={new_confidence:.2f}")
        return True

    # -------------------------------------------------------------------
    # prune
    # -------------------------------------------------------------------

    def prune(self, min_confidence: float = 0.3, max_age_days: int = 90) -> int:
        """Remove low-confidence, unused skill docs."""
        conn = self._connect()
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

        rows = conn.execute(
            """SELECT skill_id, title, confidence, last_used
               FROM skills
               WHERE confidence < ? AND last_used < ?""",
            (min_confidence, cutoff),
        ).fetchall()

        if not rows:
            log("No skills to prune")
            return 0

        ids = [r["skill_id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM skills WHERE skill_id IN ({placeholders})", ids)
        conn.commit()

        for r in rows:
            audit("pruned", {
                "skill_id": r["skill_id"],
                "title": r["title"],
                "confidence": r["confidence"],
                "last_used": r["last_used"],
            })

        log(f"Pruned {len(rows)} stale skills (confidence < {min_confidence}, unused > {max_age_days}d)")
        return len(rows)

    # -------------------------------------------------------------------
    # export_injection
    # -------------------------------------------------------------------

    def export_injection(self, query: str, max_chars: int = 600) -> str:
        """Return a compact string suitable for prepending to an agent prompt.

        Searches for relevant skills and formats them as injection context.
        """
        results = self.recall(query, top_k=3)
        if not results:
            return ""

        parts = []
        for r in results:
            steps = r["solution_steps"]
            if isinstance(steps, list):
                steps_str = "; ".join(s[:80] for s in steps[:4])
            else:
                steps_str = str(steps)[:200]
            parts.append(f"[{r['title'][:60]}] Steps: {steps_str}")

        injection = "PROCEDURAL MEMORY: " + " | ".join(parts)

        if len(injection) > max_chars:
            injection = injection[:max_chars - 3] + "..."

        return injection

    # -------------------------------------------------------------------
    # stats
    # -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return statistics about the skill store."""
        conn = self._connect()

        total = conn.execute("SELECT COUNT(*) as c FROM skills").fetchone()["c"]
        if total == 0:
            return {
                "total_skills": 0,
                "avg_confidence": 0,
                "avg_success_count": 0,
                "most_used": [],
                "highest_confidence": [],
                "db_size_kb": 0,
            }

        avg_conf = conn.execute(
            "SELECT AVG(confidence) as a FROM skills"
        ).fetchone()["a"]

        avg_success = conn.execute(
            "SELECT AVG(success_count) as a FROM skills"
        ).fetchone()["a"]

        most_used = conn.execute(
            """SELECT skill_id, title, success_count, confidence
               FROM skills ORDER BY success_count DESC LIMIT 5"""
        ).fetchall()

        highest_conf = conn.execute(
            """SELECT skill_id, title, success_count, confidence
               FROM skills ORDER BY confidence DESC LIMIT 5"""
        ).fetchall()

        db_size = 0
        try:
            db_size = self.db_path.stat().st_size // 1024
        except OSError:
            pass

        return {
            "total_skills": total,
            "avg_confidence": round(avg_conf or 0, 3),
            "avg_success_count": round(avg_success or 0, 1),
            "most_used": [dict(r) for r in most_used],
            "highest_confidence": [dict(r) for r in highest_conf],
            "db_size_kb": db_size,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procedural Memory: skill document system for autonomous agents"
    )
    parser.add_argument("--capture", type=str, metavar="TASK",
                        help="Capture a skill from task description")
    parser.add_argument("--result", type=str, metavar="RESULT",
                        help="Result/outcome of the captured task")
    parser.add_argument("--recall", type=str, metavar="QUERY",
                        help="Search skills by query")
    parser.add_argument("--consolidate", action="store_true",
                        help="Run consolidation pipeline on conversation logs")
    parser.add_argument("--stats", action="store_true",
                        help="Show skill store statistics")
    parser.add_argument("--inject", type=str, metavar="QUERY",
                        help="Get injection string for agent prompts")
    parser.add_argument("--prune", action="store_true",
                        help="Remove stale/low-confidence skills")
    parser.add_argument("--improve", type=str, metavar="SKILL_ID",
                        help="Improve a skill with new outcome")
    parser.add_argument("--outcome", type=str, metavar="TEXT",
                        help="New outcome text for --improve")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Min confidence for pruning (default 0.3)")
    parser.add_argument("--max-age", type=int, default=90,
                        help="Max age in days for pruning (default 90)")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of results for recall (default 5)")

    args = parser.parse_args()

    if not any([args.capture, args.recall, args.consolidate,
                args.stats, args.inject, args.prune, args.improve]):
        parser.print_help()
        sys.exit(1)

    mem = ProceduralMemory()

    try:
        if args.capture:
            result = args.result or "Completed successfully"
            doc = mem.capture(task=args.capture, result=result)
            if doc:
                print(f"\nCaptured skill: {doc.skill_id}")
                print(f"  Title: {doc.title}")
                print(f"  Steps: {len(doc.solution_steps)}")
                print(f"  Tags: {', '.join(doc.context_tags)}")
                print(f"  Confidence: {doc.confidence}")
            else:
                print("Capture failed or skipped.")

        elif args.recall:
            results = mem.recall(args.recall, top_k=args.top)
            if results:
                print(f"\n=== Skills matching: \"{args.recall}\" ===\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. [{r['skill_id']}] {r['title']}")
                    print(f"   Confidence: {r['confidence']:.2f} | Used: {r['success_count']}x")
                    steps = r["solution_steps"]
                    if isinstance(steps, list):
                        for j, step in enumerate(steps[:4], 1):
                            print(f"   {j}. {step[:120]}")
                    else:
                        print(f"   Steps: {str(steps)[:200]}")
                    print(f"   Tags: {', '.join(r['context_tags'][:6])}")
                    print()
            else:
                print("No matching skills found.")

        elif args.consolidate:
            print("Running consolidation pipeline...")
            count = mem.consolidate()
            print(f"\nConsolidated {count} new skills from conversation history.")

        elif args.stats:
            s = mem.stats()
            print(f"\n{'='*50}")
            print("PROCEDURAL MEMORY STATS")
            print(f"{'='*50}")
            print(f"  Total skills:      {s['total_skills']}")
            print(f"  Avg confidence:    {s['avg_confidence']:.3f}")
            print(f"  Avg success count: {s['avg_success_count']:.1f}")
            print(f"  DB size:           {s['db_size_kb']} KB")

            if s["most_used"]:
                print(f"\n  Most used:")
                for r in s["most_used"]:
                    print(f"    [{r['skill_id']}] {r['title'][:50]} (x{r['success_count']}, conf={r['confidence']:.2f})")

            if s["highest_confidence"]:
                print(f"\n  Highest confidence:")
                for r in s["highest_confidence"]:
                    print(f"    [{r['skill_id']}] {r['title'][:50]} (conf={r['confidence']:.2f}, x{r['success_count']})")

            print(f"{'='*50}")

        elif args.inject:
            injection = mem.export_injection(args.inject)
            if injection:
                print(injection)
            else:
                print("[No matching skills found]")

        elif args.prune:
            count = mem.prune(
                min_confidence=args.min_confidence,
                max_age_days=args.max_age,
            )
            print(f"Pruned {count} stale skills.")

        elif args.improve:
            outcome = args.outcome or "Reused successfully"
            ok = mem.improve(args.improve, outcome)
            if ok:
                print(f"Skill {args.improve} improved.")
            else:
                print(f"Skill {args.improve} not found.")

    finally:
        mem.close()


if __name__ == "__main__":
    main()
