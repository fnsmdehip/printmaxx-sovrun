#!/usr/bin/env python3
"""
Conversation Index -- SQLite FTS5 full-text search for conversation history.

Indexes the JSONL output from conversation_logger.py into a SQLite FTS5
database for sub-second search across all logged conversations and prompts.

Usage:
    sovrun-conversations --index          # Build/update index from JSONL
    sovrun-conversations --search QUERY   # Search indexed conversations
    sovrun-conversations --stats          # Show index statistics
    sovrun-conversations --rebuild        # Full reindex from JSONL

Stdlib only (sqlite3 is stdlib).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

CONVERSATION_FILE = Path(os.environ.get(
    "SOVRUN_CONVERSATIONS", PROJECT_ROOT / "data" / "conversations.jsonl"))
INDEX_DB = Path(os.environ.get(
    "SOVRUN_CONVERSATION_INDEX", PROJECT_ROOT / "data" / "conversations.db"))
STATE_FILE = PROJECT_ROOT / "state" / "conversation_index_state.json"

LOGS_DIR = PROJECT_ROOT / "logs"


def safe_path(target: str | Path) -> Path:
    """Verify path is within project root."""
    resolved = Path(target).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"BLOCKED: {resolved} outside {root}")
    return resolved


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [CONV-INDEX] [{level}] {msg}")


# ---------------------------------------------------------------------------
# ConversationIndex
# ---------------------------------------------------------------------------

class ConversationIndex:
    """SQLite FTS5 index for conversation history.

    Indexes JSONL entries from conversation_logger.py with fields:
    timestamp, session_id, role, content.

    FTS5 provides ranked BM25 search across all conversation content.
    """

    def __init__(self, db_path: Path | None = None,
                 conversation_file: Path | None = None) -> None:
        self.db_path = db_path or INDEX_DB
        self.conversation_file = conversation_file or CONVERSATION_FILE
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
            CREATE TABLE IF NOT EXISTS conversations (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                content_length INTEGER DEFAULT 0,
                session_file TEXT DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                ts,
                session_id,
                role,
                content,
                content=conversations,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conversations BEGIN
                INSERT INTO conversations_fts(rowid, ts, session_id, role, content)
                VALUES (new.rowid, new.ts, new.session_id, new.role, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS conv_ad AFTER DELETE ON conversations BEGIN
                INSERT INTO conversations_fts(conversations_fts, rowid, ts, session_id, role, content)
                VALUES ('delete', old.rowid, old.ts, old.session_id, old.role, old.content);
            END;
        """)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------
    # index (incremental)
    # -------------------------------------------------------------------

    def index(self) -> int:
        """Incrementally index new entries from the conversation JSONL.

        Tracks byte offset in a state file to avoid reprocessing.
        Returns count of newly indexed entries.
        """
        if not self.conversation_file.exists():
            log(f"No conversation file at {self.conversation_file}", "WARN")
            return 0

        # Load state
        offset = 0
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                offset = state.get("offset", 0)
            except (json.JSONDecodeError, OSError):
                pass

        file_size = self.conversation_file.stat().st_size
        if file_size <= offset:
            log("No new entries to index")
            return 0

        conn = self._connect()
        indexed = 0

        with open(self.conversation_file, "r", encoding="utf-8",
                  errors="replace") as f:
            if offset > 0:
                f.seek(offset)

            batch: list[tuple[str, str, str, str, int, str]] = []

            while True:
                line = f.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = entry.get("ts", "")
                session_id = entry.get("session_id", "")
                role = entry.get("role", "")
                content = entry.get("content", "")
                content_length = entry.get("content_length", len(content))
                session_file = entry.get("session_file", "")

                if not content:
                    continue

                batch.append((
                    ts, session_id, role, content,
                    content_length, session_file,
                ))
                indexed += 1

                # Batch insert every 500 entries
                if len(batch) >= 500:
                    conn.executemany(
                        """INSERT INTO conversations
                           (ts, session_id, role, content,
                            content_length, session_file)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        batch,
                    )
                    conn.commit()
                    batch.clear()

            # Final batch
            if batch:
                conn.executemany(
                    """INSERT INTO conversations
                       (ts, session_id, role, content,
                        content_length, session_file)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                conn.commit()

            new_offset = f.tell()

        # Save state
        state_data = {"offset": new_offset, "last_indexed": indexed,
                      "last_run": datetime.now().isoformat(timespec="seconds")}
        STATE_FILE.write_text(json.dumps(state_data, indent=2))

        log(f"Indexed {indexed} new entries (offset {offset} -> {new_offset})")
        return indexed

    # -------------------------------------------------------------------
    # rebuild
    # -------------------------------------------------------------------

    def rebuild(self) -> int:
        """Drop and rebuild the entire index from the JSONL file."""
        conn = self._connect()
        conn.executescript("""
            DELETE FROM conversations;
            DELETE FROM conversations_fts;
        """)
        conn.commit()

        # Reset state
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            STATE_FILE.write_text("{}")

        log("Index cleared. Rebuilding...")
        return self.index()

    # -------------------------------------------------------------------
    # search
    # -------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Full-text search across indexed conversations.

        Returns ranked results with BM25 scoring.
        """
        if not query.strip():
            return []

        conn = self._connect()

        # Tokenize and build FTS5 query
        import re
        tokens = re.findall(r'\w+', query)
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        try:
            rows = conn.execute(
                """SELECT c.*, rank
                   FROM conversations_fts f
                   JOIN conversations c ON c.rowid = f.rowid
                   WHERE conversations_fts MATCH ?
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
                "ts": row["ts"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "content_length": row["content_length"],
                "rank": row["rank"],
            })

        return results

    # -------------------------------------------------------------------
    # stats
    # -------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        conn = self._connect()

        total = conn.execute(
            "SELECT COUNT(*) as c FROM conversations"
        ).fetchone()["c"]

        if total == 0:
            return {
                "total_entries": 0,
                "user_count": 0,
                "assistant_count": 0,
                "sessions": 0,
                "date_range": {"earliest": None, "latest": None},
                "db_size_kb": 0,
            }

        user_count = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE role IN ('user', 'human')"
        ).fetchone()["c"]

        asst_count = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE role = 'assistant'"
        ).fetchone()["c"]

        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) as c FROM conversations"
        ).fetchone()["c"]

        date_range = conn.execute(
            "SELECT MIN(ts) as earliest, MAX(ts) as latest FROM conversations"
        ).fetchone()

        # Role distribution
        roles = conn.execute(
            "SELECT role, COUNT(*) as c FROM conversations GROUP BY role ORDER BY c DESC"
        ).fetchall()

        db_size = 0
        try:
            db_size = self.db_path.stat().st_size // 1024
        except OSError:
            pass

        return {
            "total_entries": total,
            "user_count": user_count,
            "assistant_count": asst_count,
            "sessions": sessions,
            "date_range": {
                "earliest": date_range["earliest"],
                "latest": date_range["latest"],
            },
            "role_distribution": {r["role"]: r["c"] for r in roles},
            "db_size_kb": db_size,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversation Index -- FTS5 search for conversation history"
    )
    parser.add_argument("--index", action="store_true",
                        help="Build/update index from conversation JSONL")
    parser.add_argument("--search", type=str, metavar="QUERY",
                        help="Search indexed conversations")
    parser.add_argument("--stats", action="store_true",
                        help="Show index statistics")
    parser.add_argument("--rebuild", action="store_true",
                        help="Full reindex from JSONL (drops existing)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of results for search (default 10)")

    args = parser.parse_args()

    if not any([args.index, args.search, args.stats, args.rebuild]):
        parser.print_help()
        sys.exit(1)

    idx = ConversationIndex()

    try:
        if args.rebuild:
            log("Rebuilding full index...")
            count = idx.rebuild()
            print(f"\nRebuilt index: {count} entries indexed.")

        elif args.index:
            count = idx.index()
            print(f"\nIndexed {count} new entries.")

        elif args.search:
            results = idx.search(args.search, top_k=args.top)
            if results:
                print(f"\n=== Search: \"{args.search}\" "
                      f"({len(results)} results) ===\n")
                for i, r in enumerate(results, 1):
                    ts = r["ts"][:19] if r["ts"] else "?"
                    role = r["role"].upper()
                    session = r["session_id"][:8] if r["session_id"] else "?"
                    content = r["content"][:200].replace("\n", " ")
                    if len(r["content"]) > 200:
                        content += "..."
                    print(f"  {i}. [{ts}] {role} (session:{session})")
                    print(f"     {content}")
                    print()
            else:
                print("No matching conversations found.")

        elif args.stats:
            s = idx.stats()
            print(f"\n{'='*50}")
            print("CONVERSATION INDEX STATS")
            print(f"{'='*50}")
            print(f"  Total entries:      {s['total_entries']:,}")
            print(f"  User messages:      {s['user_count']:,}")
            print(f"  Assistant messages:  {s['assistant_count']:,}")
            print(f"  Unique sessions:     {s['sessions']:,}")
            dr = s["date_range"]
            earliest = (dr["earliest"] or "N/A")[:19]
            latest = (dr["latest"] or "N/A")[:19]
            print(f"  Date range:          {earliest} to {latest}")
            print(f"  DB size:             {s['db_size_kb']} KB")
            if s.get("role_distribution"):
                print(f"\n  Role distribution:")
                for role, count in s["role_distribution"].items():
                    print(f"    {role}: {count:,}")
            print(f"{'='*50}")

    finally:
        idx.close()


if __name__ == "__main__":
    main()
