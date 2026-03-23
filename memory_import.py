#!/usr/bin/env python3
"""
Memory Import -- Import conversation history from ANY LLM provider into sovrun.

Parses exports from ChatGPT (conversations.json), Claude Code (JSONL transcripts),
Google Gemini (Takeout), and generic JSONL/CSV formats. Deduplicates by content
hash, tracks incremental imports, and outputs to the same JSONL format as
conversation_logger.py.

Usage:
    sovrun-import --import FILE                  # Import a single file
    sovrun-import --import-dir DIR               # Import all files in a directory
    sovrun-import --provider chatgpt --import F  # Force a specific parser
    sovrun-import --detect FILE                  # Detect format without importing
    sovrun-import --build                        # Build cognitive model from imports
    sovrun-import --stats                        # Show import statistics

Supported formats:
    chatgpt    - conversations.json from ChatGPT data export (ZIP or JSON)
    claude     - Claude Code JSONL transcripts (~/.claude/projects/)
    gemini     - Google Takeout Gemini export (JSON)
    jsonl      - Generic JSONL with role/content fields
    csv        - Generic CSV with role/content columns

No external dependencies. stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Paths (configurable via environment)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SOVRUN_ROOT", Path.cwd()))

OUTPUT_FILE = Path(os.environ.get(
    "SOVRUN_CONVERSATIONS", PROJECT_ROOT / "data" / "conversations.jsonl"))
STATE_FILE = PROJECT_ROOT / "state" / "memory_import_state.json"
COGNITIVE_MODEL_FILE = PROJECT_ROOT / "output" / "cognitive_model.json"

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
    print(f"[{ts}] [IMPORT] [{level}] {msg}")


def log_to_file(msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "memory_import.log"
    with open(path, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConversationEntry:
    """Single message from any LLM provider."""
    role: str                           # user, assistant, system
    content: str                        # message text
    timestamp: str = ""                 # ISO 8601 or empty
    provider: str = "unknown"           # chatgpt, claude, gemini, generic
    session_id: str = ""                # conversation/session identifier
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of role + content for dedup."""
        raw = f"{self.role}:{self.content}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Convert to the JSONL format used by conversation_logger."""
        d: dict[str, Any] = {
            "ts": self.timestamp or datetime.now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "content_length": len(self.content),
            "provider": self.provider,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# State tracking (dedup + incremental)
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load import state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "imported_files": {},
        "seen_hashes": [],
        "total_imported": 0,
        "total_duplicates_skipped": 0,
        "last_run": None,
    }


def save_state(state: dict[str, Any]) -> None:
    """Persist import state."""
    state["last_run"] = datetime.now().isoformat(timespec="seconds")
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_seen_hashes(state: dict[str, Any]) -> set[str]:
    """Load set of previously seen content hashes."""
    return set(state.get("seen_hashes", []))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class ChatGPTParser:
    """Parse ChatGPT data export (conversations.json or ZIP containing it).

    ChatGPT export format: ZIP containing conversations.json with an array of
    conversation objects, each having a "mapping" dict where values have
    "message" objects with "author.role" and "content.parts".
    """

    provider = "chatgpt"

    @staticmethod
    def can_parse(filepath: Path) -> bool:
        """Check if file looks like a ChatGPT export."""
        name = filepath.name.lower()

        # ZIP export
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(filepath, "r") as zf:
                    return "conversations.json" in zf.namelist()
            except (zipfile.BadZipFile, OSError):
                return False

        # Direct conversations.json
        if name == "conversations.json":
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    # Read first bytes to check structure
                    head = f.read(512)
                    return '"mapping"' in head or '"title"' in head
            except OSError:
                return False

        return False

    @staticmethod
    def parse(filepath: Path) -> Iterator[ConversationEntry]:
        """Parse ChatGPT export and yield ConversationEntry objects."""
        data = None

        if filepath.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(filepath, "r") as zf:
                    with zf.open("conversations.json") as cf:
                        raw = cf.read().decode("utf-8", errors="replace")
                        data = json.loads(raw)
            except (zipfile.BadZipFile, json.JSONDecodeError, KeyError, OSError) as e:
                log(f"Failed to parse ChatGPT ZIP: {e}", "ERROR")
                return
        else:
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log(f"Failed to parse ChatGPT JSON: {e}", "ERROR")
                return

        if not isinstance(data, list):
            log("ChatGPT export: expected array of conversations", "ERROR")
            return

        for conversation in data:
            if not isinstance(conversation, dict):
                continue

            conv_id = conversation.get("id", conversation.get("title", ""))
            conv_title = conversation.get("title", "")
            create_time = conversation.get("create_time")

            mapping = conversation.get("mapping", {})
            if not isinstance(mapping, dict):
                continue

            # Collect messages in order
            messages: list[tuple[float | None, ConversationEntry]] = []

            for _node_id, node in mapping.items():
                if not isinstance(node, dict):
                    continue
                message = node.get("message")
                if not isinstance(message, dict):
                    continue

                author = message.get("author", {})
                if not isinstance(author, dict):
                    continue
                role = author.get("role", "")
                if role not in ("user", "assistant", "system"):
                    continue

                content_obj = message.get("content", {})
                if not isinstance(content_obj, dict):
                    continue

                parts = content_obj.get("parts", [])
                if not isinstance(parts, list):
                    continue

                text_parts = []
                for part in parts:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])

                text = "\n".join(text_parts).strip()
                if not text:
                    continue

                msg_create = message.get("create_time")
                ts = ""
                sort_time = msg_create
                if msg_create:
                    try:
                        ts = datetime.fromtimestamp(float(msg_create)).isoformat(timespec="seconds")
                    except (ValueError, TypeError, OSError):
                        ts = ""
                        sort_time = None

                entry = ConversationEntry(
                    role=role,
                    content=text,
                    timestamp=ts,
                    provider="chatgpt",
                    session_id=str(conv_id)[:64] if conv_id else "",
                    metadata={"title": conv_title} if conv_title else {},
                )
                messages.append((sort_time, entry))

            # Sort by creation time within this conversation
            messages.sort(key=lambda x: x[0] if x[0] is not None else 0)
            for _t, entry in messages:
                yield entry


class ClaudeTranscriptParser:
    """Parse Claude Code JSONL transcripts.

    Format: JSONL at ~/.claude/projects/*/*.jsonl with type/message fields
    where message has role and content.
    """

    provider = "claude"

    @staticmethod
    def can_parse(filepath: Path) -> bool:
        """Check if file looks like a Claude Code transcript."""
        if not filepath.name.lower().endswith(".jsonl"):
            return False
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Claude format has "type" field with "user" or "assistant"
                        if isinstance(obj, dict) and "type" in obj:
                            return True
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return False
        return False

    @staticmethod
    def parse(filepath: Path) -> Iterator[ConversationEntry]:
        """Parse Claude Code JSONL transcript."""
        session_id = filepath.stem

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    msg_type = obj.get("type", "")
                    if msg_type not in ("user", "assistant"):
                        continue

                    message = obj.get("message", {})
                    if not isinstance(message, dict):
                        continue

                    role = message.get("role", msg_type)
                    content_raw = message.get("content", "")
                    text = _extract_text_content(content_raw)

                    if not text:
                        continue

                    timestamp = obj.get("timestamp", "")
                    if not timestamp:
                        timestamp = ""

                    yield ConversationEntry(
                        role=role,
                        content=text,
                        timestamp=timestamp,
                        provider="claude",
                        session_id=session_id,
                    )
        except (OSError, PermissionError) as e:
            log(f"Could not read {filepath}: {e}", "WARN")


class GeminiParser:
    """Parse Google Takeout Gemini export.

    Gemini Takeout exports are JSON files with conversation objects containing
    a list of entries with "text" and "role" fields, or similar structures.
    """

    provider = "gemini"

    @staticmethod
    def can_parse(filepath: Path) -> bool:
        """Check if file looks like a Gemini export."""
        name = filepath.name.lower()
        if not name.endswith(".json"):
            return False
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(2048)
                # Gemini Takeout markers
                if "Gemini" in head or "bard" in head.lower():
                    return True
                # Check for Google AI Studio format
                if '"candidates"' in head or '"promptFeedback"' in head:
                    return True
                # Check for Takeout conversation structure
                if '"parts"' in head and '"role"' in head:
                    return True
        except OSError:
            return False
        return False

    @staticmethod
    def parse(filepath: Path) -> Iterator[ConversationEntry]:
        """Parse Gemini/Bard export."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log(f"Failed to parse Gemini export: {e}", "ERROR")
            return

        # Handle different Gemini export structures
        conversations = []
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            # Single conversation or wrapper
            if "conversations" in data:
                conversations = data["conversations"]
            elif "contents" in data:
                # Google AI Studio format
                conversations = [data]
            else:
                conversations = [data]

        for conv in conversations:
            if not isinstance(conv, dict):
                continue

            conv_id = conv.get("id", conv.get("title", ""))

            # Google AI Studio / Gemini API format: contents array
            contents = conv.get("contents", [])
            if isinstance(contents, list):
                for content_obj in contents:
                    if not isinstance(content_obj, dict):
                        continue
                    role = content_obj.get("role", "user")
                    if role == "model":
                        role = "assistant"
                    parts = content_obj.get("parts", [])
                    text_parts = []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    text = "\n".join(text_parts).strip()
                    if text:
                        yield ConversationEntry(
                            role=role,
                            content=text,
                            provider="gemini",
                            session_id=str(conv_id)[:64] if conv_id else "",
                        )
                continue

            # Takeout format: entries/messages array
            entries = conv.get("entries", conv.get("messages", []))
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    role = entry.get("role", "user")
                    if role == "model":
                        role = "assistant"
                    text = entry.get("text", entry.get("content", ""))
                    if isinstance(text, str) and text.strip():
                        ts = entry.get("timestamp", entry.get("create_time", ""))
                        yield ConversationEntry(
                            role=role,
                            content=text.strip(),
                            timestamp=str(ts) if ts else "",
                            provider="gemini",
                            session_id=str(conv_id)[:64] if conv_id else "",
                        )


class GenericJSONLParser:
    """Parse generic JSONL with role/content fields.

    Expects each line to be a JSON object with at least 'role' and 'content'
    (or 'text', 'message') fields.
    """

    provider = "generic"

    @staticmethod
    def can_parse(filepath: Path) -> bool:
        """Check if file is parseable JSONL."""
        if not filepath.name.lower().endswith(".jsonl"):
            return False
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # Must have some content field
                            has_content = any(k in obj for k in
                                              ("content", "text", "message", "prompt"))
                            if has_content:
                                return True
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return False
        return False

    @staticmethod
    def parse(filepath: Path) -> Iterator[ConversationEntry]:
        """Parse generic JSONL file."""
        session_id = filepath.stem

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    role = obj.get("role", "user")
                    content = (obj.get("content")
                               or obj.get("text")
                               or obj.get("message")
                               or obj.get("prompt")
                               or "")

                    if isinstance(content, list):
                        content = _extract_text_content(content)
                    elif not isinstance(content, str):
                        content = str(content)

                    content = content.strip()
                    if not content:
                        continue

                    ts = obj.get("ts", obj.get("timestamp", obj.get("created_at", "")))

                    yield ConversationEntry(
                        role=role,
                        content=content,
                        timestamp=str(ts) if ts else "",
                        provider="generic",
                        session_id=obj.get("session_id", session_id),
                    )
        except (OSError, PermissionError) as e:
            log(f"Could not read {filepath}: {e}", "WARN")


class GenericCSVParser:
    """Parse generic CSV with role/content columns.

    Expects a CSV with at least a 'content' (or 'text', 'message') column.
    Optionally has 'role', 'timestamp' columns.
    """

    provider = "generic"

    @staticmethod
    def can_parse(filepath: Path) -> bool:
        """Check if file is a parseable CSV."""
        if not filepath.name.lower().endswith(".csv"):
            return False
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return False
                lower_header = [h.lower().strip() for h in header]
                return any(col in lower_header for col in
                           ("content", "text", "message", "prompt"))
        except (OSError, csv.Error):
            return False

    @staticmethod
    def parse(filepath: Path) -> Iterator[ConversationEntry]:
        """Parse generic CSV file."""
        session_id = filepath.stem

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)

                # Normalize field names
                for row in reader:
                    lower_row = {k.lower().strip(): v for k, v in row.items() if k}

                    content = (lower_row.get("content")
                               or lower_row.get("text")
                               or lower_row.get("message")
                               or lower_row.get("prompt")
                               or "")

                    content = content.strip()
                    if not content:
                        continue

                    role = lower_row.get("role", "user").strip() or "user"
                    ts = (lower_row.get("timestamp")
                          or lower_row.get("ts")
                          or lower_row.get("created_at")
                          or lower_row.get("date")
                          or "")

                    yield ConversationEntry(
                        role=role,
                        content=content,
                        timestamp=ts.strip() if ts else "",
                        provider="csv",
                        session_id=lower_row.get("session_id", session_id),
                    )
        except (OSError, csv.Error) as e:
            log(f"Could not read {filepath}: {e}", "WARN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_content(content: Any) -> str:
    """Extract plain text from various content formats.

    Handles strings, lists of content blocks (Claude-style), etc.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block and btype != "tool_use":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()

    return str(content).strip() if content else ""


# Parser registry -- ordered by specificity (most specific first)
PARSERS = [
    ChatGPTParser,
    ClaudeTranscriptParser,
    GeminiParser,
    GenericJSONLParser,
    GenericCSVParser,
]

PROVIDER_MAP: dict[str, type] = {
    "chatgpt": ChatGPTParser,
    "claude": ClaudeTranscriptParser,
    "gemini": GeminiParser,
    "jsonl": GenericJSONLParser,
    "csv": GenericCSVParser,
}


# ---------------------------------------------------------------------------
# MemoryImporter
# ---------------------------------------------------------------------------

class MemoryImporter:
    """Import conversation history from any LLM provider."""

    def __init__(self) -> None:
        self._state = load_state()
        self._seen_hashes = load_seen_hashes(self._state)
        self._new_count = 0
        self._dup_count = 0

    def detect_format(self, filepath: Path) -> str | None:
        """Detect the format of a file. Returns provider name or None."""
        filepath = Path(filepath)
        if not filepath.exists():
            log(f"File not found: {filepath}", "ERROR")
            return None

        for parser_cls in PARSERS:
            try:
                if parser_cls.can_parse(filepath):
                    return parser_cls.provider
            except Exception:
                continue
        return None

    def import_conversations(
        self,
        filepath: Path,
        provider: str | None = None,
    ) -> int:
        """Import conversations from a file.

        Args:
            filepath: Path to the file to import.
            provider: Force a specific parser. Auto-detects if None.

        Returns:
            Number of new entries imported (excludes duplicates).
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            log(f"File not found: {filepath}", "ERROR")
            return 0

        fkey = str(filepath)

        # Select parser
        parser_cls = None
        if provider and provider in PROVIDER_MAP:
            parser_cls = PROVIDER_MAP[provider]
        else:
            for cls in PARSERS:
                try:
                    if cls.can_parse(filepath):
                        parser_cls = cls
                        break
                except Exception:
                    continue

        if parser_cls is None:
            log(f"Could not detect format for: {filepath}", "WARN")
            return 0

        log(f"Parsing {filepath.name} as {parser_cls.provider}")

        # Parse and write
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        imported = 0
        duplicates = 0

        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
                for entry in parser_cls.parse(filepath):
                    h = entry.content_hash
                    if h in self._seen_hashes:
                        duplicates += 1
                        continue

                    self._seen_hashes.add(h)
                    out.write(json.dumps(entry.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    imported += 1
        except OSError as e:
            log(f"Write error: {e}", "ERROR")
            return 0

        self._new_count += imported
        self._dup_count += duplicates

        # Update state
        self._state["imported_files"][fkey] = {
            "provider": parser_cls.provider,
            "imported": imported,
            "duplicates": duplicates,
            "last_import": datetime.now().isoformat(timespec="seconds"),
        }
        # Keep seen_hashes as list in state (set is not JSON serializable)
        self._state["seen_hashes"] = list(self._seen_hashes)
        self._state["total_imported"] = self._state.get("total_imported", 0) + imported
        self._state["total_duplicates_skipped"] = (
            self._state.get("total_duplicates_skipped", 0) + duplicates
        )
        save_state(self._state)

        log(f"  Imported: {imported} new, {duplicates} duplicates skipped")
        return imported

    def import_directory(
        self,
        dirpath: Path,
        provider: str | None = None,
        recursive: bool = True,
    ) -> int:
        """Import all supported files from a directory.

        Returns total new entries imported.
        """
        dirpath = Path(dirpath).resolve()
        if not dirpath.exists() or not dirpath.is_dir():
            log(f"Directory not found: {dirpath}", "ERROR")
            return 0

        log(f"Scanning directory: {dirpath}")

        extensions = {".json", ".jsonl", ".csv", ".zip"}
        files: list[Path] = []

        if recursive:
            for ext in extensions:
                files.extend(dirpath.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(dirpath.glob(f"*{ext}"))

        files = sorted(set(files))
        log(f"  Found {len(files)} candidate files")

        total = 0
        for fp in files:
            try:
                count = self.import_conversations(fp, provider=provider)
                total += count
            except Exception as e:
                log(f"  Error processing {fp.name}: {e}", "WARN")

        log(f"Directory import complete: {total} new entries from {len(files)} files")
        return total

    def build_cognitive_model(self) -> dict[str, Any]:
        """Build a cognitive model from all imported conversations.

        Analyzes imported conversations to extract provider distribution,
        session patterns, role distribution, and temporal coverage.
        """
        if not OUTPUT_FILE.exists():
            log("No conversation data. Import some files first.", "ERROR")
            return {}

        log("Building cognitive model from imported conversations...")

        providers: dict[str, int] = {}
        roles: dict[str, int] = {}
        sessions: dict[str, int] = {}
        total = 0
        total_chars = 0
        earliest = ""
        latest = ""

        with open(OUTPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total += 1

                prov = entry.get("provider", "unknown")
                providers[prov] = providers.get(prov, 0) + 1

                role = entry.get("role", "unknown")
                roles[role] = roles.get(role, 0) + 1

                sid = entry.get("session_id", "")
                if sid:
                    sessions[sid] = sessions.get(sid, 0) + 1

                total_chars += entry.get("content_length", len(entry.get("content", "")))

                ts = entry.get("ts", "")
                if ts:
                    if not earliest or ts < earliest:
                        earliest = ts
                    if not latest or ts > latest:
                        latest = ts

        # Session size distribution
        session_sizes = sorted(sessions.values(), reverse=True) if sessions else []
        avg_session = round(sum(session_sizes) / len(session_sizes)) if session_sizes else 0

        model: dict[str, Any] = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "total_entries": total,
            "total_characters": total_chars,
            "unique_sessions": len(sessions),
            "avg_messages_per_session": avg_session,
            "date_range": {
                "earliest": earliest or "N/A",
                "latest": latest or "N/A",
            },
            "provider_distribution": dict(sorted(
                providers.items(), key=lambda x: x[1], reverse=True
            )),
            "role_distribution": dict(sorted(
                roles.items(), key=lambda x: x[1], reverse=True
            )),
            "top_sessions": dict(sorted(
                sessions.items(), key=lambda x: x[1], reverse=True
            )[:20]),
        }

        COGNITIVE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        COGNITIVE_MODEL_FILE.write_text(
            json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"Cognitive model written to {COGNITIVE_MODEL_FILE}")
        log(f"  {total} entries, {len(sessions)} sessions, {len(providers)} providers")

        return model

    def stats(self) -> dict[str, Any]:
        """Return import statistics."""
        state = load_state()

        result: dict[str, Any] = {
            "total_imported": state.get("total_imported", 0),
            "total_duplicates_skipped": state.get("total_duplicates_skipped", 0),
            "unique_hashes": len(state.get("seen_hashes", [])),
            "files_processed": len(state.get("imported_files", {})),
            "last_run": state.get("last_run"),
            "files": {},
        }

        for fpath, finfo in state.get("imported_files", {}).items():
            result["files"][Path(fpath).name] = finfo

        return result


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_import(filepath: str, provider: str | None = None) -> None:
    """Import a single file."""
    importer = MemoryImporter()
    count = importer.import_conversations(Path(filepath), provider=provider)
    print(f"\n  Imported {count} new entries")
    print(f"  Output: {OUTPUT_FILE}")


def cmd_import_dir(dirpath: str, provider: str | None = None) -> None:
    """Import all files from a directory."""
    importer = MemoryImporter()
    count = importer.import_directory(Path(dirpath), provider=provider)
    print(f"\n  Total imported: {count} new entries")
    print(f"  Output: {OUTPUT_FILE}")


def cmd_detect(filepath: str) -> None:
    """Detect the format of a file."""
    importer = MemoryImporter()
    fmt = importer.detect_format(Path(filepath))
    if fmt:
        print(f"  Detected format: {fmt}")
    else:
        print(f"  Could not detect format for: {filepath}")


def cmd_build() -> None:
    """Build cognitive model from imported conversations."""
    importer = MemoryImporter()
    model = importer.build_cognitive_model()
    if model:
        print(f"\n  Cognitive Model Summary")
        print(f"  {'='*40}")
        print(f"  Total entries:    {model.get('total_entries', 0):,}")
        print(f"  Total characters: {model.get('total_characters', 0):,}")
        print(f"  Sessions:         {model.get('unique_sessions', 0):,}")
        print(f"  Avg msgs/session: {model.get('avg_messages_per_session', 0)}")
        dr = model.get("date_range", {})
        print(f"  Date range:       {dr.get('earliest', 'N/A')} to {dr.get('latest', 'N/A')}")
        print(f"\n  Provider Distribution:")
        for prov, count in model.get("provider_distribution", {}).items():
            print(f"    {prov}: {count:,}")
        print(f"\n  Output: {COGNITIVE_MODEL_FILE}")


def cmd_stats() -> None:
    """Show import statistics."""
    importer = MemoryImporter()
    s = importer.stats()

    print(f"  Memory Import Statistics")
    print(f"  {'='*40}")
    print(f"  Total imported:       {s.get('total_imported', 0):,}")
    print(f"  Duplicates skipped:   {s.get('total_duplicates_skipped', 0):,}")
    print(f"  Unique content hashes: {s.get('unique_hashes', 0):,}")
    print(f"  Files processed:      {s.get('files_processed', 0)}")
    print(f"  Last run:             {s.get('last_run', 'never')}")

    files = s.get("files", {})
    if files:
        print(f"\n  Imported Files:")
        for fname, info in files.items():
            prov = info.get("provider", "?")
            count = info.get("imported", 0)
            dups = info.get("duplicates", 0)
            last = info.get("last_import", "?")
            print(f"    {fname} ({prov}): {count} imported, {dups} dups [{last}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory Import: import conversation history from any LLM provider"
    )
    parser.add_argument(
        "--import", dest="import_file", type=str, metavar="FILE",
        help="Import conversations from a file"
    )
    parser.add_argument(
        "--import-dir", type=str, metavar="DIR",
        help="Import all supported files from a directory"
    )
    parser.add_argument(
        "--provider", type=str, metavar="NAME",
        choices=list(PROVIDER_MAP.keys()),
        help="Force a specific parser (chatgpt, claude, gemini, jsonl, csv)"
    )
    parser.add_argument(
        "--detect", type=str, metavar="FILE",
        help="Detect format of a file without importing"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build cognitive model from imported conversations"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show import statistics"
    )

    args = parser.parse_args()

    if not any([args.import_file, args.import_dir, args.detect, args.build, args.stats]):
        parser.print_help()
        sys.exit(1)

    if args.detect:
        cmd_detect(args.detect)
    elif args.import_file:
        cmd_import(args.import_file, provider=args.provider)
    elif args.import_dir:
        cmd_import_dir(args.import_dir, provider=args.provider)

    if args.build:
        cmd_build()
    if args.stats:
        cmd_stats()


if __name__ == "__main__":
    main()
