"""Small, atomic cache for independent cover-analysis stages.

The cache deliberately stores JSON evidence rather than opaque Python objects.
That makes interrupted runs inspectable and lets a later run reuse only stages
whose source hash, model revision and options still match.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class StageCache:
    """Persistent cache keyed by the complete source audio contents."""

    schema = 1

    def __init__(self, root: Path | None, audio: Path):
        self.source_sha256 = sha256_file(audio)
        self.root = Path(root) / self.source_sha256 if root else None
        self.records: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, stage: str) -> Path:
        if self.root is None:
            raise RuntimeError("stage cache is disabled")
        return self.root / f"{stage}.json"

    def load(self, stage: str, signature: dict[str, Any]) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self._path(stage)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return None
        if (
            envelope.get("schema") != self.schema
            or envelope.get("source_sha256") != self.source_sha256
            or envelope.get("signature") != signature
            or not isinstance(envelope.get("payload"), dict)
        ):
            return None
        self.records[stage] = {
            "status": "reused",
            "signature": signature,
            "saved_at": envelope.get("saved_at"),
            "elapsed_seconds": envelope.get("elapsed_seconds"),
        }
        return envelope["payload"]

    def save(self, stage: str, signature: dict[str, Any], payload: dict[str, Any], elapsed_seconds: float | None = None) -> None:
        if self.root is None:
            return
        envelope = {
            "schema": self.schema,
            "source_sha256": self.source_sha256,
            "stage": stage,
            "signature": signature,
            "saved_at": time.time(),
            "elapsed_seconds": elapsed_seconds,
            "payload": payload,
        }
        _atomic_json(self._path(stage), envelope)
        self.records[stage] = {
            "status": "complete",
            "signature": signature,
            "saved_at": envelope["saved_at"],
            "elapsed_seconds": elapsed_seconds,
        }

    def fail(self, stage: str, signature: dict[str, Any], error: str) -> None:
        if self.root is None:
            return
        envelope = {
            "schema": self.schema,
            "source_sha256": self.source_sha256,
            "stage": stage,
            "signature": signature,
            "saved_at": time.time(),
            "status": "failed",
            "error": error,
        }
        _atomic_json(self.root / f"{stage}.failure.json", envelope)
        self.records[stage] = {"status": "failed", "signature": signature, "error": error}
