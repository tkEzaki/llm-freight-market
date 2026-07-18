"""Disk-based cache so identical prompts don't hit the API twice.

Keyed on (backend, model, temperature, prompt). One JSON file per cell run
keeps it human-inspectable; lookups are O(1) via an in-memory dict.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from .base import LLMResponse


CACHE_LOCK = threading.Lock()


class ResponseCache:
    """Append-only JSONL cache. Process-safe via a coarse lock."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._mem: Dict[str, dict] = {}
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["value"]
                except Exception:
                    continue

    @staticmethod
    def make_key(backend: str, model: str, temperature: float, prompt: str) -> str:
        h = hashlib.sha256()
        h.update(backend.encode("utf-8"))
        h.update(b"\x00")
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(f"{temperature:.4f}".encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------
    def get(self, backend: str, model: str, temperature: float,
            prompt: str) -> Optional[LLMResponse]:
        key = self.make_key(backend, model, temperature, prompt)
        v = self._mem.get(key)
        if v is None:
            return None
        return LLMResponse(**{**v, "cached": True})

    def put(self, response: LLMResponse, prompt: str,
            temperature: float) -> None:
        key = self.make_key(response.backend, response.model,
                            temperature, prompt)
        rec_value = asdict(response)
        rec_value["cached"] = False
        with CACHE_LOCK:
            self._mem[key] = rec_value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": rec_value},
                                       ensure_ascii=False) + "\n")
