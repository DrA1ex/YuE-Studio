#!/usr/bin/env python3
"""Download and verify the pinned YuE2 model snapshots with JSON progress output."""
import argparse
import importlib
import json
import os
import sys
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi
from yue2.models import DEFAULT_MODELS
from yue2.storage import model_identity, resolve_model, verify_model_snapshot

hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")

parser = argparse.ArgumentParser()
parser.add_argument("repos", nargs="*", help="Optional unpinned repositories for development use")
parser.add_argument("--verify-only", action="store_true", help="Fail instead of downloading if a snapshot is missing or corrupt")
args = parser.parse_args()
SPECS = [(repo, None) for repo in args.repos] if args.repos else list(DEFAULT_MODELS)


def verify_local(repo, revision):
    try:
        path = resolve_model(repo, revision=revision, local_files_only=True)
        verify_model_snapshot(path)
        model_identity(path, verify=True)
        return path
    except (FileNotFoundError, ValueError, OSError):
        return None


ready = [(repo, revision, verify_local(repo, revision)) for repo, revision in SPECS]
if all(path is not None for _, _, path in ready):
    print(json.dumps({"done": True, "cached": True, "models": len(ready)}), flush=True)
    raise SystemExit(0)
if args.verify_only:
    missing = [repo for repo, _, path in ready if path is None]
    raise SystemExit("Missing or corrupt model snapshots: " + ", ".join(missing))

api = HfApi()
total = sum((f.size or 0) for repo, revision in SPECS
            for f in api.model_info(repo, revision=revision, files_metadata=True).siblings)
state = {"bars": {}, "last": 0.0, "samples": []}
lock = threading.Lock()


def report(force=False):
    now = time.time()
    if not force and now - state["last"] < 0.5:
        return
    state["last"] = now
    done = sum(state["bars"].values())
    state["samples"].append((now, done))
    state["samples"] = [sample for sample in state["samples"] if now - sample[0] <= 15]
    rate = 0.0
    if len(state["samples"]) > 1 and now - state["samples"][0][0] >= 2:
        rate = (done - state["samples"][0][1]) / (now - state["samples"][0][0]) / 1e6
    print(json.dumps({"bytes": done, "total": total, "rate_mbps": round(rate, 1)}), flush=True)


class Progress(hf_tqdm.tqdm):
    def __init__(self, *progress_args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*progress_args, **kwargs)
        self._bytes = kwargs.get("unit") == "B"
        self._count = kwargs.get("initial", 0) or 0
        if self._bytes:
            with lock:
                state["bars"][id(self)] = self._count

    def update(self, n=1):
        if self._bytes:
            self._count += n
            with lock:
                state["bars"][id(self)] = self._count
                report()
        return super().update(n)


hf_tqdm.tqdm = Progress
print(json.dumps({"bytes": 0, "total": total, "rate_mbps": 0.0}), flush=True)
for repo, revision in SPECS:
    path = resolve_model(repo, revision=revision, max_workers=16)
    try:
        verify_model_snapshot(path)
        model_identity(path, verify=True)
    except (FileNotFoundError, ValueError, OSError):
        # A stale snapshot can exist after disk corruption. Force a fresh copy once before failing.
        path = resolve_model(repo, revision=revision, force_download=True, max_workers=16)
        verify_model_snapshot(path)
        model_identity(path, verify=True)
with lock:
    report(force=True)
print(json.dumps({"done": True, "cached": False, "total": total}), flush=True)
