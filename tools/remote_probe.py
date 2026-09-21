"""First real-song test of the iPhone engine, without the app.

    PYTHONPATH=src .venv/bin/python tools/remote_probe.py HOST PORT outputs/<run>/<song>

Loads the model (do not run while YuE Studio has a live song), sends the weights if the phone
lacks them, and solves the song's synthesis on the phone with 4 steps; then, for comparison,
the same 4 steps on the Mac's Neural Engine. Prints per-pass times and the latent correlation.
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from yue2 import YuE2Pipeline
from yue2.nar import synthesize
from yue2.pipeline import SymbolicPlan
from yue2.remote.client import RemoteClient

host, port, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
client = RemoteClient(host, port)
print(json.dumps({"phone": client.info}), flush=True)
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="mps", progress=False)
model = pipe._load_model(for_nar=True)
plan = SymbolicPlan.load(d); tokens = np.load(f"{d}/semantic.npy").tolist()
print(json.dumps({"frames": len(tokens), "prefix": len(plan.prefix)}), flush=True)
phases = []
def phase(text):
    if not phases or phases[-1] != text:
        phases.append(text); print("  " + text, flush=True)
t0 = time.perf_counter()
remote = synthesize(model, plan.prefix, tokens, plan.request.seed, engine="remote", remote=client, steps=steps,
                    on_phase=phase, on_progress=lambda a, b: print(f"  step {a}/{b}", flush=True))
print(json.dumps({"remote_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
t0 = time.perf_counter()
local = synthesize(model, plan.prefix, tokens, plan.request.seed, engine="ane", steps=steps,
                   on_progress=lambda a, b: print(f"  step {a}/{b}", flush=True))
a, b = remote.numpy().ravel(), local.numpy().ravel()
print(json.dumps({"ane_seconds": round(time.perf_counter() - t0, 1), "corr": round(float(np.corrcoef(a, b)[0, 1]), 6),
                  "rel_rms": round(float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean())), 5)}), flush=True)
client.close()
