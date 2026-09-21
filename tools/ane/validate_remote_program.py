"""Synthetic Core ML regression probe; no YuE weights, songs, phone, or UI required.

PYTHONPATH=src python tools/ane/validate_remote_program.py --rows 1024 --prefix 2048
Builds the old/new remote graphs, checks actual Neural Engine placement on this Mac,
and compares the new graph's predictions against the old graph running on CPU.
This does not establish placement or speed on an iPhone.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from yue2.ane.mil import build_program, weight_input_shapes
from yue2.remote.client import QBLK, KCHUNK, LAYERS, program_name


def inputs(rows, prefix, seed=42):
    """Full-sized, deterministic inputs including masked prefix and song padding."""
    rng = np.random.default_rng(seed)
    feed = {"x": rng.normal(0, 0.5, (1, 1, rows, 2048)).astype(np.float16)}
    angles = np.arange(prefix, prefix + rows)[:, None] / 10000 ** (np.arange(64)[None, :] / 64)
    feed.update(cos=np.cos(angles)[None, None].astype(np.float16), sin=np.sin(angles)[None, None].astype(np.float16))
    bias = np.zeros((1, 1, 1, rows + prefix), np.float16)
    bias[..., prefix - 13:prefix] = -1e4
    bias[..., -29:] = -1e4
    feed["bias"] = bias
    for i in range(LAYERS):
        for kind in ("pk", "pv"):
            feed[f"{kind}{i}"] = rng.normal(0, 0.1 if kind == "pk" else 0.5, (1, 8, prefix, 128)).astype(np.float16)
        for name, shape in weight_input_shapes().items():
            if name.endswith("norm"):
                value = np.ones(shape, np.float16)
                if name == "q_norm": value *= np.float16(128 ** -0.5)
            else:
                value = (rng.standard_normal(shape, dtype=np.float32) * (shape[-1] ** -0.5)).astype(np.float16)
            feed[f"w{i}_{name}"] = value
    return feed


def reference_prediction(feed):
    """Independent float32 implementation, with no MIL/Core ML graph optimizations."""
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(4)
    tensor = lambda name: torch.from_numpy(feed[name].astype(np.float32))
    x = tensor("x")[0, 0]
    cos, sin = tensor("cos")[0, 0][:, None], tensor("sin")[0, 0][:, None]
    bias = tensor("bias")[0, 0]
    def norm(value, weight):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 6.1e-5) * weight
    def rotary(value):
        a, b = value.chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    for layer in range(LAYERS):
        w = {name: tensor(f"w{layer}_{name}")[0, 0] for name in weight_input_shapes()}
        hn = norm(x, w["in_norm"])
        q = rotary(norm((hn @ w["q"].T).reshape(-1, 16, 128), w["q_norm"]))
        k = rotary(norm((hn @ w["k"].T).reshape(-1, 8, 128), w["k_norm"]))
        v = (hn @ w["v"].T).reshape(-1, 8, 128)
        keys = torch.cat((tensor(f"pk{layer}")[0], k.permute(1, 0, 2)), dim=1).repeat_interleave(2, dim=0)
        values = torch.cat((tensor(f"pv{layer}")[0], v.permute(1, 0, 2)), dim=1).repeat_interleave(2, dim=0)
        parts = []
        for start in range(0, len(x), 128):
            scores = q[start:start + 128].permute(1, 0, 2) @ keys.transpose(-1, -2) + bias
            parts.append((scores.softmax(dim=-1) @ values).permute(1, 0, 2).reshape(-1, 2048))
        x = x + torch.cat(parts) @ w["o"].T
        hn = norm(x, w["mlp_norm"])
        x = x + (functional.silu(hn @ w["gate"].T) * (hn @ w["up"].T)) @ w["down"].T
    return x.numpy()[None, None]


def validate(rows, prefix, output, predict=True):
    import coremltools as ct
    output.mkdir(parents=True, exist_ok=True)
    new = output / (program_name(rows, prefix) + ".mlpackage")
    old = output / f"reference_online_{rows}_{prefix}.mlpackage"
    for path, mode, query in ((new, "softmax", QBLK), (old, "online", 512)):
        if path == old and not predict: continue
        build_program([None] * LAYERS, None, S=rows, P=prefix, weights_as_inputs=True,
                      target="iOS18", package_path=path, qblk=query, kchunk=KCHUNK, attention_mode=mode)
    compiled = ct.models.utils.compile_model(str(new))
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage: counts[type(usage.preferred_compute_device).__name__] += 1
    report = {"rows": rows, "prefix": prefix, "program": new.name, "placement": dict(counts)}
    print(json.dumps(report), flush=True)
    if not counts or set(counts) != {"MLNeuralEngineComputeDevice"}:
        raise AssertionError(f"Not entirely on the Neural Engine: {counts}")
    if predict:
        feed = inputs(rows, prefix)
        fp32 = reference_prediction(feed)
        reference = ct.models.MLModel(str(old), compute_units=ct.ComputeUnit.CPU_ONLY)
        expected = np.asarray(reference.predict(feed)["h"], dtype=np.float32)
        del reference
        cpu_model = ct.models.MLModel(str(new), compute_units=ct.ComputeUnit.CPU_ONLY)
        cpu_actual = np.asarray(cpu_model.predict(feed)["h"], dtype=np.float32)
        del cpu_model
        formulation_rms = float(np.sqrt(np.mean((cpu_actual - expected) ** 2)) / np.sqrt(np.mean(expected ** 2)))
        model = ct.models.MLModel(str(new), compute_units=ct.ComputeUnit.CPU_AND_NE)
        t0 = time.perf_counter()
        actual = np.asarray(model.predict(feed)["h"], dtype=np.float32)
        seconds = time.perf_counter() - t0
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        rms = float(np.sqrt(np.mean((actual - expected) ** 2)) / np.sqrt(np.mean(expected ** 2)))
        corr = float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
        relative_error = lambda result: float(np.sqrt(np.mean((result - fp32) ** 2)) / np.sqrt(np.mean(fp32 ** 2)))
        report.update(reference_cpu_error=relative_error(expected), new_cpu_error=relative_error(cpu_actual), new_ane_error=relative_error(actual))
        report.update(finite=finite, formulation_relative_rms=formulation_rms, relative_rms=rms, correlation=corr, first_prediction_seconds=seconds)
        print(json.dumps(report), flush=True)
        # FP16 reductions/matmuls differ between CPU and ANE. Judge accuracy against
        # float32, and require no material regression relative to the original graph.
        baseline_error = report["reference_cpu_error"]
        if (not finite or corr < 0.999 or report["new_ane_error"] > 0.015
                or report["new_ane_error"] > baseline_error * 1.05 + 1e-4
                or report["new_cpu_error"] > baseline_error * 1.05 + 1e-4):
            raise AssertionError(f"Numerical regression: {report}")
    (output / f"validation_{rows}_{prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("build/remote-ane-validation"))
    parser.add_argument("--placement-only", action="store_true")
    args = parser.parse_args()
    validate(args.rows, args.prefix, args.output, not args.placement_only)
