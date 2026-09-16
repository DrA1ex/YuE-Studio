"""Validate and time the MLX token engine against the PyTorch one.

1. First-step logits: PyTorch vs MLX bf16 (should agree closely) vs MLX int8 (small drift).
2. Greedy decoding of N tokens from the same prefix: how long the three agree.
3. Steps per second at batch sizes 1, 2, 4 for PyTorch bf16 and MLX int8.
Usage: python tools/bench_ar_mlx.py [tokens=64]
"""
import os, sys, time, dataclasses, numpy as np, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Library/Application Support/YuE Studio/models"))
from yue2 import YuE2Pipeline
from yue2 import batched, ar_mlx
from yue2.protocol import token_prefixes, MUSIC_END, CODEC_OFFSET, CODEC_SIZE
import mlx.core as mx

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="mps", progress=False, lean=True)
model = pipe._load_model()
req = pipe._request(style="English, warm piano pop, expressive female voice, 88 BPM",
                    lyrics="[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\nLet the day come into view",
                    cot="off", seed=831001, id="s", cfg_scale=1.0)
prefix = token_prefixes(req, pipe.tokenizer)
greedy = dataclasses.replace(pipe.generation_config.semantic, max_tokens=N, min_tokens=0, temperature=0.0)

def run(engine, bits=None, n=1, sampling=greedy):
    prefixes, seeds = [prefix] * n, list(range(1, n + 1))
    if engine == "torch":
        rows, t = batched.generate_tokens_batched(model, prefixes, sampling, seeds, "semantic", legacy_off=True)
    else:
        rows, t = ar_mlx.generate_tokens_batched(model, prefixes, sampling, seeds, "semantic", legacy_off=True, bits=bits)
    return rows[0][0], t

# 1 + 2: greedy agreement
t_tok, tt = run("torch")
print(f"torch bf16   : {N} greedy tokens, {1000 * tt['mean_step_seconds']:.1f} ms/step")
m_tok, tm = run("mlx", bits=None)
print(f"mlx bf16     : {N} greedy tokens, {1000 * tm['mean_step_seconds']:.1f} ms/step")
q_tok, tq = run("mlx", bits=8)
print(f"mlx int8     : {N} greedy tokens, {1000 * tq['mean_step_seconds']:.1f} ms/step")
def agree(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y: break
        k += 1
    return k
print(f"agreement: torch vs mlx-bf16 first {agree(t_tok, m_tok)}/{N} tokens; mlx-bf16 vs int8 first {agree(m_tok, q_tok)}/{N}; torch vs int8 first {agree(t_tok, q_tok)}/{N}")

# first-step logits
def first_logits_torch():
    ids = torch.tensor([prefix], device="mps")
    with torch.inference_mode():
        out = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu().numpy()
    return out[MUSIC_END:CODEC_OFFSET + CODEC_SIZE]
def first_logits_mlx(bits):
    w = ar_mlx.weights_for(model, bits)
    ids = np.array([prefix], dtype=np.int32); T = len(prefix)
    cache = ar_mlx.KVCache(len(w.layers), 1, w.KV, w.HD, T, w.dtype)
    mask = np.tril(np.ones((T, T), dtype=bool))[None, None]
    h = ar_mlx.forward(w, w.embedding(mx.array(ids)), mx.array(np.arange(T, dtype=np.int32)[None]), mx.array(mask), cache, True)
    return np.array(w.head_logits(h, MUSIC_END, CODEC_OFFSET + CODEC_SIZE).astype(mx.float32))[0]
lt, lm, lq = first_logits_torch(), first_logits_mlx(None), first_logits_mlx(8)
for name, l in (("mlx bf16", lm), ("mlx int8", lq)):
    print(f"first-step logits {name} vs torch: corr {np.corrcoef(lt, l)[0,1]:.6f}, top1 {'same' if l.argmax() == lt.argmax() else 'DIFFERENT'}, "
          f"max|diff| {np.abs(lt - l).max():.3f} (logit range {lt.max() - lt.min():.1f})")

# 3: speed at batch sizes with real sampling
sampled = dataclasses.replace(pipe.generation_config.semantic, max_tokens=48, min_tokens=0)
for n in (1, 2, 4):
    _, tt = run("torch", n=n, sampling=sampled); _, tq = run("mlx", bits=8, n=n, sampling=sampled)
    print(f"batch {n}: torch {1000 * tt['mean_step_seconds']:.1f} ms/step, mlx int8 {1000 * tq['mean_step_seconds']:.1f} ms/step "
          f"(x{tt['mean_step_seconds'] / tq['mean_step_seconds']:.2f}); prefill torch {tt['prefill_seconds']:.2f}s mlx {tq['prefill_seconds']:.2f}s")
print(f"mlx int8 weights: {ar_mlx.weights_for(model, 8).bytes() / 1e9:.2f} GB")
