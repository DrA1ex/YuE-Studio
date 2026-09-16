"""Batched autoregressive token decoding in MLX with 8-bit weights.

The token loop is bound by the bytes of weights it streams per step (4 GB in bf16 at ~109 GB/s
on an M4: 37 ms), so the lever is fewer bytes: weights stored as 8-bit integers with a 16-bit
scale and bias per group of 64, expanded on the fly inside MLX's quantized matmul. Everything
else mirrors ``batched.generate_tokens_batched``: left-padded rows, per-row seeded sampling on
the CPU, rows leaving the batch when they end, per-row token budgets.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import mlx.core as mx

from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT
from .batched import sample_batched

GROUP, BITS = 64, 8


def _to_mx(t):
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return mx.view(mx.array(t.contiguous().view(torch.int16).cpu().numpy()), mx.bfloat16)
    return mx.array(t.float().cpu().numpy())


def _quantize(w):
    """(wq, scales, biases) for a [out, in] weight; rows stay independent so row slices remain valid."""
    return tuple(mx.quantize(w, group_size=GROUP, bits=BITS))


def _qmatmul(x, q):
    wq, scales, biases = q
    return mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=GROUP, bits=BITS)


class ARWeights:
    """The AR path of a YuE2ForCausalLM as quantized MLX arrays (converted once, ~2.2 GB)."""

    def __init__(self, model, bits=BITS):
        cfg = model.config
        self.eps, self.theta = cfg.rms_norm_eps, cfg.rope_theta
        self.H, self.KV, self.HD, self.D, self.V = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                                                   cfg.hidden_size, cfg.vocab_size)
        self.dtype = mx.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else mx.float32
        self.quantized = bits is not None
        lin = _quantize if self.quantized else (lambda w: w)
        self.embed = lin(_to_mx(model.model.embed_tokens.weight))
        self.layers = []
        for layer in model.model.layers:
            a, m = layer.self_attn, layer.mlp
            self.layers.append({
                "in_norm": _to_mx(layer.input_layernorm.weight),
                "qkv": lin(mx.concatenate([_to_mx(a.q_proj.weight), _to_mx(a.k_proj.weight), _to_mx(a.v_proj.weight)], axis=0)),
                "q_norm": _to_mx(a.q_norm.weight), "k_norm": _to_mx(a.k_norm.weight),
                "o": lin(_to_mx(a.o_proj.weight)),
                "mlp_norm": _to_mx(layer.post_attention_layernorm.weight),
                "gate_up": lin(mx.concatenate([_to_mx(m.gate_proj.weight), _to_mx(m.up_proj.weight)], axis=0)),
                "down": lin(_to_mx(m.down_proj.weight)),
            })
            mx.eval(*[v for v in self.layers[-1].values() if not isinstance(v, tuple)],
                    *[t for v in self.layers[-1].values() if isinstance(v, tuple) for t in v])
        self.final_norm = _to_mx(model.model.norm.weight)
        self.head = lin(_to_mx(model.lm_head.weight))
        # Rotary tables in fp32, cast to the model dtype at use like the PyTorch model does.
        inv = 1.0 / (self.theta ** (np.arange(0, self.HD, 2, dtype=np.float32) / self.HD))
        angles = np.arange(CONTEXT, dtype=np.float32)[:, None] * inv[None]
        self.cos = mx.array(np.cos(angles)).astype(self.dtype)
        self.sin = mx.array(np.sin(angles)).astype(self.dtype)
        mx.eval(self.final_norm, self.cos, self.sin, *(self.embed if isinstance(self.embed, tuple) else (self.embed,)),
                *(self.head if isinstance(self.head, tuple) else (self.head,)))

    def bytes(self):
        def size(v):
            return sum(t.nbytes for t in v) if isinstance(v, tuple) else v.nbytes
        return size(self.embed) + size(self.head) + sum(size(v) for l in self.layers for v in l.values())

    def linear(self, x, w):
        return _qmatmul(x, w) if self.quantized else mx.matmul(x, w.T)

    def embedding(self, ids):
        if not self.quantized:
            return self.embed[ids]
        wq, scales, biases = self.embed
        flat = ids.reshape(-1)
        rows = mx.dequantize(wq[flat], scales[flat], biases[flat], group_size=GROUP, bits=BITS)
        return rows.reshape(*ids.shape, self.D).astype(self.dtype)

    def head_logits(self, x, lo, hi):
        """Logits for vocabulary rows [lo, hi) only (the sampler masks everything else anyway)."""
        if self.quantized:
            wq, scales, biases = self.head
            return _qmatmul(x, (wq[lo:hi], scales[lo:hi], biases[lo:hi]))
        return mx.matmul(x, self.head[lo:hi].T)


class KVCache:
    """Per-layer K/V for a batch, preallocated to the run's maximum length."""

    def __init__(self, layers, n, kv, hd, capacity, dtype):
        self.keys = [mx.zeros((n, kv, capacity, hd), dtype=dtype) for _ in range(layers)]
        self.values = [mx.zeros((n, kv, capacity, hd), dtype=dtype) for _ in range(layers)]
        self.offset = 0

    def update(self, i, k, v):
        T = k.shape[2]
        self.keys[i][..., self.offset:self.offset + T, :] = k
        self.values[i][..., self.offset:self.offset + T, :] = v
        end = self.offset + T
        return self.keys[i][..., :end, :], self.values[i][..., :end, :]


def _rotary(x, cos, sin):
    """x [n, heads, T, HD]; cos/sin [n, 1, T, HD/2]: the model's rotate-half convention."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def forward(w, x, positions, mask, cache, advance):
    """One pass over [n, T] embeddings; returns the last position's hidden state [n, D]."""
    n, T, _ = x.shape
    cos, sin = w.cos[positions][:, None], w.sin[positions][:, None]          # [n, 1, T, HD/2]
    q_end, k_end = w.H * w.HD, (w.H + w.KV) * w.HD
    scale = w.HD ** -0.5
    for i, layer in enumerate(w.layers):
        h = mx.fast.rms_norm(x, layer["in_norm"], w.eps)
        qkv = w.linear(h, layer["qkv"])
        q = mx.fast.rms_norm(qkv[..., :q_end].reshape(n, T, w.H, w.HD), layer["q_norm"], w.eps)
        k = mx.fast.rms_norm(qkv[..., q_end:k_end].reshape(n, T, w.KV, w.HD), layer["k_norm"], w.eps)
        v = qkv[..., k_end:].reshape(n, T, w.KV, w.HD)
        q = _rotary(mx.transpose(q, (0, 2, 1, 3)), cos, sin)
        k = _rotary(mx.transpose(k, (0, 2, 1, 3)), cos, sin)
        v = mx.transpose(v, (0, 2, 1, 3))
        k, v = cache.update(i, k, v)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        x = x + w.linear(mx.transpose(a, (0, 2, 1, 3)).reshape(n, T, w.H * w.HD), layer["o"])
        m = mx.fast.rms_norm(x, layer["mlp_norm"], w.eps)
        gu = w.linear(m, layer["gate_up"])
        g, u = mx.split(gu, 2, axis=-1)
        x = x + w.linear(g * mx.sigmoid(g) * u, layer["down"])
    if advance:
        cache.offset += T
    return mx.fast.rms_norm(x[:, -1], w.final_norm, w.eps)


def weights_for(model, bits=BITS):
    cached = getattr(model, "_yue2_ar_mlx", None)
    if cached is None or cached.quantized != (bits is not None):
        cached = ARWeights(model, bits=bits)
        model._yue2_ar_mlx = cached
    return cached


def generate_tokens_batched(model, prefixes, sampling, seeds, phase, *, legacy_off=False, cancelled=None,
                            on_token=None, pad_id=0, lock=None, on_row_done=None, limits=None, bits=BITS):
    """Drop-in for ``batched.generate_tokens_batched`` running in MLX (``lock`` is accepted and unused)."""
    n = len(prefixes)
    if n < 1 or len(seeds) != n:
        raise ValueError("Provide at least one prefix and one seed per prefix")
    w = weights_for(model, bits)
    lengths = [len(p) for p in prefixes]
    longest = max(lengths)
    if longest + sampling.max_tokens > CONTEXT:
        raise ValueError("Longest prefix + generation budget exceeds 24576")
    if limits is not None and (len(limits) != n or any(l < 1 or l > sampling.max_tokens for l in limits)):
        raise ValueError("limits must give each row a budget between 1 and sampling.max_tokens")
    end = ABC_END if phase == "abc" else MUSIC_END
    lo, hi = (0, ABC_END + 1) if phase == "abc" else (MUSIC_END, CODEC_OFFSET + CODEC_SIZE)

    ids = np.full((n, longest), pad_id, dtype=np.int32)
    real = np.zeros((n, longest), dtype=bool)
    for i, prefix in enumerate(prefixes):
        ids[i, longest - len(prefix):] = prefix
        real[i, longest - len(prefix):] = True
    positions = np.maximum(np.cumsum(real, axis=1) - 1, 0).astype(np.int32)
    allowed = np.tril(np.ones((longest, longest), dtype=bool))[None] & real[:, None, :]
    allowed[:, np.arange(longest), np.arange(longest)] = True         # padded queries see themselves: finite K/V
    capacity = longest + sampling.max_tokens
    cache = KVCache(len(w.layers), n, w.KV, w.HD, capacity, w.dtype)

    def logits_of(hidden):
        part = np.array(w.head_logits(hidden, lo, hi).astype(mx.float32))
        full = torch.full((n, w.V), float("-inf"), dtype=torch.float32)
        full[:, lo:hi] = torch.from_numpy(part)
        return full

    generators = [torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
    start = time.perf_counter()
    hidden = forward(w, w.embedding(mx.array(ids)), mx.array(positions), mx.array(allowed)[:, None], cache, advance=True)
    mx.eval(hidden)
    logits = logits_of(hidden)
    prefill_seconds = time.perf_counter() - start

    histories = [[] for _ in range(n)]
    done, eos, first = [False] * n, [False] * n, [None] * n
    key_mask = np.zeros((n, capacity), dtype=bool)
    key_mask[:, :longest] = real
    lengths_np = np.array(lengths, dtype=np.int32)
    step_seconds, steps = [], 0
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during batched {phase}")
        tokens = sample_batched(logits, sampling, histories, step, phase, generators, legacy_off)
        for i, token in enumerate(tokens):
            if done[i]:
                tokens[i] = end
                continue
            if first[i] is None:
                first[i] = time.perf_counter() - start
            if on_token is not None:
                on_token(i, phase, token)
            if token == end:
                eos[i] = done[i] = True
                if on_row_done is not None:
                    elapsed = time.perf_counter() - start
                    on_row_done(i, list(histories[i]), _timing(elapsed, prefill_seconds, first[i], len(histories[i]) + 1,
                                                                len(histories[i]), lengths[i], n, i))
            else:
                histories[i].append(token)
                if limits is not None and len(histories[i]) >= limits[i]:
                    done[i] = True
                    if on_row_done is not None:
                        elapsed = time.perf_counter() - start
                        t = _timing(elapsed, prefill_seconds, first[i], len(histories[i]), len(histories[i]), lengths[i], n, i)
                        t["truncated"] = True
                        on_row_done(i, list(histories[i]), t)
        steps = step + 1
        if all(done) or step + 1 >= sampling.max_tokens:
            break
        tick = time.perf_counter()
        key_mask[:, longest + step] = True
        L = longest + step + 1
        hidden = forward(w, w.embedding(mx.array(np.array(tokens, dtype=np.int32)[:, None])),
                         mx.array((lengths_np + step)[:, None]), mx.array(key_mask[:, None, None, :L]), cache, advance=True)
        mx.eval(hidden)
        logits = logits_of(hidden)
        step_seconds.append(time.perf_counter() - tick)
    seconds = time.perf_counter() - start
    rows = []
    for i in range(n):
        count = len(histories[i]) + int(eos[i])
        rows.append((histories[i], _timing(seconds, prefill_seconds, first[i], count, len(histories[i]), lengths[i], n, i), not eos[i]))
    total = sum(len(h) + int(e) for h, e in zip(histories, eos))
    batch = {"batch_size": n, "steps": steps, "seconds": seconds, "prefill_seconds": prefill_seconds,
             "mean_step_seconds": sum(step_seconds) / len(step_seconds) if step_seconds else None,
             "total_output_tokens": total, "aggregate_tps": total / seconds, "engine": "mlx-int8" if w.quantized else "mlx"}
    return rows, batch


def _timing(seconds, prefill, ttft, output, content, prefix, n, i):
    return {"seconds": seconds, "prefill_seconds": prefill, "ttft_seconds": ttft, "output_tokens": output,
            "content_tokens": content, "output_tps": output / seconds if seconds else 0.0, "prefix_tokens": prefix,
            "cfg_branches": 1, "execution": "eager_batched_mlx", "batch_size": n, "batch_index": i}
