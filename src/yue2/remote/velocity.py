"""Solver for one chunk whose 28 decoder layers run on the iPhone (see ane/runtime.ANEVelocity)."""
from __future__ import annotations
import time
import numpy as np
import torch

from .client import RemoteClient, S_STEP, P_STEP, bucket
from ..ane.mil import NEG


class RemoteVelocity:
    """Same contract as ``ANEVelocity``: ``velocity(state, raw_t)`` and ``solve``; the prefix K/V
    move from the GPU to the phone in the constructor."""

    def __init__(self, engine, model, client: RemoteClient, on_prepare=None, on_phase=None):
        cfg = model.config
        self.model, self.engine, self.client = model, engine, client
        if engine.visible_length != engine.ar_length:
            raise ValueError("Remote engine does not support restricted prefix visibility")
        self.D, self.KV, self.HD, self.H = cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim, cfg.num_attention_heads
        self.S_real, self.P_real = engine.nar_length, engine.ar_length
        self.S, self.P = bucket(self.S_real, S_STEP), bucket(self.P_real, P_STEP)
        self.K = self.S + self.P
        phase = on_phase if on_phase is not None else (lambda text: None)
        f32 = lambda t: t.detach().float().cpu()
        self.vae2llm = (f32(model.vae2llm.weight), f32(model.vae2llm.bias))
        self.llm2vae = (f32(model.llm2vae.weight), f32(model.llm2vae.bias))
        self.final_norm, self.eps = f32(model.model.norm.weight), cfg.rms_norm_eps
        import copy
        self.time_embedder = copy.deepcopy(model.time_embedder).to("cpu").eval()
        self.pos_emb = f32(engine.pos_emb[0])
        program = client.ensure_program(self.S, self.P, cfg, on_progress=phase)
        phase("opening the session on the iPhone")
        client.open(self.S, self.P, self.S_real, self.P_real, program, on_progress=phase)
        t0 = time.perf_counter()
        for i, (k, v) in enumerate(engine.cache):
            pk = f32(k).numpy().transpose(1, 0, 2).astype(np.float16)     # [KV, P_real, HD]
            pv = f32(v).numpy().transpose(1, 0, 2).astype(np.float16)
            client.send_kv(i, pk, pv)
            phase(f"sending the prefix to the iPhone: layer {i + 1}/{len(engine.cache)}")
        engine.cache = []
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        positions = np.arange(self.P_real, self.P_real + self.S, dtype=np.float32)
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, self.HD, 2, dtype=np.float32) / self.HD))
        angles = positions[:, None] * inv[None]
        bias = np.zeros((self.K,), dtype=np.float16)
        bias[self.P_real:self.P] = NEG
        bias[self.P + self.S_real:] = NEG
        client.send_tables(np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16), bias)
        self.transfer_seconds = time.perf_counter() - t0
        self.layer_seconds = []

    @torch.inference_mode()
    def _time_embedding(self, raw_t):
        shifted = self.model._shift_t_value(raw_t, torch.device("cpu"), torch.bfloat16)
        return self.time_embedder(shifted.expand(1)).float()[0]

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        x_nar = torch.nn.functional.pad(state, (0, 0, 1, 1))
        x = x_nar @ self.vae2llm[0].T + self.vae2llm[1] + self._time_embedding(raw_t) + self.pos_emb
        x0 = np.zeros((self.S, self.D), dtype=np.float16)
        x0[:self.S_real] = x.numpy().astype(np.float16)
        t0 = time.perf_counter()
        out, phone_seconds = self.client.velocity(x0)
        self.layer_seconds.append(time.perf_counter() - t0)
        h = torch.from_numpy(out.astype(np.float32))[:self.S_real]
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.final_norm
        return (h @ self.llm2vae[0].T + self.llm2vae[1])[1:-1]

    def solve(self, noise, steps=32, cancelled=None, on_progress=None):
        state = noise.to(torch.float32).clone()
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            raw = float(torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20))
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            raw_mid = float(torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20))
            state = state - self.velocity(mid, raw_mid) * dt
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        if not torch.isfinite(state).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return state

    def close(self):
        self.client.close_session()
