#!/usr/bin/env python3
"""Long-lived YuE2 worker for native front ends: JSON lines in on stdin, JSON events out on stdout.

Songs are processes, the GPU and the Neural Engine are resources, and one scheduler assigns
resources to songs by priority (the order the songs were added). See ``Scheduler.schedule`` for
the whole policy; in short:

  Neural Engine   the highest-priority song that wants it, including a song already solving on
                  the GPU, which moves across mid-solve if it outranks everything waiting.
  GPU, tokenizing the highest-priority queued song, taking every other queued song of the same
                  kind along in one batch (one row or four cost the same per step).
  GPU, synthesis  drafts, and full-quality songs while the Neural Engine is busy. A queued song
                  that outranks the song synthesizing on the GPU starts its batch and the
                  synthesis pauses between steps until the GPU is free again.
  Rendering       always admitted: its tiles interleave with token steps.

Requests: {"cmd": "generate", "style", "lyrics", "cot": "full|melody|off", "seed", "random_seed", "batch",
           "max_tokens", "abc": str|null, "quality": "draft|full", "draft_steps", "instrumental": bool}
          {"cmd": "render", "path": song directory or its audio.flac, "quality": "full|draft"}
          {"cmd": "cancel", "path"}   {"cmd": "stop"}   {"cmd": "ping"}   {"cmd": "quit"}
Events:   {"event": "ready"}   {"event": "log", "message"}   {"event": "pong"}   {"event": "error", "message"}
          {"event": "started", "job", "output", "songs": [{"index", "seed", "path", "priority"}]}
          {"event": "stage", "path", "priority", "stage": "queued|planning|tokens|synth|decode|ready|failed|cancelled",
           "detail", "engine"}
          {"event": "progress", "path", "fraction": 0-1, "detail", "gflops": rate}
          {"event": "song", "index", "path", "score", "seconds", "seed", "truncated", "quality", "steps", "engine"}
          {"event": "failed", "path", "message"}   {"event": "idle"}   (every song finished or was cancelled)
"""
import datetime as dt, itertools, json, os, sys, threading, time, traceback
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")          # coremltools progress bars would otherwise flood the app log
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("YUE2_OUTPUT_DIR", ROOT / "outputs" / "app"))
PIPE = None
LOCK = threading.Lock()                              # stdout
MODEL_LOCK = threading.Lock()                        # model load/unload and memory release
IDLE_UNLOAD_S = float(os.environ.get("YUE2_IDLE_UNLOAD_S", 600))   # drop the model after this long idle (reloads in ~1 s)
LAST_ACTIVE = [time.time()]
PHYSICAL_GIB = float(os.environ.get("YUE2_PHYSICAL_GIB") or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
CONCURRENT = os.environ.get("YUE2_PIPELINE", "1" if PHYSICAL_GIB >= 24 else "0") != "0"   # overlap the resources
MAX_BATCH = int(os.environ.get("YUE2_MAX_BATCH", 4 if PHYSICAL_GIB >= 24 else 2))          # songs per token batch
ANE_MAX_FRAMES = 12288    # MIL v7 compiles up to here (9216 x 14336 verified); 12288 x 14336 is refused, and a
                          # refused compile fails within seconds and falls back to the GPU
# PyTorch's Metal backend encodes every thread into one command buffer, so all PyTorch GPU work
# (token steps, the synthesis prefill, decode tiles) takes turns on this lock; the Neural Engine
# and MLX solvers run outside it, which is what lets synthesis overlap token generation.
from yue2.locks import FairLock
TORCH_LOCK = FairLock()


def emit(**event):
    event.setdefault("t", round(time.time(), 2))
    with LOCK:
        sys.stdout.write(json.dumps(event) + "\n"); sys.stdout.flush()


def log(message):
    emit(event="log", message=message, time=dt.datetime.now().strftime("%H:%M:%S"))


def pipeline():
    """The loaded pipeline (loading or reloading the model as needed). Call under MODEL_LOCK."""
    global PIPE
    if PIPE is None:
        import torch
        from yue2 import YuE2Pipeline
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        log(f"Loading YuE2 model on {device} (first run only)")
        t0 = time.perf_counter()
        # Apple Silicon: synthesis runs on the Neural Engine (or MLX), so only the AR path is
        # kept in PyTorch; the NAR weights are read from the checkpoint by those engines.
        lean = device == "mps" and os.environ.get("YUE2_LEAN", "1") != "0"
        # The VAE decodes in tiles; smaller tiles halve its activation peak (about 3.6 GB at 1024
        # frames) on machines where the whole memory is shared with the model.
        PIPE = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device=device, progress=False, lean=lean,
                                            vae_core_frames=1024 if PHYSICAL_GIB >= 24 else 512)
        log(f"Physical memory {PHYSICAL_GIB:.0f} GB: {'lean' if lean else 'full'} model, VAE tile {PIPE.vae_core_frames} frames, "
            f"token batches of up to {MAX_BATCH}, resources {'overlap' if CONCURRENT else 'take turns'}")
        PIPE._load_model()
        log(f"Model ready in {time.perf_counter() - t0:.0f} s")
    elif PIPE._model is None:                        # dropped while idle
        t0 = time.perf_counter()
        PIPE._load_model()
        log(f"Model reloaded in {time.perf_counter() - t0:.0f} s")
    return PIPE


def acquire_model():
    with MODEL_LOCK:
        LAST_ACTIVE[0] = time.time()
        pipe = pipeline()
        return pipe, pipe._load_model()


def ane_available():
    from yue2.ane import runtime as ane_runtime
    return ane_runtime.available()


def ane_can_take(n_frames):
    from yue2.ane import runtime as ane_runtime
    return ane_available() and ane_runtime.bucket(n_frames + 2, ane_runtime.S_STEP) <= ANE_MAX_FRAMES


# ── Throughput estimates (rough, for the status line) ────────────────────────
# Matmul work only, 2 FLOP per multiply-add, derived from the model shapes; attention counted as
# QK^T and PV over the attended length. The VAE figure was measured with torch's FLOP counter.
VAE_GFLOP_PER_FRAME = 2.31


def _shapes(cfg):
    return cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.num_hidden_layers


def flops_per_token(cfg, context):
    """One autoregressive step of one sequence attending ``context`` tokens."""
    D, F, H, KV, HD, L = _shapes(cfg)
    linear = 2 * (D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F)
    attention = 4 * context * H * HD
    return L * (linear + attention) + 2 * D * 32770             # sliced output head (song phase)


def flops_per_pass(cfg, S, P):
    """One velocity evaluation of the synthesis network over S frames attending S + P keys."""
    D, F, H, KV, HD, L = _shapes(cfg)
    linear = 2 * S * (D * H * HD + 2 * D * KV * HD + H * HD * D + 3 * D * F)
    attention = 4 * S * (S + P) * H * HD
    return L * (linear + attention)


class Rate:
    """Smoothed throughput in GFLOP/s from (work done, time) samples."""

    def __init__(self, alpha=0.3):
        self.alpha, self.value, self.last = alpha, None, None

    def add(self, flops):
        now = time.perf_counter()
        if self.last is not None and now > self.last:
            sample = flops / (now - self.last) / 1e9
            self.value = sample if self.value is None else self.alpha * sample + (1 - self.alpha) * self.value
        self.last = now
        return self.value


# ── Processes ────────────────────────────────────────────────────────────────

QUEUED, PLANNING, TOKENIZING, SYNTH_WAIT, SYNTHING, RENDER_WAIT, RENDERING, DONE, FAILED, CANCELLED = (
    "queued", "planning", "tokenizing", "synth_wait", "synthing", "render_wait", "rendering", "done", "failed", "cancelled")
UI_STAGE = {QUEUED: "queued", PLANNING: "planning", TOKENIZING: "tokens", SYNTH_WAIT: "synth", SYNTHING: "synth",
            RENDER_WAIT: "decode", RENDERING: "decode", DONE: "ready", FAILED: "failed", CANCELLED: "cancelled"}


class Song:
    """One song on its way through the stages: a process with a priority (its arrival number)."""
    _sequence = itertools.count(1)

    def __init__(self, run, index, seed, request, directory, quality, steps, limit, instrumental=False):
        self.priority = next(Song._sequence)
        self.run, self.index, self.seed, self.request = run, index, seed, request
        self.directory = Path(directory)
        self.path = str(self.directory / "audio.flac")
        self.quality, self.steps, self.limit, self.instrumental = quality, steps, limit, instrumental
        self.mode = request.cot
        self.needs_plan = request.cot != "off" and request.abc is None
        self.cancel = threading.Event()
        self.state = QUEUED
        self.plan = None; self.codec = None; self.timing = {}; self.truncated = False
        self.latents = None; self.used_engine = None; self.nar_seconds = 0.0
        self.wants_ane = False                    # decided once the song's length is known
        self.program_ready = threading.Event()    # its Neural Engine program is compiled
        self.program_failed = False
        self.migrate = False                      # the scheduler granted it the Neural Engine mid-solve
        self.migrated = False

    @property
    def label(self):
        return f"{self.directory.parent.name}/{self.directory.name}"

    @property
    def frames(self):
        return len(self.codec) if self.codec is not None else 0

    def set_state(self, state, detail="", **extra):
        self.state = state
        emit(event="stage", path=self.path, priority=self.priority, stage=UI_STAGE[state], detail=detail, **extra)

    def progress(self, fraction, detail="", gflops=None):
        extra = {} if gflops is None else {"gflops": round(gflops, 1)}
        emit(event="progress", path=self.path, fraction=max(0.0, min(1.0, fraction)), detail=detail, **extra)

    def decide_engine(self):
        """Full-quality songs go to the Neural Engine when its compiler accepts their length."""
        self.wants_ane = self.quality == "full" and ane_can_take(self.frames)


# ── Scheduler ────────────────────────────────────────────────────────────────

class Scheduler:
    """Assigns the GPU and the Neural Engine to songs by priority. Every event (a song arrives, a
    stage ends, a resource frees) calls ``tick``; ``schedule`` holds the entire policy."""

    def __init__(self):
        self.cv = threading.Condition()
        self.songs = []                           # live processes, any state before done/failed/cancelled
        self.token_batch = None                   # songs tokenizing together on the GPU
        self.gpu_synth = None                     # song synthesizing on the GPU
        self.ane_synth = None                     # song on the Neural Engine (or granted it)
        self.render = None                        # song rendering on the GPU
        self.ane_prefill = False                  # the Neural Engine song is briefly on the GPU (prefill)

    # -- bookkeeping ---------------------------------------------------------
    def submit(self, songs):
        with self.cv:
            self.songs.extend(songs)
        for s in songs:
            s.set_state(s.state, "waiting for the GPU" if s.state == QUEUED else "waiting")
        self.tick()

    def finish(self, song, state, detail="", **extra):
        with self.cv:
            if song not in self.songs:
                return
            self.songs.remove(song)
            for slot in ("gpu_synth", "ane_synth", "render"):
                if getattr(self, slot) is song:
                    setattr(self, slot, None)
            empty = not self.songs
        song.set_state(state, detail, **extra)
        if empty:
            self.drained()
        else:
            self.tick()

    def drained(self):
        LAST_ACTIVE[0] = time.time()
        with MODEL_LOCK:
            with self.cv:
                if self.songs:
                    return
            try:
                with TORCH_LOCK:
                    release()
            except Exception as exc:
                log(f"Memory release failed: {exc}")
        emit(event="idle")

    def cancel(self, song, reason="cancelled"):
        song.cancel.set()
        if song.state in (QUEUED, SYNTH_WAIT, RENDER_WAIT):
            self.finish(song, CANCELLED, reason)         # waiting: gone at once
        else:
            log(f"Cancelling {song.label} ({song.state})")   # running: its thread drops it at the next check

    def stop(self):
        with self.cv:
            songs = sorted(self.songs, key=lambda s: s.priority)
        log(f"Stop requested: cancelling {len(songs)} song(s)")
        for s in songs:
            self.cancel(s, "stopped")

    def find(self, path):
        with self.cv:
            for s in self.songs:
                if s.path == path or str(s.directory) == path:
                    return s
        return None

    def tick(self):
        with self.cv:
            starts = self.schedule()
        for target, arg in starts:
            threading.Thread(target=target, args=(arg,), daemon=True).start()

    # -- the policy ------------------------------------------------------------
    def schedule(self):
        """Decide what starts now. Returns the worker functions to run in threads."""
        live = sorted(self.songs, key=lambda s: s.priority)
        def first(state, ok=lambda s: True):
            return next((s for s in live if s.state == state and not s.cancel.is_set() and ok(s)), None)
        running = any(x is not None for x in (self.token_batch, self.gpu_synth, self.ane_synth, self.render))
        starts = []

        # Neural Engine: the highest-priority song that wants it, whether waiting or already on the GPU.
        if self.ane_synth is None and (CONCURRENT or not running):
            waiting = first(SYNTH_WAIT, lambda s: s.wants_ane)
            g = self.gpu_synth
            movable = g if (g is not None and g.wants_ane and g.program_ready.is_set() and not g.migrate and not g.cancel.is_set()) else None
            best = min((s for s in (waiting, movable) if s is not None), key=lambda s: s.priority, default=None)
            if best is not None and best is movable:
                best.migrate = True; self.ane_synth = best          # picked up at its next step boundary
            elif best is not None:
                self.ane_synth = best; best.state = SYNTHING
                starts.append((run_ane, best))
                running = True

        # Rendering: always admitted (tiles interleave with token steps; a GPU synthesis pauses for it).
        if self.render is None and (CONCURRENT or not running):
            s = first(RENDER_WAIT)
            if s is not None:
                self.render = s; s.state = RENDERING
                starts.append((run_render, s))
                running = True

        # GPU: the highest-priority song that needs it decides between tokenizing and synthesis.
        q = first(QUEUED)
        cand = first(SYNTH_WAIT, lambda s: not s.wants_ane or self.ane_synth is not None) if self.gpu_synth is None else None
        if q is not None and cand is not None and cand.priority < q.priority:
            q = None                                            # the synthesis candidate outranks the queue
        # Tokenizing: the top queued song and every queued song of the same kind with it. It also
        # starts over a lower-priority GPU synthesis, which then pauses (preemption).
        if q is not None and self.token_batch is None and (CONCURRENT or not running) \
                and (self.gpu_synth is None or q.priority < self.gpu_synth.priority):
            batch = [s for s in live if s.state == QUEUED and not s.cancel.is_set()
                     and s.mode == q.mode and s.needs_plan == q.needs_plan][:MAX_BATCH]
            self.token_batch = batch
            for s in batch:
                s.state = PLANNING if s.needs_plan else TOKENIZING
            starts.append((run_batch, batch))
            running = True
        # Synthesis on the GPU: drafts, and full-quality songs while the Neural Engine is busy (they
        # move across when granted). Only when no batch runs and no queued song outranks it.
        elif cand is not None and q is None and self.token_batch is None and (CONCURRENT or not running):
            self.gpu_synth = cand; cand.state = SYNTHING
            starts.append((run_gpu, cand))
        return starts

    def gpu_wanted_elsewhere(self):
        """A GPU synthesis pauses while this is true (tokenizing, rendering, or the Neural Engine
        song's prefill are on the GPU: MLX's long kernels would starve them)."""
        return self.token_batch is not None or self.render is not None or self.ane_prefill

    def summary(self):
        with self.cv:
            return {"queued": sum(s.state == QUEUED for s in self.songs), "live": len(self.songs)}


SCHED = Scheduler()


# ── Workers (one thread each, started by the scheduler) ───────────────────────

def fail(song, exc):
    traceback.print_exc(file=sys.stderr)
    log(f"{song.label} failed: {type(exc).__name__}: {exc}")
    emit(event="failed", path=song.path, message=f"{type(exc).__name__}: {exc}")
    SCHED.finish(song, FAILED, str(exc)[:200])


def save_tokens(song):
    """Persist the plan and song tokens as soon as they exist, so a song can be synthesized later
    (render command) even if the worker stops before its audio is written."""
    import numpy as np
    from yue2.storage import write_json
    song.directory.mkdir(parents=True, exist_ok=True)
    song.plan.save(song.directory)
    np.save(song.directory / "semantic.npy", np.asarray(song.codec, dtype=np.int32))
    write_json(song.directory / "request.json", song.request.to_dict())
    write_json(song.directory / "tokens.json", {"seed": song.seed, "frames": len(song.codec), "quality": song.quality,
                                               "steps": song.steps, "priority": song.priority, "truncated": bool(song.truncated),
                                               "timing": song.timing})


def run_batch(batch):
    """Plan (if needed) and tokenize a batch of songs together on the GPU."""
    import dataclasses
    from yue2.batched import generate_tokens_batched
    from yue2.pipeline import SymbolicPlan
    from yue2.protocol import CODEC_OFFSET, token_prefixes
    songs = list(batch)
    try:
        pipe, model = acquire_model()
        # The GPU is ours now: take along any songs of the same kind that queued meanwhile (a job
        # submitted during the model load, or while the GPU was busy) up to the batch limit.
        with SCHED.cv:
            if SCHED.token_batch is batch:
                for s in sorted(SCHED.songs, key=lambda s: s.priority):
                    if len(songs) >= MAX_BATCH:
                        break
                    if s.state == QUEUED and not s.cancel.is_set() and s.mode == songs[0].mode and s.needs_plan == songs[0].needs_plan and s not in songs:
                        s.state = PLANNING if s.needs_plan else TOKENIZING
                        songs.append(s)
                SCHED.token_batch = songs
        n = len(songs)
        tokenizer = pipe.tokenizer
        cancelled = lambda: all(s.cancel.is_set() for s in songs)
        log(f"Tokenizing batch: {[s.label for s in songs]} (priorities {[s.priority for s in songs]})")
        counts = [0] * n; last = [0.0]

        def reporter(phase, expected, prefix_len):
            """Per-song progress and throughput: a batch step costs the same however many rows still
            produce, so each row's figure is its own tokens per second."""
            from yue2.protocol import ABC_END, MUSIC_END
            end = ABC_END if phase == "abc" else MUSIC_END
            rates, reported, finished = [Rate() for _ in songs], [0] * n, [False] * n
            for r in rates:
                r.add(0)
            def on_token(row, _phase, token):
                counts[row] += 1
                if token == end or (phase == "semantic" and counts[row] >= songs[row].limit):
                    finished[row] = True
                if time.perf_counter() - last[0] > 0.5 or finished[row]:
                    last[0] = time.perf_counter()
                    for i, s in enumerate(songs):
                        if finished[i]:
                            if reported[i] >= 0:
                                s.progress(1.0, "tokens finished"); reported[i] = -1
                            continue
                        gflops = rates[i].add((counts[i] - reported[i]) * flops_per_token(model.config, prefix_len + counts[i]))
                        reported[i] = counts[i]
                        detail = f"{counts[i]} score tokens" if phase == "abc" else f"{counts[i]} tokens (about {counts[i] / 25:.0f} s of audio)"
                        s.progress(min(1.0, counts[i] / expected), detail, gflops)
            return on_token

        requests = [s.request for s in songs]
        if songs[0].needs_plan:
            for s in songs:
                s.set_state(PLANNING, "planning the score")
            prefixes = [token_prefixes(r, tokenizer) for r in requests]
            rows, timing = generate_tokens_batched(model, prefixes, pipe.generation_config.abc, [s.seed for s in songs], "abc",
                                                   cancelled=cancelled, on_token=reporter("abc", 900, max(len(p) for p in prefixes)),
                                                   lock=TORCH_LOCK)
            plans = [SymbolicPlan(r, tokenizer.decode(ids), ids, token_prefixes(r, tokenizer, ids), t, trunc)
                     for r, (ids, t, trunc) in zip(requests, rows)]
            log(f"Scores planned: {[len(p.abc_ids) for p in plans]} tokens in {timing['seconds']:.0f} s")
            if any(s.instrumental for s in songs):
                # Re-plan from each score with its vocal voice silenced: the tokens then carry no sung melody.
                from yue2.instrumental import silence_vocals
                plans = [pipe.plan(request=dataclasses.replace(p.request, abc=silence_vocals(p.abc))) if (s.instrumental and p.abc) else p
                         for s, p in zip(songs, plans)]
                log("Instrumental: vocal voice silenced in the planned score(s)")
        else:
            plans = [pipe.plan(request=r) for r in requests]
        counts[:] = [0] * n
        for s in songs:
            s.set_state(TOKENIZING, "generating song tokens")
        limits = [s.limit for s in songs]
        sampling = dataclasses.replace(pipe.generation_config.semantic, max_tokens=max(limits))

        released = set()
        def release_song(i, tokens, t, truncated):
            s, plan = songs[i], plans[i]
            released.add(i)
            s.plan, s.codec, s.timing, s.truncated = plan, [int(x) - CODEC_OFFSET for x in tokens], t, truncated
            s.decide_engine()
            save_tokens(s)
            if s.cancel.is_set():
                SCHED.finish(s, CANCELLED, "stopped"); return
            log(f"Song tokens for {s.label}: {len(tokens)} ({t['seconds']:.0f} s)" + (", truncated" if truncated else "")
                + ("; Neural Engine" if s.wants_ane else "; GPU"))
            s.set_state(SYNTH_WAIT, "waiting", engine="ane" if s.wants_ane else "mlx")
            SCHED.tick()                                   # synthesis can start while the batch continues
        def on_row_done(i, tokens, t):
            if i not in released:
                release_song(i, tokens, t, bool(t.get("truncated", False)))

        rows, timing = generate_tokens_batched(model, [p.prefix for p in plans], sampling, [s.seed for s in songs], "semantic",
                                               legacy_off=(songs[0].mode == "off"), cancelled=cancelled,
                                               on_token=reporter("semantic", max(limits), max(len(p.prefix) for p in plans)),
                                               lock=TORCH_LOCK, on_row_done=on_row_done, limits=limits)
        log(f"Batch done: {[len(r[0]) for r in rows]} tokens in {timing['seconds']:.0f} s ({1000 * (timing['mean_step_seconds'] or 0):.0f} ms per step)")
        for i, (tokens, t, truncated) in enumerate(rows):
            if i in released:
                songs[i].timing = t
            else:
                release_song(i, tokens, t, truncated)
    except InterruptedError:
        for s in songs:
            if s.state in (PLANNING, TOKENIZING):
                SCHED.finish(s, CANCELLED, "stopped")
    except Exception as exc:
        for s in songs:
            if s.state in (PLANNING, TOKENIZING):
                fail(s, exc)
    finally:
        with SCHED.cv:
            SCHED.token_batch = None
        SCHED.tick()


def synth_common(song, model, pipe):
    """Progress, throughput and phase callbacks shared by both synthesis resources."""
    audio_s = song.frames / 25
    S, P = song.frames + 2, len(song.plan.prefix) + song.frames + 1
    per_step = 2 * flops_per_pass(model.config, S, P)               # midpoint solver: two passes per step
    rate, seen = Rate(), [0]
    def on_nar(done, total):
        if seen[0] == 0:
            rate.last = time.perf_counter(); gflops = None            # the first step also carried the prefill/compile: skip it
        else:
            gflops = rate.add((done - seen[0]) * per_step)
        seen[0] = done
        song.progress(done / max(total, 1), f"solver step {done}/{total}", gflops)
    def on_prepare(done, total):
        song.progress(0.0, f"compiling Neural Engine program {done}/{total}")
        if done in (1, total):
            log(f"Neural Engine program {done}/{total} ready for {song.label}")
    return dict(steps=song.steps, context=pipe.generation_config.context, cancelled=song.cancel.is_set,
                on_progress=on_nar, on_prepare=on_prepare, lock=TORCH_LOCK), audio_s, (S, P)


WARMING = threading.Lock()


def precompile(song, model, why):
    """Compile a song's Neural Engine program in the background; sets song.program_ready."""
    from yue2.ane import runtime as ane_runtime
    S, P = song.frames + 2, len(song.plan.prefix) + song.frames + 1
    bucket = ane_runtime.buckets_for((S, P))
    programs = ane_runtime.programs_for(model)
    if bucket in programs.loaded:
        song.program_ready.set(); return
    def work():
        with WARMING:                                 # the compiler service is single-threaded anyway
            try:
                if bucket not in programs.loaded:
                    log(f"Background: compiling Neural Engine programs for {song.label} (bucket {bucket[0]} x {bucket[1]}) {why}")
                    t0 = time.perf_counter()
                    programs.precompile(*bucket)
                    log(f"Background: Neural Engine programs for {song.label} ready in {time.perf_counter() - t0:.0f} s")
            except Exception as exc:
                song.program_failed = True
                log(f"{song.label}: the Neural Engine cannot compile its program ({str(exc).splitlines()[0][:100]})")
            finally:
                song.program_ready.set()
                SCHED.tick()
    threading.Thread(target=work, daemon=True).start()


def run_ane(song):
    """Synthesize on the Neural Engine."""
    from yue2.nar import synthesize
    try:
        pipe, model = acquire_model()
        kwargs, audio_s, _ = synth_common(song, model, pipe)
        log(f"Synthesizing {song.label} on the Neural Engine: about {audio_s:.0f} s of audio, {song.quality} quality ({song.steps} steps)")
        song.set_state(SYNTHING, "preparing", engine="ane")
        warmed = [False]
        on_nar = kwargs["on_progress"]
        def on_nar_warm(done, total):
            if not warmed[0]:                          # solving: compile the next Neural Engine song's program meanwhile
                warmed[0] = True
                with SCHED.cv:
                    nxt = next((s for s in sorted(SCHED.songs, key=lambda s: s.priority)
                                if s.state == SYNTH_WAIT and s.wants_ane and not s.program_ready.is_set()), None)
                if nxt is not None:
                    precompile(nxt, model, "while the current song solves")
            on_nar(done, total)
        def on_phase(text):
            song.progress(0.0, text)
            with SCHED.cv:
                SCHED.ane_prefill = text.startswith("prefilling")
            SCHED.tick()
        kwargs.update(on_progress=on_nar_warm, on_phase=on_phase)
        song.used_engine = "ane"
        t0 = time.perf_counter()
        try:
            latents = synthesize(model, song.plan.prefix, song.codec, song.seed, engine="ane", offload_ar=pipe.offload_ar, **kwargs)
        except RuntimeError as exc:
            if not str(exc).startswith("compile"):
                raise
            # The compiler rejects some very large shapes; the song goes back to wait for the GPU.
            log(f"The Neural Engine cannot compile programs for {song.label} ({str(exc).splitlines()[0][:100]}); it will use the GPU")
            song.wants_ane = False; song.program_failed = True
            with SCHED.cv:
                SCHED.ane_synth = None; SCHED.ane_prefill = False
            song.set_state(SYNTH_WAIT, "waiting for the GPU", engine="mlx")
            SCHED.tick(); return
        song.latents = latents.detach().float().cpu().numpy()
        song.nar_seconds = time.perf_counter() - t0
        log(f"Synthesis of {song.label} done in {song.nar_seconds:.0f} s ({song.nar_seconds / audio_s:.1f} s per second of audio, "
            f"{song.steps} steps, ane)")
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        if song.cancel.is_set():
            SCHED.finish(song, CANCELLED, "stopped"); return
        song.set_state(RENDER_WAIT, "waiting", engine="ane")
        SCHED.tick()
    except InterruptedError:
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        with SCHED.cv:
            SCHED.ane_synth = None; SCHED.ane_prefill = False
        fail(song, exc)


def run_gpu(song):
    """Synthesize on the GPU (MLX). A full-quality song moves to the Neural Engine when the
    scheduler grants it, and pauses between steps whenever the GPU is wanted elsewhere."""
    from yue2.nar_switch import synthesize_switchable
    try:
        pipe, model = acquire_model()
        kwargs, audio_s, _ = synth_common(song, model, pipe)
        kwargs["on_phase"] = lambda text: song.progress(0.0, text)
        song.used_engine = "mlx"
        t0 = time.perf_counter()
        movable = song.wants_ane
        if movable:
            log(f"Synthesizing {song.label} on the GPU until the Neural Engine is granted: about {audio_s:.0f} s of audio ({song.steps} steps)")
            song.set_state(SYNTHING, "on the GPU · moves to the Neural Engine when granted", engine="mlx")
            precompile(song, model, "for the move")
            SCHED.tick()                                   # the program may already be compiled
        else:
            log(f"Synthesizing {song.label} on the GPU: about {audio_s:.0f} s of audio, {song.quality} quality ({song.steps} steps)")
            song.set_state(SYNTHING, "preparing", engine="mlx")
        idle_text = "on the GPU · moves to the Neural Engine when granted" if movable else "on the GPU"
        paused = [False]
        def should_wait():
            # Tokenizing, rendering and the Neural Engine song's prefill own the GPU: MLX's long
            # kernels would starve them, so this song sits out until they are done.
            busy = SCHED.gpu_wanted_elsewhere()
            if busy != paused[0]:
                paused[0] = busy
                if busy:
                    what = "tokenizing" if SCHED.token_batch is not None else "rendering" if SCHED.render is not None else "prefilling for the Neural Engine"
                    log(f"{song.label} pauses: the GPU is {what}")
                    song.set_state(SYNTHING, f"paused · the GPU is busy {what}", engine="mlx")
                else:
                    log(f"{song.label} resumes on the GPU")
                    song.set_state(SYNTHING, idle_text, engine="mlx")
            return busy
        def on_switch(step):
            song.migrated = True; song.used_engine = "ane"
            log(f"{song.label} moved to the Neural Engine at solver step {step}/{song.steps}")
            song.set_state(SYNTHING, f"moved to the Neural Engine at step {step}", engine="ane")
            with SCHED.cv:
                if SCHED.gpu_synth is song:
                    SCHED.gpu_synth = None                   # the GPU is free for the next song
            SCHED.tick()
        latents, used, switched_at = synthesize_switchable(model, song.plan.prefix, song.codec, song.seed,
                                                           may_switch=(lambda: song.migrate) if movable else None,
                                                           should_wait=should_wait, on_switch=on_switch, **kwargs)
        song.used_engine = "mlx+ane" if switched_at is not None else "mlx"
        song.latents = latents.detach().float().cpu().numpy()
        song.nar_seconds = time.perf_counter() - t0
        log(f"Synthesis of {song.label} done in {song.nar_seconds:.0f} s ({song.nar_seconds / audio_s:.1f} s per second of audio, "
            f"{song.steps} steps, {song.used_engine})")
        with SCHED.cv:
            if SCHED.gpu_synth is song:
                SCHED.gpu_synth = None
            if SCHED.ane_synth is song:
                SCHED.ane_synth = None
        if song.cancel.is_set():
            SCHED.finish(song, CANCELLED, "stopped"); return
        song.set_state(RENDER_WAIT, "waiting", engine=song.used_engine)
        SCHED.tick()
    except InterruptedError:
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        fail(song, exc)


def run_render(song):
    """Decode the latents to a waveform on the GPU and write the song's files."""
    from yue2.pipeline import SemanticResult, SongResult
    from yue2.storage import identity
    try:
        pipe, _ = acquire_model()
        song.set_state(RENDERING, "decoding waveform", engine=song.used_engine)
        rate, seen = Rate(), [0]
        rate.add(0)
        def on_tile(done, total):
            gflops = rate.add((done - seen[0]) * pipe.vae_core_frames * VAE_GFLOP_PER_FRAME * 1e9); seen[0] = done
            song.progress(done / max(total, 1), f"decoding tile {done}/{total}", gflops)
            if song.cancel.is_set():
                raise InterruptedError("stopped")
            TORCH_LOCK.yield_turn()                  # let a waiting token step or prefill in between tiles
        t1 = time.perf_counter()
        with TORCH_LOCK:
            audio = pipe.decode(song.latents, on_progress=on_tile)
        if song.cancel.is_set():
            raise InterruptedError("stopped")
        plan = song.plan
        config = pipe.effective_config(plan.request)
        config.update({"execution": "eager_batched", "nar_engine": song.used_engine, "ode_steps": song.steps, "quality": song.quality})
        semantic = SemanticResult(plan, song.codec, song.timing or {}, song.truncated)
        result_song = SongResult(audio, 48000, semantic, song.latents, config, pipe.weights,
                                 {"semantic": song.timing or {}, "nar_seconds": song.nar_seconds, "vae_seconds": time.perf_counter() - t1},
                                 identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights}))
        directory = song.directory
        if song.quality == "full" and (directory / "audio.flac").exists() and \
                json.loads((directory / "result.json").read_text()).get("quality") == "draft":
            (directory / "audio.flac").replace(directory / "draft.flac")      # keep the preview beside the final render
        result = result_song.save_artifacts(directory)
        result.update({"quality": song.quality, "ode_steps": song.steps, "nar_engine": song.used_engine, "priority": song.priority})
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        length = len(audio) / 48000
        log(f"Saved {directory / 'audio.flac'} ({length:.1f} s, {song.quality})")
        emit(event="song", index=song.index, path=song.path, score=plan.abc or "", seconds=round(length, 1), seed=song.seed,
             truncated=bool(song.truncated or plan.truncated), quality=song.quality, steps=song.steps, engine=song.used_engine)
        SCHED.finish(song, DONE)
    except InterruptedError:
        SCHED.finish(song, CANCELLED, "stopped")
    except Exception as exc:
        fail(song, exc)
    finally:
        song.latents = None


# ── Requests ─────────────────────────────────────────────────────────────────

def steps_for(quality, req):
    if quality == "draft":
        return max(1, min(int(req.get("draft_steps", 8)), 32))
    from yue2.protocol import GenerationConfig
    return (PIPE.generation_config if PIPE is not None else GenerationConfig()).ode_steps


def submit_generate(req):
    from yue2.protocol import SongRequest
    n = int(req.get("batch", 1)); mode = req.get("cot", "full")
    style, lyrics = req["style"].strip(), req["lyrics"].strip()
    instrumental = bool(req.get("instrumental"))
    if instrumental:
        from yue2.instrumental import instrumental_tags, structure_only
        style, lyrics = instrumental_tags(style), structure_only(lyrics)
        if mode == "off":
            mode = "full"                      # the vocal voice can only be silenced in a planned score
    quality = "draft" if req.get("quality", "draft") == "draft" else "full"
    base = int(time.time()) % 10_000_000 if req.get("random_seed") else int(req.get("seed", 831001))
    seeds = [base + i for i in range(n)]
    abc = (req.get("abc") or "").strip() or None
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = OUTPUT_DIR / stamp
    with SCHED.cv:
        taken = {s.directory.parent for s in SCHED.songs}
    while out_root in taken or out_root.exists():                                      # two jobs in one second
        stamp += "b"; out_root = OUTPUT_DIR / stamp
    steps = steps_for(quality, req)
    limit = max(1, min(int(req.get("max_tokens", 9000)), 9000))
    songs = []
    for i, seed in enumerate(seeds):
        request = SongRequest(style=style, lyrics=lyrics, cot=mode, seed=seed, abc=abc,
                              id=f"song{i + 1}", **({"cfg_scale": 1.0} if mode == "off" else {}))
        songs.append(Song(stamp, i + 1, seed, request, out_root / f"song{i + 1}", quality, steps, limit, instrumental))
    emit(event="started", job=stamp, output=str(out_root),
         songs=[{"index": s.index, "seed": s.seed, "path": s.path, "priority": s.priority} for s in songs])
    log(f"Queued {stamp}: {n} song(s), {quality} quality ({steps} steps), seeds {seeds}, priorities {[s.priority for s in songs]}"
        + (", instrumental" if instrumental else ""))
    SCHED.submit(songs)


def submit_render(req):
    """Synthesize a song from its saved tokens: a full-quality render of a draft, or a song whose
    tokens were saved but never synthesized. Same tokens, seed and noise."""
    import numpy as np
    from yue2.pipeline import SymbolicPlan
    directory = Path(req["path"])
    if directory.is_file():
        directory = directory.parent
    if SCHED.find(str(directory / "audio.flac")) is not None:
        emit(event="error", message=f"{directory.name} is already queued"); return
    quality = "draft" if req.get("quality", "full") == "draft" else "full"
    plan = SymbolicPlan.load(directory)
    codec = np.load(directory / "semantic.npy", allow_pickle=False).astype(int).tolist()
    previous = {}
    for name in ("result.json", "tokens.json"):
        if (directory / name).exists():
            previous = json.loads((directory / name).read_text()); break
    truncated = previous.get("truncated")
    truncated = bool(truncated.get("semantic", False) if isinstance(truncated, dict) else truncated)
    index = int(directory.name[4:]) if directory.name.startswith("song") and directory.name[4:].isdigit() else 1
    song = Song(directory.parent.name, index, plan.request.seed, plan.request, directory, quality, steps_for(quality, req), 9000)
    song.plan, song.codec, song.truncated = plan, codec, truncated
    song.timing = previous.get("timing", {}) if "frames" in previous else {}
    song.decide_engine()
    song.state = SYNTH_WAIT
    emit(event="started", job=directory.parent.name, output=str(directory.parent),
         songs=[{"index": index, "seed": song.seed, "path": song.path, "priority": song.priority}])
    log(f"Queued {song.label} for {quality} synthesis ({song.steps} steps, priority {song.priority}): about {len(codec) / 25:.0f} s of audio")
    SCHED.submit([song])


def submit(req):
    try:
        (submit_render if req.get("cmd") == "render" else submit_generate)(req)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        log(f"Error: {type(exc).__name__}: {exc}"); emit(event="error", message=f"{type(exc).__name__}: {exc}")


# ── Memory ───────────────────────────────────────────────────────────────────

def footprint_mb():
    """Physical footprint of this process (macOS), or None."""
    try:
        import subprocess
        out = subprocess.run(["/usr/bin/footprint", "-p", str(os.getpid())], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "phys_footprint:" in line:
                value, unit = line.split()[1:3]
                return float(value) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(unit, 1)
    except Exception:
        return None


def release(deep=False):
    """Free GPU/engine memory held between jobs: MLX weights and cache, ANE weight surfaces and
    program mappings, the VAE and the MPS cache. deep=True also drops the model (lean reload ~1 s)
    and the compiled ANE programs. Call under MODEL_LOCK with no live songs."""
    global PIPE
    import gc, torch
    before = footprint_mb()
    if PIPE is not None and PIPE._model is not None:
        model = PIPE._model
        model._yue2_mlx_weights = None
        programs = getattr(model, "_yue2_ane_programs", None)
        if programs is not None:                 # unmap programs before freeing the surfaces they bind
            with programs.lock:
                for key in list(programs.loaded):
                    for program in programs.loaded[key]:
                        if deep:
                            program.free()
                        elif program.resident:
                            program.unload()
                if deep:
                    programs.loaded.clear()
        weights = getattr(model, "_yue2_ane_weights", None)
        if weights is not None:
            weights.close(); model._yue2_ane_weights = None
        PIPE._vae = None
        if deep:
            PIPE._model = None
        del model, programs, weights             # no local may keep the model alive through the collection below
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
    except Exception:
        pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    after = footprint_mb()
    if before is not None and after is not None:
        log(f"{'Unloaded the model' if deep else 'Released working memory'}: {before / 1024:.1f} GB -> {after / 1024:.1f} GB")


def idle_watch():
    while True:
        time.sleep(30)
        if PIPE is not None and PIPE._model is not None and time.time() - LAST_ACTIVE[0] > IDLE_UNLOAD_S:
            with MODEL_LOCK:
                if SCHED.songs or time.time() - LAST_ACTIVE[0] <= IDLE_UNLOAD_S or PIPE._model is None:
                    continue
                with TORCH_LOCK:
                    release(deep=True)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    emit(event="ready", root=str(ROOT), concurrent=CONCURRENT, max_batch=MAX_BATCH)
    threading.Thread(target=idle_watch, name="idle_watch", daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            emit(event="error", message="bad json"); continue
        cmd = req.get("cmd")
        if cmd == "ping":
            emit(event="pong")
        elif cmd == "quit":
            SCHED.stop(); break
        elif cmd == "stop":
            SCHED.stop()
        elif cmd == "cancel":
            song = SCHED.find(str(Path(req.get("path", ""))))
            if song is None:
                emit(event="error", message="not queued")
            else:
                SCHED.cancel(song)
        elif cmd in ("generate", "render"):
            threading.Thread(target=submit, args=(req,), daemon=True).start()     # never block the command loop on a model load
        else:
            emit(event="error", message=f"unknown command {cmd}")


if __name__ == "__main__":
    main()
