"""Client for the iPhone companion: weights, programs, and per-song sessions."""
from __future__ import annotations
import hashlib, os, threading, time
from pathlib import Path
import numpy as np

from .protocol import Connection, RemoteError
from ..ane.mil import MIL_VERSION, WEIGHT_SHAPES, arrays_from_state, build_program, weight_input_shapes
from ..lean import nar_layer_state

S_STEP, P_STEP = 512, 1024
QBLK, KCHUNK = 128, 1024                          # small query tiles bound full-key softmax memory
REMOTE_PROGRAM_VERSION = "coreml2"                # invalidate only phone programs, not weights or Mac ANE caches
ROW_BLOCK, HEAD_NORM = None, "matmul"             # the plain layer form: proven up to 4096 rows on an iPhone 17 Pro
LAYERS = 2                                        # layers per program call
MAX_ROWS = 4096                                   # longer songs stay on the Mac until the 6656 program compiles there
CACHE = Path(os.environ.get("YUE2_ANE_CACHE", Path.home() / ".cache" / "yue2-ane")) / "remote"


def bucket(n, step):
    return max(step, -(-n // step) * step)


def program_name(S, P):
    form = f"rb{ROW_BLOCK}" if ROW_BLOCK else "hr" if HEAD_NORM == "reduce" else "plain"
    return f"layers{LAYERS}_{S}_{P}_{form}_q{QBLK}_k{KCHUNK}_{MIL_VERSION}_{REMOTE_PROGRAM_VERSION}"


def weight_identity(model):
    cached = getattr(model, "_yue2_weight_identity", None)
    if cached:
        return cached
    return hashlib.sha256(str(sorted((n, tuple(p.shape)) for n, p in model.named_parameters())).encode()).hexdigest()[:16]


class RemoteClient:
    """One connection to the phone. Methods are serialized with a lock; a song holds the
    session from ``open`` to ``close``."""

    def __init__(self, host, port, name=""):
        self.host, self.port, self.name = host, int(port), name or host
        self.conn = Connection(host, port)
        self.lock = threading.RLock()
        self.info = self.conn.call("hello")[0]
        if int(self.info.get("version", 0)) != 1:
            raise RemoteError(f"the iPhone app speaks protocol {self.info.get('version')}, expected 1")

    def close(self):
        self.conn.close()

    def label(self):
        return self.info.get("device") or self.name

    # -- weights ------------------------------------------------------------------
    def ensure_weights(self, model, on_progress=None):
        identity = weight_identity(model)
        if self.info.get("weights") == identity:
            return False
        cfg = model.config
        shapes = weight_input_shapes(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size)
        n = len(model.model.layers)
        with self.lock:
            self.conn.call("weights_begin", identity=identity, layers=n)
            t0 = time.perf_counter()
            sent = 0
            for i in range(n):
                arrays = arrays_from_state(nar_layer_state(model, i), cfg)
                parts = [np.ascontiguousarray(arrays[name].reshape(shapes[name])) for name in WEIGHT_SHAPES]
                layer_bytes = sum(part.nbytes for part in parts)
                self.conn.call_parts("weights_layer", parts, identity=identity, layer=i)
                sent += layer_bytes
                if on_progress is not None:
                    mb = sent / 2**20
                    on_progress(f"sending weights to the iPhone: layer {i + 1}/{n} ({mb / max(time.perf_counter() - t0, 1e-3):.0f} MB/s)")
            self.conn.call("weights_end", identity=identity)
        self.info["weights"] = identity
        return True

    # -- programs -----------------------------------------------------------------
    def ensure_program(self, S, P, cfg, on_progress=None):
        name = program_name(S, P)
        if name in self.info.get("programs", []):
            return name
        pkg = CACHE / f"{name}.mlpackage"
        if not pkg.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            build_program([None] * LAYERS, None, S=S, P=P, D=cfg.hidden_size, H=cfg.num_attention_heads, KV=cfg.num_key_value_heads,
                          HD=cfg.head_dim, F=cfg.intermediate_size, eps=cfg.rms_norm_eps, qblk=QBLK, kchunk=KCHUNK,
                          weights_as_inputs=True, target="iOS18", row_block=ROW_BLOCK, head_norm=HEAD_NORM,
                          attention_mode="softmax", package_path=pkg)
        files, blobs = [], []
        for path in sorted(p for p in pkg.rglob("*") if p.is_file()):
            data = path.read_bytes()
            files.append({"path": str(path.relative_to(pkg)), "size": len(data)})
            blobs.append(data)
        with self.lock:
            self.conn.call("program", b"".join(blobs), on_progress=on_progress, name=name, files=files)
        self.info.setdefault("programs", []).append(name)
        return name

    # -- session ------------------------------------------------------------------
    def open(self, S, P, S_real, P_real, program, on_progress=None):
        with self.lock:
            self.conn.call("open", on_progress=on_progress, S=S, P=P, S_real=S_real, P_real=P_real, program=program, layers=LAYERS)

    def send_kv(self, layer, pk, pv):
        """pk/pv: fp16 [KV, P_real, HD]."""
        with self.lock:
            self.conn.call("kv", np.ascontiguousarray(pk).tobytes() + np.ascontiguousarray(pv).tobytes(), layer=layer)

    def send_tables(self, cos, sin, bias):
        with self.lock:
            self.conn.call("tables", np.ascontiguousarray(cos).tobytes() + np.ascontiguousarray(sin).tobytes() + np.ascontiguousarray(bias).tobytes())

    def velocity(self, x):
        """x: fp16 [S, D] -> fp16 [S, D] after the 28 layers."""
        with self.lock:
            header, data = self.conn.call("velocity", np.ascontiguousarray(x).tobytes())
        return np.frombuffer(data, dtype=np.float16).reshape(x.shape), float(header.get("seconds", 0.0))

    def close_session(self):
        try:
            with self.lock:
                self.conn.call("close")
        except (RemoteError, OSError):
            pass
