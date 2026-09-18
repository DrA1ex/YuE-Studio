"""MIL generation for one YuE2 NAR (synthesis) layer, for the Apple Neural Engine.

coremltools is used purely offline to emit MIL text and the weight blob. The
program is compiled and run through ``libyue2ane`` (private ANE route); nothing
touches Core ML at run time.

Program signature (all fp16, declaration order = input binding order):
    x    [1, 1, S, D]        residual stream in (S = bucketed NAR length)
    pk   [1, KV, P, HD]      cached prefix keys for this layer (bucketed P)
    pv   [1, KV, P, HD]      cached prefix values
    cos  [1, 1, S, HD/2]     rotary tables for the NAR positions
    sin  [1, 1, S, HD/2]
    bias [1, 1, 1, P + S]    0 for real keys, -1e4 for padding
    -> h [1, 1, S, D]        residual stream out
"""
from __future__ import annotations
import json, os, shutil, tempfile, warnings
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np

EPS_SCALE = 1.0 / 16          # RMSNorm statistics computed on x/16 to keep x^2 inside fp16
NEG = -1e4
MIL_VERSION = "v7"                 # v6: weights as program inputs; v7: query heads on the batch axis (S up to 16384)                 # bump when the generated program changes; keys the on-disk cache
ACC_SHIFT = float(np.log(8.0))      # softmax numerator scaled by 1/8: keeps the unnormalized P@V sum inside fp16 (worst seen 38k)
                                    # without pushing the probability tail below fp16's normal range (a 1/4096 shift lost fidelity)


def layer_arrays(layer, config):
    """fp16 numpy weights for one decoder layer's NAR path (attention scale folded into q_norm)."""
    attn, mlp = layer.nar_self_attn, layer.nar_mlp
    return arrays_from_state(dict(
        in_norm=layer.nar_input_layernorm.weight, q=attn.q_proj.weight, k=attn.k_proj.weight, v=attn.v_proj.weight,
        o=attn.o_proj.weight, q_norm=attn.q_norm.weight, k_norm=attn.k_norm.weight, mlp_norm=layer.nar_pre_mlp_layernorm.weight,
        gate=mlp.gate_proj.weight, up=mlp.up_proj.weight, down=mlp.down_proj.weight), config)


def arrays_from_state(state, config):
    """Same, from a {name: torch tensor} dict (see yue2.lean.nar_layer_state)."""
    f16 = lambda t: t.detach().float().cpu().numpy().astype(np.float16)
    scale = config.head_dim ** -0.5
    out = {name: f16(t) for name, t in state.items()}
    out["q_norm"] = (out["q_norm"].astype(np.float32) * scale).astype(np.float16)
    return out


def build_layer_program(W, out_dir, **kw):
    """Single-layer program (kept for tools); see build_program."""
    return build_program([W], out_dir, **kw)


WEIGHT_SHAPES = dict(in_norm=("D",), q="H*HD,D", k="KV*HD,D", v="KV*HD,D", o="D,H*HD", q_norm=("HD",), k_norm=("HD",),
                     mlp_norm=("D",), gate="F,D", up="F,D", down="D,F")


def spread_matrices(n, HD=128):
    """(M, Tt) for the "spread" head norm: M [n*HD, n*HD] block-diagonal ones; Tt [n*HD, HD] with
    Tt[j, j % HD] = 1 (so w [1, HD] @ Tt^T tiles w across the heads)."""
    M = np.zeros((n * HD, n * HD), np.float16)
    for h in range(n):
        M[h * HD:(h + 1) * HD, h * HD:(h + 1) * HD] = 1.0
    Tt = np.zeros((n * HD, HD), np.float16)
    for j in range(n * HD):
        Tt[j, j % HD] = 1.0
    return M, Tt


def spread_input_shapes(H=16, KV=8, HD=128):
    return {"mq": (1, 1, H * HD, H * HD), "tq": (1, 1, H * HD, HD), "mk": (1, 1, KV * HD, KV * HD), "tk": (1, 1, KV * HD, HD)}


def weight_input_shapes(D=2048, H=16, KV=8, HD=128, F=6144):
    """4-D shapes of the per-layer weight inputs of a weights_as_inputs program (name -> shape)."""
    dims = dict(D=D, H=H, KV=KV, HD=HD, F=F)
    out = {}
    for name, spec in WEIGHT_SHAPES.items():
        if isinstance(spec, tuple):
            out[name] = (1, 1, 1, dims[spec[0]])
        else:
            rows, cols = spec.split(",")
            out[name] = (1, 1, eval(rows, {}, dims), eval(cols, {}, dims))
    return out


def build_program(Ws, out_dir, *, S, P, D=2048, H=16, KV=8, HD=128, F=6144, eps=1e-6, qblk=1024, kchunk=2048, weights_as_inputs=False,
                  target="macOS15", package_path=None, row_block=None, head_norm="matmul"):
    """Write model.mil + weights/weight.bin + meta.json for a stack of layers at bucket (S, P).

    Inputs: x, cos, sin, bias, and pk{i}/pv{i} for each layer i in the stack. Fewer, larger
    programs matter: the engine refuses to run more than ~17 distinct programs in sequence.

    weights_as_inputs=True declares every layer weight as a program input (w{i}_{name}) instead
    of a baked constant, so one compiled program serves all 28 layers (one compile per bucket
    instead of 14); Ws then only supplies the layer count.
   
    ``target`` names the coremltools deployment target (macOS15 for the private route, iOS18
    for the iPhone companion). ``package_path`` keeps the converted .mlpackage there instead of
    extracting model.mil into ``out_dir`` (public Core ML loads the package directly).
    ``row_block`` runs the projections, output projection and MLP in blocks of that many rows.
    ``head_norm`` picks the per-head RMSNorm form: "matmul" (block sums by a [n, n*HD] constant,
    spread back by its transpose; the Mac's route), "reduce" (a plain reduce over [.., S, HD];
    the Mac refuses it above 8192 rows, the iPhone refuses it outright at 6656) or "spread" (one
    matmul with a block-diagonal [n*HD, n*HD] constant that yields each head's sum already spread
    across its columns, and a tiny matmul in place of the tile op: only ops the iPhone's compiler
    accepted at 6656 rows in the plain layer form).
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    G = H // KV
    K = P + S
    qblk = min(qblk, S)          # the last query block may be shorter (S is a multiple of 512, not of qblk)

    def linear(x, w):
        return mb.matmul(x=x, y=w, transpose_y=True)

    def rms(x, w):
        # Statistics on x/16 keep x^2 inside fp16; rescale the variance back and add an
        # fp16-representable epsilon (6.1e-5, the smallest normal) so all-zero padding rows
        # give a finite rsqrt instead of inf. For real rows this differs negligibly from 1e-6.
        xs = mb.mul(x=x, y=np.float16(EPS_SCALE))
        var = mb.mul(x=mb.reduce_mean(x=mb.mul(x=xs, y=xs), axes=[-1], keep_dims=True), y=np.float16(1.0 / (EPS_SCALE * EPS_SCALE)))
        r = mb.rsqrt(x=mb.add(x=var, y=np.float16(6.1e-5)))
        return mb.mul(x=mb.mul(x=x, y=r), y=w)

    def rms_heads(x, w, n):
        """Per-head RMSNorm of x [1,1,S,n*HD] in the flat layout (the compiler rejects a reduce
        over [.., S, HD] once S exceeds 8192). Per-head mean squares come from a constant
        block-averaging matmul; the multipliers are spread back with its transpose."""
        block = np.zeros((n, n * HD), np.float16)
        for h in range(n):
            block[h, h * HD:(h + 1) * HD] = 1.0          # block sums (a 1/HD weight would push products into fp16 subnormals)
        spread = block.T.copy()                                                     # [n*HD, n]
        xs = mb.mul(x=x, y=np.float16(EPS_SCALE))
        ss = mb.matmul(x=mb.mul(x=xs, y=xs), y=block, transpose_y=True)             # [1,1,S,n] sum of squares
        var = mb.mul(x=ss, y=np.float16(1.0 / (EPS_SCALE * EPS_SCALE * HD)))
        r = mb.matmul(x=mb.rsqrt(x=mb.add(x=var, y=np.float16(6.1e-5))), y=spread, transpose_y=True)   # [1,1,S,n*HD]
        wt = np.tile(w, n) if isinstance(w, np.ndarray) else mb.tile(x=w, reps=[1, 1, 1, n])
        return mb.mul(x=mb.mul(x=x, y=r), y=wt)

    def rms_heads4(t, w):
        """Per-head RMSNorm of t [1,R,n,HD] by a plain reduce (fine for R <= 2048); w [1,1,1,HD] or (HD,) broadcasts."""
        ts = mb.mul(x=t, y=np.float16(EPS_SCALE))
        var = mb.mul(x=mb.reduce_mean(x=mb.mul(x=ts, y=ts), axes=[-1], keep_dims=True), y=np.float16(1.0 / (EPS_SCALE * EPS_SCALE)))
        r = mb.rsqrt(x=mb.add(x=var, y=np.float16(6.1e-5)))
        return mb.mul(x=mb.mul(x=t, y=r), y=w)

    def rms_heads_spread(x, w, n, M=None, Tt=None):
        """Per-head RMSNorm of x [1,1,S,n*HD]: sums of squares spread across each head's columns by
        one block-diagonal matmul, so no [S, n] tensor, no reduce and no tile op exist. ``M``
        [1,1,n*HD,n*HD] and ``Tt`` [1,1,n*HD,HD] are the matrices as program inputs (the iPhone's
        compiler refuses the same matmuls with baked constants at 6656 rows); None bakes them."""
        xs = mb.mul(x=x, y=np.float16(EPS_SCALE))
        if M is None:
            M = spread_matrices(n)[0]
        ss = mb.matmul(x=mb.mul(x=xs, y=xs), y=M, transpose_y=True)                  # [1,1,S,n*HD] (M is symmetric)
        var = mb.mul(x=ss, y=np.float16(1.0 / (EPS_SCALE * EPS_SCALE * HD)))
        r = mb.rsqrt(x=mb.add(x=var, y=np.float16(6.1e-5)))
        if isinstance(w, np.ndarray):
            wt = np.tile(w, n)
        else:
            wt = mb.matmul(x=w, y=spread_matrices(n)[1] if Tt is None else Tt, transpose_y=True)   # [1,1,1,n*HD]
        return mb.mul(x=mb.mul(x=x, y=r), y=wt)

    def rotary(t, cos4, sin4):          # t [1, R, n, HD]; cos4/sin4 [1, R, 1, HD/2]
        half = HD // 2
        R, n = t.shape[1], t.shape[2]
        x1 = mb.slice_by_index(x=t, begin=[0, 0, 0, 0], end=[1, R, n, half])
        x2 = mb.slice_by_index(x=t, begin=[0, 0, 0, half], end=[1, R, n, HD])
        return mb.concat(values=[mb.sub(x=mb.mul(x=x1, y=cos4), y=mb.mul(x=x2, y=sin4)),
                                 mb.add(x=mb.mul(x=x2, y=cos4), y=mb.mul(x=x1, y=sin4))], axis=-1)

    def attend(qb, kk, vv, bias):       # qb [G,KV,R,HD]; kk/vv [1,KV,K,HD] (broadcast over G); bias [1,1,1,K]
        m = l = acc = None
        for c in range(0, K, kchunk):
            e = min(c + kchunk, K)
            kc = mb.slice_by_index(x=kk, begin=[0, 0, c, 0], end=[1, KV, e, HD])
            vc = mb.slice_by_index(x=vv, begin=[0, 0, c, 0], end=[1, KV, e, HD])
            bc = mb.slice_by_index(x=bias, begin=[0, 0, 0, c], end=[1, 1, 1, e])
            sc = mb.add(x=mb.matmul(x=qb, y=kc, transpose_y=True), y=bc)
            mc = mb.reduce_max(x=sc, axes=[-1], keep_dims=True)
            if m is None:
                m = mc
                p = mb.exp(x=mb.sub(x=sc, y=mb.add(x=m, y=np.float16(ACC_SHIFT))))
                l = mb.reduce_sum(x=p, axes=[-1], keep_dims=True)
                acc = mb.matmul(x=p, y=vc)
            else:
                m_new = mb.maximum(x=m, y=mc)
                alpha = mb.exp(x=mb.sub(x=m, y=m_new))
                p = mb.exp(x=mb.sub(x=sc, y=mb.add(x=m_new, y=np.float16(ACC_SHIFT))))
                l = mb.add(x=mb.mul(x=l, y=alpha), y=mb.reduce_sum(x=p, axes=[-1], keep_dims=True))
                acc = mb.add(x=mb.mul(x=acc, y=alpha), y=mb.matmul(x=p, y=vc))
                m = m_new
        return mb.real_div(x=acc, y=l)

    def attention(qg, kk, vv, bias):    # qg [G,KV,S,HD]
        # The query heads of a K/V group sit on the batch axis, not stacked along the rows: the
        # compiler rejects any dimension above 16384, and 2*S rows hit that at 8192 frames.
        outs = []
        for s in range(0, S, qblk):
            qb = mb.slice_by_index(x=qg, begin=[0, 0, s, 0], end=[G, KV, min(s + qblk, S), HD])
            outs.append(attend(qb, kk, vv, bias))
        return mb.concat(values=outs, axis=2) if len(outs) > 1 else outs[0]

    def heads_to_rows(att, R):          # [G,KV,R,HD] -> [1,1,R,H*HD] in head order h = j*G + g
        heads = mb.reshape(x=mb.transpose(x=att, perm=[1, 0, 2, 3]), shape=[1, H, R, HD])
        return mb.reshape(x=mb.transpose(x=heads, perm=[0, 2, 1, 3]), shape=[1, 1, R, H * HD])

    def mlp_out(r, W):                  # residual r [1,1,R,D] -> layer output
        mn = rms(r, W["mlp_norm"])
        return mb.add(x=r, y=linear(mb.mul(x=mb.silu(x=linear(mn, W["gate"])), y=linear(mn, W["up"])), W["down"]))

    def layer(x, W, pk, pv, cos4, sin4, bias):
        if row_block:
            return layer_blocked(x, W, pk, pv, cos4, sin4, bias)
        hn = rms(x, W["in_norm"])
        if head_norm == "reduce":
            q = rotary(rms_heads4(mb.reshape(x=linear(hn, W["q"]), shape=[1, S, H, HD]), W["q_norm"]), cos4, sin4)
            k = rotary(rms_heads4(mb.reshape(x=linear(hn, W["k"]), shape=[1, S, KV, HD]), W["k_norm"]), cos4, sin4)
        elif head_norm == "spread":
            sp = W.get("_spread", {})
            q = rotary(mb.reshape(x=rms_heads_spread(linear(hn, W["q"]), W["q_norm"], H, sp.get("mq"), sp.get("tq")), shape=[1, S, H, HD]), cos4, sin4)
            k = rotary(mb.reshape(x=rms_heads_spread(linear(hn, W["k"]), W["k_norm"], KV, sp.get("mk"), sp.get("tk")), shape=[1, S, KV, HD]), cos4, sin4)
        else:
            q = rotary(mb.reshape(x=rms_heads(linear(hn, W["q"]), W["q_norm"], H), shape=[1, S, H, HD]), cos4, sin4)
            k = rotary(mb.reshape(x=rms_heads(linear(hn, W["k"]), W["k_norm"], KV), shape=[1, S, KV, HD]), cos4, sin4)
        v = mb.reshape(x=linear(hn, W["v"]), shape=[1, S, KV, HD])
        # heads h = j*G + g: [1,S,H,HD] -> [KV,G,S,HD] -> [G,KV,S,HD] so group j shares K/V row j
        qg = mb.transpose(x=mb.reshape(x=mb.transpose(x=q, perm=[0, 2, 1, 3]), shape=[KV, G, S, HD]), perm=[1, 0, 2, 3])
        kk = mb.concat(values=[pk, mb.transpose(x=k, perm=[0, 2, 1, 3])], axis=2)
        vv = mb.concat(values=[pv, mb.transpose(x=v, perm=[0, 2, 1, 3])], axis=2)
        att = attention(qg, kk, vv, bias)                                                     # [G,KV,S,HD]
        r = mb.add(x=x, y=linear(heads_to_rows(att, S), W["o"]))
        return mlp_out(r, W)

    def layer_blocked(x, W, pk, pv, cos4, sin4, bias):
        """Same layer with every row-wise tensor (norms, projections, MLP) built in row blocks.

        The per-head norms use the reduce form here: the iPhone's compiler refuses the block-matmul
        form (and its tile op) at long lengths, while a reduce over [.., R, HD] is fine for R <= 2048.
        """
        R = row_block
        qs, ks, vs = [], [], []
        for r0 in range(0, S, R):
            r1 = min(r0 + R, S)
            xb = mb.slice_by_index(x=x, begin=[0, 0, r0, 0], end=[1, 1, r1, D])
            cb = mb.slice_by_index(x=cos4, begin=[0, r0, 0, 0], end=[1, r1, 1, HD // 2])
            sb = mb.slice_by_index(x=sin4, begin=[0, r0, 0, 0], end=[1, r1, 1, HD // 2])
            hn = rms(xb, W["in_norm"])
            qs.append(rotary(rms_heads4(mb.reshape(x=linear(hn, W["q"]), shape=[1, r1 - r0, H, HD]), W["q_norm"]), cb, sb))
            ks.append(rotary(rms_heads4(mb.reshape(x=linear(hn, W["k"]), shape=[1, r1 - r0, KV, HD]), W["k_norm"]), cb, sb))
            vs.append(mb.reshape(x=linear(hn, W["v"]), shape=[1, r1 - r0, KV, HD]))
        cat = lambda parts, axis: mb.concat(values=parts, axis=axis) if len(parts) > 1 else parts[0]
        q, k, v = cat(qs, 1), cat(ks, 1), cat(vs, 1)                                          # [1,S,n,HD]
        qg = mb.transpose(x=mb.reshape(x=mb.transpose(x=q, perm=[0, 2, 1, 3]), shape=[KV, G, S, HD]), perm=[1, 0, 2, 3])
        kk = mb.concat(values=[pk, mb.transpose(x=k, perm=[0, 2, 1, 3])], axis=2)
        vv = mb.concat(values=[pv, mb.transpose(x=v, perm=[0, 2, 1, 3])], axis=2)
        outs = []
        for r0 in range(0, S, min(R, qblk)):
            r1 = min(r0 + min(R, qblk), S)
            qb = mb.slice_by_index(x=qg, begin=[0, 0, r0, 0], end=[G, KV, r1, HD])
            xb = mb.slice_by_index(x=x, begin=[0, 0, r0, 0], end=[1, 1, r1, D])
            r = mb.add(x=xb, y=linear(heads_to_rows(attend(qb, kk, vv, bias), r1 - r0), W["o"]))
            outs.append(mlp_out(r, W))
        return cat(outs, 2)

    n = len(Ws)
    specs = [mb.TensorSpec(shape=(1, 1, S, D), dtype=types.fp16), mb.TensorSpec(shape=(1, 1, S, HD // 2), dtype=types.fp16),
             mb.TensorSpec(shape=(1, 1, S, HD // 2), dtype=types.fp16), mb.TensorSpec(shape=(1, 1, 1, K), dtype=types.fp16)]
    specs += [mb.TensorSpec(shape=(1, KV, P, HD), dtype=types.fp16) for _ in range(2 * n)]
    wshapes = weight_input_shapes(D, H, KV, HD, F)
    wnames = [f"w{i}_{name}" for i in range(n) for name in WEIGHT_SHAPES] if weights_as_inputs else []
    specs += [mb.TensorSpec(shape=wshapes[name.split("_", 1)[1]], dtype=types.fp16) for name in wnames]
    spread_names = list(spread_input_shapes()) if (head_norm == "spread" and weights_as_inputs) else []
    sshapes = spread_input_shapes(H, KV, HD)
    specs += [mb.TensorSpec(shape=sshapes[name], dtype=types.fp16) for name in spread_names]

    def body(x, cos, sin, bias, kvs, ws, sps):
        cos4 = mb.reshape(x=cos, shape=[1, S, 1, HD // 2])
        sin4 = mb.reshape(x=sin, shape=[1, S, 1, HD // 2])
        h = x
        for i, W in enumerate(Ws):
            if weights_as_inputs:
                # matmul(transpose_y) with a [1,1,out,in] weight broadcasts over the leading dims;
                # norm weights are [1,1,1,n] and broadcast the same way as the baked 1-D constants.
                W = {name: ws[i * len(WEIGHT_SHAPES) + j] for j, name in enumerate(WEIGHT_SHAPES)}
                W["_spread"] = dict(zip(spread_names, sps))
            h = layer(h, W, kvs[2 * i], kvs[2 * i + 1], cos4, sin4, bias)
        return mb.identity(x=h, name="h")

    # The builder maps input specs to named parameters, so synthesize a function with pk{i}/pv{i} names.
    kv_names = [f"{kind}{i}" for i in range(n) for kind in ("pk", "pv")]
    namespace = {"mb": mb, "body": body, "ct": ct, "specs": specs}
    exec(f"@mb.program(input_specs=specs, opset_version=ct.target.{target})\n"
         f"def prog(x, cos, sin, bias, {', '.join(kv_names + wnames + spread_names)}):\n"
         f"    return body(x, cos, sin, bias, [{', '.join(kv_names)}], [{', '.join(wnames)}], [{', '.join(spread_names)}])\n", namespace)
    prog = namespace["prog"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=getattr(ct.target, target),
                           outputs=[ct.TensorType(name="h", dtype=np.float16)], skip_model_load=True)
    meta = {"S": S, "P": P, "K": K, "D": D, "H": H, "KV": KV, "HD": HD, "qblk": qblk, "kchunk": kchunk, "layers": n,
            "weights_as_inputs": weights_as_inputs, "target": target, "row_block": row_block, "head_norm": head_norm,
            "inputs": ["x", "cos", "sin", "bias"] + kv_names + wnames + spread_names}
    if package_path is not None:
        package_path = Path(package_path)
        if package_path.exists():
            shutil.rmtree(package_path)
        model.save(str(package_path))
        if getattr(model, "is_temp_package", False) and getattr(model, "package_path", None):
            shutil.rmtree(model.package_path, ignore_errors=True)
        package_path.with_suffix(".meta.json").write_text(json.dumps(meta))
        return meta
    tmp = Path(tempfile.mkdtemp())
    pkg = tmp / "m.mlpackage"
    model.save(str(pkg))
    compiled = Path(ct.models.utils.compile_model(str(pkg), destination_path=str(tmp / "m.mlmodelc")))
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "weights").mkdir()
    shutil.copy(compiled / "model.mil", out / "model.mil")
    if (compiled / "weights" / "weight.bin").exists():
        shutil.copy(compiled / "weights" / "weight.bin", out / "weights" / "weight.bin")
    else:   # weights_as_inputs: no constants; the loader still expects a blob file
        (out / "weights" / "weight.bin").write_bytes(bytes(64))
    # coremltools leaves its own temporary package (~200 MB) and the compiled .mlmodelc behind
    # unless removed explicitly; a killed process would otherwise leak them into the temp folder.
    shutil.rmtree(tmp, ignore_errors=True)
    if getattr(model, "is_temp_package", False) and getattr(model, "package_path", None):
        shutil.rmtree(model.package_path, ignore_errors=True)
    mil = (out / "model.mil").read_text()
    meta.update(mil_ops=mil.count(")[name = string("), weights_MB=(out / "weights/weight.bin").stat().st_size // 2**20)
    (out / "meta.json").write_text(json.dumps(meta))
    return meta
