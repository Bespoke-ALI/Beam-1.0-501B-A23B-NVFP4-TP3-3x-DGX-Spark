#!/usr/bin/env python3
# pad_beam_tp3_v0_1_0.py -- zero-pad the Beam-1.0-501B-A23B-NVFP4 checkpoint so vLLM can shard it at TP=3.
# Method of record: Beam v1.1 review (Oct 5, 2026), Route A, after marianbusoi/Hy3-NVFP4-FP8-TP3.
#   query heads 48->54 and KV heads 8->9 (GQA ratio 6 kept: query head i still reads KV head i//6, so heads 0-47 are
#   untouched and the 6 new query heads read only the new all-zero KV head 8); dense MLP (layers 0-2, BF16)
#   20480->20736; routed experts (NVFP4: packed uint8 weights + FP8-E4M3 block scales) 2560->--moe-to.
#   Zeros are appended at the END of each padded dimension (or of each half, for fused gate|up / q|gate tensors).
#   Vocab is NOT padded here: vLLM's vocab padding constant is patched instead (as Hy3 did), so no fake tokens exist.
# Fail-closed: every tensor must match a known rule with the expected shape, or the run aborts before writing.
# Modes: --dry-run (plan only) | default (write) | --verify (byte-level proof vs source) | --numeric-test (CPU math).
# Needs only Python 3.10+ and numpy. Never loads a whole shard into memory.
import argparse, hashlib, json, os, re, shutil, struct, sys, time
import numpy as np

VERSION = "0.1.0"
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "I16": 2, "I32": 4, "I64": 8, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1}
FLOAT_DTYPES = ("BF16", "F16", "F32")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, meta, 8 + n


def storage_shape(dtype, shape):
    # shape in bytes along the last axis; all other axes in elements
    if not shape:
        return [ITEMSIZE[dtype]]
    return list(shape[:-1]) + [shape[-1] * ITEMSIZE[dtype]]


class Plan:
    """pads: list of (axis, mode, old_len, new_len) in STORAGE units (last axis in bytes); mode tail|halves."""
    def __init__(self, rule, pads=()):
        self.rule, self.pads = rule, list(pads)


class Rules:
    def __init__(self, cfg, a):
        self.H, self.KV, self.D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.HID, self.ID, self.IM = cfg["hidden_size"], cfg["intermediate_size"], cfg["moe_intermediate_size"]
        self.H2, self.KV2, self.ID2, self.IM2 = a.heads_to, a.kv_to, a.dense_to, a.moe_to
        self.qgate = a.qgate_layout
        assert self.H % self.KV == 0 and self.H2 % self.KV2 == 0 and self.H // self.KV == self.H2 // self.KV2, \
            "GQA ratio must be preserved"
        assert self.H2 > self.H and self.KV2 > self.KV
        for name, v in (("dense", self.ID2), ("moe", self.IM2)):
            assert v % (3 * 16) == 0, "%s target %d must give a per-rank width that is a multiple of 16" % (name, v)

    def plan(self, name, dtype, shape):
        H, KV, D, HID = self.H, self.KV, self.D, self.HID
        st = storage_shape(dtype, shape)
        m = re.match(r"^model\.layers\.(\d+)\.(.+)$", name)
        if not m:
            if re.match(r"^(model\.embed_tokens\.weight|lm_head\.weight|model\.norm\.weight)$", name):
                return Plan("unchanged-top")
            raise SystemExit("ABORT unknown top-level tensor: %s %s %s" % (name, dtype, shape))
        rest = m.group(2)
        if re.match(r"^[a-z_]*(layernorm|norm)\.weight$", rest):
            return Plan("unchanged-norm")
        if rest.startswith("self_attn."):
            return self._attn(name, rest[10:], dtype, shape, st)
        if rest.startswith("mlp."):
            return self._mlp(name, rest[4:], dtype, shape, st)
        raise SystemExit("ABORT unknown layer tensor: %s %s %s" % (name, dtype, shape))

    def _need(self, name, cond, what):
        if not cond:
            raise SystemExit("ABORT %s: %s" % (name, what))

    def _attn(self, name, t, dtype, shape, st):
        H, KV, D, HID, H2, KV2 = self.H, self.KV, self.D, self.HID, self.H2, self.KV2
        self._need(name, dtype in FLOAT_DTYPES, "attention tensor is %s, expected an unquantized float type" % dtype)
        isz = ITEMSIZE[dtype]
        U = (lambda n: n * isz) if len(shape) == 1 else (lambda n: n)   # axis 0 is the byte axis only for 1-D tensors
        if t in ("q_proj.weight", "q_proj.bias"):
            rows = shape[0]
            if rows == H * D:
                return Plan("q", [(0, "tail", U(rows), U(H2 * D))])
            self._need(name, rows == 2 * H * D, "q_proj has %d rows; expected %d or %d" % (rows, H * D, 2 * H * D))
            self._need(name, self.qgate in ("head_major", "halves"),
                       "q_proj carries a fused gate (2x rows); pass --qgate-layout head_major|halves (from the vLLM source)")
            return Plan("q+gate-" + self.qgate,
                        [(0, "tail" if self.qgate == "head_major" else "halves", U(rows), U(2 * H2 * D))])
        if t in ("k_proj.weight", "v_proj.weight", "k_proj.bias", "v_proj.bias"):
            self._need(name, shape[0] == KV * D, "rows %d != KV*D %d" % (shape[0], KV * D))
            return Plan("kv", [(0, "tail", U(KV * D), U(KV2 * D))])
        if t == "o_proj.weight":
            self._need(name, shape == [HID, H * D], "shape %s != [%d, %d]" % (shape, HID, H * D))
            return Plan("o", [(1, "tail", H * D * isz, H2 * D * isz)])
        if t == "o_proj.bias":
            return Plan("unchanged-obias")
        if t in ("q_norm.weight", "k_norm.weight"):
            n = shape[0]
            if n == D:
                return Plan("unchanged-qknorm-shared")
            if n == H * D and t == "q_norm.weight":
                return Plan("qnorm-perhead", [(0, "tail", n * isz, H2 * D * isz)])
            if n == KV * D and t == "k_norm.weight":
                return Plan("knorm-perhead", [(0, "tail", n * isz, KV2 * D * isz)])
            self._need(name, False, "unexpected norm length %d" % n)
        if re.match(r"^(gate_bias|sinks|attn_sinks)$", t):
            # Beam: self_attn.gate_bias [H], per-head bias of the output gate sigmoid(gate + bias); TP-sharded on dim 0
            self._need(name, shape == [H], "%s shape %s != [%d]" % (t, shape, H))
            return Plan("perhead-" + t, [(0, "tail", H * isz, H2 * isz)])
        if re.match(r"^(g_proj|gate_proj|attn_gate|gate|o_gate|output_gate)\.(weight|bias)$", t):
            rows = shape[0]
            self._need(name, rows % H == 0, "gate rows %d not a multiple of %d heads" % (rows, H))
            per = rows // H
            if len(shape) == 1:
                return Plan("gate-perhead", [(0, "tail", rows * isz, H2 * per * isz)])
            return Plan("gate-perhead", [(0, "tail", rows, H2 * per)])
        raise SystemExit("ABORT unknown attention tensor: %s %s %s (add a rule after reading the vLLM source)"
                         % (name, dtype, shape))

    def _mlp(self, name, t, dtype, shape, st):
        HID, ID, IM, ID2, IM2 = self.HID, self.ID, self.IM, self.ID2, self.IM2
        if t in ("gate.weight", "gate.e_score_correction_bias", "router.weight", "gate.bias"):
            return Plan("unchanged-router")
        if re.match(r"^shared_expert", t):
            raise SystemExit("ABORT shared expert tensor present but config says none: %s" % name)
        if not t.startswith("experts."):
            # dense MLP (layers 0-2), expected BF16
            self._need(name, dtype in FLOAT_DTYPES, "dense MLP tensor is %s, expected unquantized" % dtype)
            isz = ITEMSIZE[dtype]
            if t in ("gate_proj.weight", "up_proj.weight"):
                self._need(name, shape == [ID, HID], "shape %s != [%d, %d]" % (shape, ID, HID))
                return Plan("dense-gu", [(0, "tail", ID, ID2)])
            if t == "gate_up_proj.weight":
                self._need(name, shape == [2 * ID, HID], "shape %s != [%d, %d]" % (shape, 2 * ID, HID))
                return Plan("dense-gu-fused", [(0, "halves", 2 * ID, 2 * ID2)])
            if t == "down_proj.weight":
                self._need(name, shape == [HID, ID], "shape %s != [%d, %d]" % (shape, HID, ID))
                return Plan("dense-down", [(1, "tail", ID * isz, ID2 * isz)])
            raise SystemExit("ABORT unknown dense MLP tensor: %s %s %s" % (name, dtype, shape))
        m = re.match(r"^experts\.(\d+\.)?(gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)\.(weight|weight_scale|weight_scale_2|input_scale)$", t)
        if not m:
            raise SystemExit("ABORT unknown expert tensor: %s %s %s" % (name, dtype, shape))
        proj, kind = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}.get(m.group(2), m.group(2)), m.group(3)
        if kind in ("weight_scale_2", "input_scale"):
            self._need(name, len(shape) <= 1 or (len(shape) == 2 and shape[1] == 1), "scalar scale has shape %s" % shape)
            return Plan("unchanged-expert-scalar")
        lead = 0 if m.group(1) else 1   # fused experts carry a leading [E] axis
        if kind == "weight":
            self._need(name, dtype == "U8", "expert weight dtype %s, expected packed NVFP4 (U8)" % dtype)
            k_bytes, k2 = 2, 2          # 2 fp4 values per byte
        else:
            self._need(name, dtype == "F8_E4M3", "expert block scale dtype %s, expected F8_E4M3" % dtype)
            k_bytes, k2 = 16, 16        # 1 scale per 16 values
        s = shape[lead:]
        if proj in ("gate_proj", "up_proj"):
            self._need(name, s == [IM, HID // k_bytes], "shape %s != [%d, %d]" % (s, IM, HID // k_bytes))
            return Plan("expert-gu-" + kind, [(lead, "tail", IM, IM2)])
        if proj == "gate_up_proj":
            self._need(name, s == [2 * IM, HID // k_bytes], "shape %s != [%d, %d]" % (s, 2 * IM, HID // k_bytes))
            return Plan("expert-gu-fused-" + kind, [(lead, "halves", 2 * IM, 2 * IM2)])
        self._need(name, s == [HID, IM // k2], "shape %s != [%d, %d]" % (s, HID, IM // k2))
        return Plan("expert-down-" + kind, [(lead + 1, "tail", IM // k2, IM2 // k2)])


def new_logical_shape(dtype, shape, plan):
    sh = list(shape)
    for axis, mode, old, new in plan.pads:
        if axis == len(sh) - 1 and ITEMSIZE[dtype] > 1 and old % ITEMSIZE[dtype] == 0 and sh[axis] * ITEMSIZE[dtype] == old:
            sh[axis] = new // ITEMSIZE[dtype]
        else:
            assert sh[axis] == old, (shape, plan.pads)
            sh[axis] = new
    return sh


def apply_pads(arr, plan):
    for axis, mode, old, new in plan.pads:
        assert arr.shape[axis] == old, (arr.shape, plan.pads)
        if mode == "tail":
            w = [(0, 0)] * arr.ndim; w[axis] = (0, new - old)
            arr = np.pad(arr, w)
        else:  # halves: each half gets its own tail
            h, h2 = old // 2, new // 2
            a, b = np.split(arr, [h], axis=axis)
            w = [(0, 0)] * arr.ndim; w[axis] = (0, h2 - h)
            arr = np.concatenate([np.pad(a, w), np.pad(b, w)], axis=axis)
    return np.ascontiguousarray(arr)


def orig_region(arr_new, plan):
    """Return (view of the original region, list of padded-region views) from a padded array."""
    pads = []
    cur = arr_new
    for axis, mode, old, new in plan.pads:
        idx = [slice(None)] * cur.ndim
        if mode == "tail":
            idx[axis] = slice(old, new); pads.append(cur[tuple(idx)])
            idx[axis] = slice(0, old); cur = cur[tuple(idx)]
        else:
            h, h2 = old // 2, new // 2
            i1 = list(idx); i1[axis] = slice(h, h2); pads.append(cur[tuple(i1)])
            i2 = list(idx); i2[axis] = slice(h2 + h, new); pads.append(cur[tuple(i2)])
            ia = list(idx); ia[axis] = slice(0, h); ib = list(idx); ib[axis] = slice(h2, h2 + h)
            cur = np.concatenate([cur[tuple(ia)], cur[tuple(ib)]], axis=axis)
    return cur, pads


def tensor_bytes(mm, base, info):
    s, e = info["data_offsets"]
    return np.frombuffer(mm, dtype=np.uint8, count=e - s, offset=base + s)


def shards(src):
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    return idx, sorted(set(idx["weight_map"].values()))


def load_cfg(src):
    cfg = json.load(open(os.path.join(src, "config.json")))
    for k in ("num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size", "intermediate_size",
              "moe_intermediate_size"):
        if k not in cfg:
            raise SystemExit("ABORT config.json lacks %s" % k)
    return cfg


def plan_all(src, rules):
    idx, files = shards(src)
    out, counts = {}, {}
    for fn in files:
        h, meta, base = read_header(os.path.join(src, fn))
        for name, info in h.items():
            p = rules.plan(name, info["dtype"], info["shape"])
            out[name] = (fn, info, p)
            key = re.sub(r"\d+", "N", name) + " -> " + p.rule
            counts[key] = counts.get(key, 0) + 1
    missing = set(idx["weight_map"]) - set(out)
    if missing:
        raise SystemExit("ABORT index names not found in shards: %s" % sorted(missing)[:5])
    return out, counts, files


def tp_check(rules):
    H2, KV2, g = rules.H2, rules.KV2, rules.H2 // rules.KV2
    qpr, kpr = H2 // 3, KV2 // 3
    ok = True
    for r in range(3):
        qs, ks = range(qpr * r, qpr * (r + 1)), range(kpr * r, kpr * (r + 1))
        for q in qs:
            kv_global = q // g
            if kv_global not in ks:
                ok = False
            if q < rules.H and kv_global != q // (rules.H // rules.KV):
                ok = False
            if q >= rules.H and kv_global < rules.KV:
                ok = False
        print("TP_RANK %d query_heads %d-%d kv_heads %d-%d" % (r, qs[0], qs[-1], ks[0], ks[-1]))
    print("TP_GQA_MAPPING", "OK (original heads keep their KV head; new heads read only new KV heads)" if ok else "BROKEN")
    return ok


def write_all(src, dst, rules, plans, files, args, cfg):
    os.makedirs(dst, exist_ok=False)
    t0 = time.time(); total_new = 0; manifest = {"padder": "pad_beam_tp3_v%s" % VERSION, "args": vars(args), "shards": {}}
    for i, fn in enumerate(files, 1):
        sp = os.path.join(src, fn); h, meta, base = read_header(sp)
        names = sorted(h, key=lambda n: h[n]["data_offsets"][0])
        newh, off = {}, 0
        for n in names:
            info, p = h[n], plans[n][2]
            nsh = new_logical_shape(info["dtype"], info["shape"], p) if p.pads else info["shape"]
            nbytes = int(np.prod(storage_shape(info["dtype"], nsh))) if nsh else ITEMSIZE[info["dtype"]]
            newh[n] = {"dtype": info["dtype"], "shape": nsh, "data_offsets": [off, off + nbytes]}; off += nbytes
        hd = dict(newh)
        hd["__metadata__"] = dict(meta or {}, **{"format": (meta or {}).get("format", "pt"), "tp3_padder": VERSION})
        hb = json.dumps(hd, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        sha = hashlib.sha256()
        tmp = os.path.join(dst, fn + ".partial")
        mm = np.memmap(sp, dtype=np.uint8, mode="r")
        with open(tmp, "wb") as f:
            for chunk in (struct.pack("<Q", len(hb)), hb):
                f.write(chunk); sha.update(chunk)
            for n in names:
                info, p = h[n], plans[n][2]
                raw = tensor_bytes(mm, base, info)
                if p.pads:
                    raw = apply_pads(raw.reshape(storage_shape(info["dtype"], info["shape"])), p)
                b = raw.tobytes()
                assert len(b) == newh[n]["data_offsets"][1] - newh[n]["data_offsets"][0], n
                f.write(b); sha.update(b)
        del mm
        os.replace(tmp, os.path.join(dst, fn))
        sz = os.path.getsize(os.path.join(dst, fn)); total_new += sz - 8 - len(hb)
        manifest["shards"][fn] = {"bytes": sz, "sha256": sha.hexdigest()}
        el = time.time() - t0
        print("%s WROTE %2d/%d %s %.2f GB (%.0f s elapsed)" % (time.strftime("%H:%M:%S"), i, len(files), fn, sz / 1e9, el), flush=True)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    idx.setdefault("metadata", {})["total_size"] = total_new
    json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    newcfg = dict(cfg)
    newcfg.update(num_attention_heads=rules.H2, num_key_value_heads=rules.KV2, intermediate_size=rules.ID2,
                  moe_intermediate_size=rules.IM2)
    json.dump(newcfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors") and fn not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, os.path.join(dst, fn))
    manifest["config_changes"] = {"num_attention_heads": [rules.H, rules.H2], "num_key_value_heads": [rules.KV, rules.KV2],
                                  "intermediate_size": [rules.ID, rules.ID2], "moe_intermediate_size": [rules.IM, rules.IM2]}
    json.dump(manifest, open(os.path.join(dst, "PADDING_MANIFEST.json"), "w"), indent=2)
    print("WRITE_DONE shards=%d seconds=%d" % (len(files), time.time() - t0))


def verify(src, dst, rules, plans, files, every=1):
    t0 = time.time(); bad = 0; checked = 0
    dcfg = json.load(open(os.path.join(dst, "config.json")))
    for k, v in (("num_attention_heads", rules.H2), ("num_key_value_heads", rules.KV2), ("intermediate_size", rules.ID2),
                 ("moe_intermediate_size", rules.IM2)):
        if dcfg.get(k) != v:
            print("VERIFY_BAD config %s=%s want %s" % (k, dcfg.get(k), v)); bad += 1
    for i, fn in enumerate(files, 1):
        if (i - 1) % every:
            continue
        hs, _, bs = read_header(os.path.join(src, fn)); hd, _, bd = read_header(os.path.join(dst, fn))
        ms = np.memmap(os.path.join(src, fn), dtype=np.uint8, mode="r"); md = np.memmap(os.path.join(dst, fn), dtype=np.uint8, mode="r")
        if set(hs) != set(hd):
            print("VERIFY_BAD %s tensor sets differ" % fn); bad += 1; continue
        for n, info in hs.items():
            p = plans[n][2]; di = hd[n]
            want = new_logical_shape(info["dtype"], info["shape"], p) if p.pads else info["shape"]
            if di["dtype"] != info["dtype"] or di["shape"] != want:
                print("VERIFY_BAD %s shape %s want %s" % (n, di["shape"], want)); bad += 1; continue
            a = tensor_bytes(ms, bs, info).reshape(storage_shape(info["dtype"], info["shape"]))
            b = tensor_bytes(md, bd, di).reshape(storage_shape(di["dtype"], di["shape"]))
            if p.pads:
                o, pads = orig_region(b, p)
                if not np.array_equal(o, a) or any(np.any(x) for x in pads):
                    print("VERIFY_BAD %s original bytes changed or padding not zero" % n); bad += 1
            elif not np.array_equal(a, b):
                print("VERIFY_BAD %s unchanged tensor differs" % n); bad += 1
            checked += 1
        del ms, md
        print("%s VERIFIED %2d/%d %s bad_so_far=%d" % (time.strftime("%H:%M:%S"), i, len(files), fn, bad), flush=True)
    print("VERIFY tensors=%d bad=%d seconds=%d" % (checked, bad, time.time() - t0))
    print("VERIFY_RESULT=" + ("PASS" if bad == 0 else "FAIL"))
    return bad == 0


# ---------------- numeric test (CPU) ----------------
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def f8e4m3_table():
    t = np.zeros(256)
    for v in range(256):
        s, e, m = v >> 7, (v >> 3) & 15, v & 7
        if e == 15 and m == 7:
            x = np.nan
        elif e == 0:
            x = m / 8 * 2 ** -6
        else:
            x = (1 + m / 8) * 2 ** (e - 7)
        t[v] = -x if s else x
    return t


F8 = f8e4m3_table()


def decode(raw, dtype, shape):
    if dtype == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float64).reshape(shape)
    if dtype == "F32":
        return raw.view(np.float32).astype(np.float64).reshape(shape)
    if dtype == "F8_E4M3":
        return F8[raw].reshape(shape)
    raise ValueError(dtype)


def nvfp4_dequant(wraw, wshape, sraw, sshape, g):
    lo, hi = E2M1[wraw & 15], E2M1[wraw >> 4]
    vals = np.stack([lo, hi], axis=-1).reshape(wshape[0], wshape[1] * 2)   # low nibble = first value (ModelOpt order)
    sc = F8[sraw].reshape(sshape)
    return vals * np.repeat(sc, 16, axis=1) * g


class Reader:
    def __init__(self, d):
        self.d = d; self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]; self.cache = {}

    def get(self, name):
        fn = self.idx[name]
        if fn not in self.cache:
            h, _, b = read_header(os.path.join(self.d, fn)); self.cache[fn] = (h, b, np.memmap(os.path.join(self.d, fn), dtype=np.uint8, mode="r"))
        h, b, mm = self.cache[fn]; info = h[name]
        return tensor_bytes(mm, b, info), info["dtype"], info["shape"]

    def f(self, name):
        raw, dt, sh = self.get(name)
        return decode(raw, dt, sh)

    def has(self, name):
        return name in self.idx


def attn_forward(R, L, H, KV, D, x):
    p = "model.layers.%d.self_attn." % L
    q = x @ R.f(p + "q_proj.weight").T; k = x @ R.f(p + "k_proj.weight").T; v = x @ R.f(p + "v_proj.weight").T
    T = x.shape[0]; q = q[:, :H * D].reshape(T, H, D); k = k.reshape(T, KV, D); v = v.reshape(T, KV, D)
    g = H // KV; out = np.zeros((T, H, D))
    for h in range(H):
        s = q[:, h] @ k[:, h // g].T / np.sqrt(D); s -= s.max(1, keepdims=True); w = np.exp(s); w /= w.sum(1, keepdims=True)
        out[:, h] = w @ v[:, h // g]
    if R.has(p + "gate_proj.weight"):
        gt = (x @ R.f(p + "gate_proj.weight").T).reshape(T, H, D) + R.f(p + "gate_bias").reshape(1, H, 1)
        out = out / (1 + np.exp(-gt))
    return out.reshape(T, H * D) @ R.f(p + "o_proj.weight").T


def expert_forward(R, L, e, x):
    p = "model.layers.%d.mlp.experts.%d." % (L, e)
    def W(proj):
        wr, _, ws = R.get(p + proj + ".weight"); sr, _, ss = R.get(p + proj + ".weight_scale")
        g = float(R.f(p + proj + ".weight_scale_2").reshape(-1)[0])
        return nvfp4_dequant(wr.reshape(ws), ws, sr, ss, g)
    a = x @ W("gate_proj").T; b = x @ W("up_proj").T
    return (a / (1 + np.exp(-a)) * b) @ W("down_proj").T


def dense_forward(R, L, x):
    p = "model.layers.%d.mlp." % L
    a = x @ R.f(p + "gate_proj.weight").T; b = x @ R.f(p + "up_proj.weight").T
    return (a / (1 + np.exp(-a)) * b) @ R.f(p + "down_proj.weight").T


def attn_forward_tp(R, L, H, KV, D, x, tp=3):
    """vLLM-style split: q/k/v column-parallel by heads, o_proj row-parallel, partial outputs summed."""
    p = "model.layers.%d.self_attn." % L
    Wq, Wk, Wv, Wo = (R.f(p + n + "_proj.weight") for n in "qkvo")
    hq, hk, g, T = H // tp, KV // tp, H // KV, x.shape[0]
    y = 0
    for r in range(tp):
        q = (x @ Wq[r * hq * D:(r + 1) * hq * D].T).reshape(T, hq, D)
        k = (x @ Wk[r * hk * D:(r + 1) * hk * D].T).reshape(T, hk, D); v = (x @ Wv[r * hk * D:(r + 1) * hk * D].T).reshape(T, hk, D)
        out = np.zeros((T, hq, D))
        for h in range(hq):
            s = q[:, h] @ k[:, h // g].T / np.sqrt(D); s -= s.max(1, keepdims=True); w = np.exp(s); w /= w.sum(1, keepdims=True)
            out[:, h] = w @ v[:, h // g]
        if R.has(p + "gate_proj.weight"):
            Wg = R.f(p + "gate_proj.weight")[r * hq * D:(r + 1) * hq * D]; bg = R.f(p + "gate_bias")[r * hq:(r + 1) * hq]
            out = out / (1 + np.exp(-((x @ Wg.T).reshape(T, hq, D) + bg.reshape(1, hq, 1))))
        y = y + out.reshape(T, hq * D) @ Wo[:, r * hq * D:(r + 1) * hq * D].T
    return y


def mlp_forward_tp(Wg, Wu, Wd, x, tp=3):
    n = Wg.shape[0] // tp; y = 0
    for r in range(tp):
        a = x @ Wg[r * n:(r + 1) * n].T; b = x @ Wu[r * n:(r + 1) * n].T
        y = y + (a / (1 + np.exp(-a)) * b) @ Wd[:, r * n:(r + 1) * n].T
    return y


def expert_weights(R, L, e):
    p = "model.layers.%d.mlp.experts.%d." % (L, e); out = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        wr, _, ws = R.get(p + proj + ".weight"); sr, _, ss = R.get(p + proj + ".weight_scale")
        g = float(R.f(p + proj + ".weight_scale_2").reshape(-1)[0])
        out.append(nvfp4_dequant(wr.reshape(ws), ws, sr, ss, g))
    return out


def numeric_test(src, dst, rules, dense_layer, moe_layer, expert):
    rng = np.random.default_rng(0); x = rng.standard_normal((5, rules.HID))
    S, P = Reader(src), Reader(dst); ok = True
    def cmp(tag, a, b):
        nonlocal ok
        d = float(np.max(np.abs(a - b))); rel = d / (float(np.max(np.abs(a))) + 1e-30)
        good = np.isfinite(a).all() and rel < 1e-9
        ok &= bool(good)
        print("NUMERIC %-28s max_abs_diff=%.3e rel=%.3e %s" % (tag, d, rel, "OK" if good else "BAD"))
    if S.has("model.layers.%d.self_attn.q_proj.weight" % dense_layer):
        cmp("attention layer %d" % dense_layer, attn_forward(S, dense_layer, rules.H, rules.KV, rules.D, x),
            attn_forward(P, dense_layer, rules.H2, rules.KV2, rules.D, x))
    if S.has("model.layers.%d.mlp.gate_proj.weight" % dense_layer):
        cmp("dense MLP layer %d" % dense_layer, dense_forward(S, dense_layer, x), dense_forward(P, dense_layer, x))
    if S.has("model.layers.%d.mlp.experts.%d.gate_proj.weight" % (moe_layer, expert)):
        cmp("expert %d layer %d" % (expert, moe_layer), expert_forward(S, moe_layer, expert, x), expert_forward(P, moe_layer, expert, x))
    # the same layers computed the way vLLM shards them at TP=3 (padded copy) vs the unsharded original
    if S.has("model.layers.%d.self_attn.q_proj.weight" % dense_layer):
        cmp("attention L%d TP=3 split" % dense_layer, attn_forward(S, dense_layer, rules.H, rules.KV, rules.D, x),
            attn_forward_tp(P, dense_layer, rules.H2, rules.KV2, rules.D, x))
    if S.has("model.layers.%d.mlp.gate_proj.weight" % dense_layer):
        pp = "model.layers.%d.mlp." % dense_layer
        cmp("dense MLP L%d TP=3 split" % dense_layer, dense_forward(S, dense_layer, x),
            mlp_forward_tp(P.f(pp + "gate_proj.weight"), P.f(pp + "up_proj.weight"), P.f(pp + "down_proj.weight"), x))
    if S.has("model.layers.%d.mlp.experts.%d.gate_proj.weight" % (moe_layer, expert)):
        cmp("expert %d L%d TP=3 split" % (expert, moe_layer), expert_forward(S, moe_layer, expert, x),
            mlp_forward_tp(*expert_weights(P, moe_layer, expert), x))
    print("NUMERIC_RESULT=" + ("PASS" if ok else "FAIL"))
    print("NUMERIC_NOTE attention test includes the output gate sigmoid(gate+bias) but omits q/k norms and RoPE/SSMax (shared per head_dim, unaffected by padding); expert test assumes ModelOpt low-nibble-first packing "
          "(identical in both copies, so equality holds either way; the byte-level --verify is the primary proof)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst")
    ap.add_argument("--heads-to", type=int, default=54); ap.add_argument("--kv-to", type=int, default=9)
    ap.add_argument("--dense-to", type=int, default=20736); ap.add_argument("--moe-to", type=int, required=True)
    ap.add_argument("--qgate-layout", choices=["none", "head_major", "halves"], default="none")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-every", type=int, default=1)
    ap.add_argument("--numeric-test", action="store_true")
    ap.add_argument("--numeric-layers", default="0,3,0")
    a = ap.parse_args()
    cfg = load_cfg(a.src); rules = Rules(cfg, a)
    print("PADDER v%s src=%s heads %d->%d kv %d->%d dense %d->%d moe %d->%d qgate=%s" % (
        VERSION, a.src, rules.H, rules.H2, rules.KV, rules.KV2, rules.ID, rules.ID2, rules.IM, rules.IM2, a.qgate_layout))
    plans, counts, files = plan_all(a.src, rules)
    for k in sorted(counts):
        print("PLAN %6d x %s" % (counts[k], k))
    if not tp_check(rules):
        raise SystemExit("ABORT GQA mapping would break under TP=3")
    print("PLAN_OK tensors=%d shards=%d" % (len(plans), len(files)))
    if a.dry_run:
        return
    if not a.dst:
        raise SystemExit("ABORT --dst required")
    if a.verify:
        ok = verify(a.src, a.dst, rules, plans, files, a.verify_every)
        sys.exit(0 if ok else 1)
    if a.numeric_test:
        dl, ml, ex = (int(v) for v in a.numeric_layers.split(","))
        sys.exit(0 if numeric_test(a.src, a.dst, rules, dl, ml, ex) else 1)
    write_all(a.src, a.dst, rules, plans, files, a, cfg)


if __name__ == "__main__":
    main()
