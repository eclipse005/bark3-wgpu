"""Convert a compressed Bark 3.0 `.bark` release into this crate's low-bit layout.

Bark 3.0 / 3.0 Plus / 3.0 Clean ship as a single `model.bark` container: the
encoder linears are per-row fp16 levels plus a 4-bit code per weight, the rest is
5/6-bit integers times a per-row scale, all ANS-coded. Running that in f16 would
need 1.2 GB of VRAM; this keeps the codes and expands them in the GEMM load
stage instead (~380 MiB).

The output is a normal model directory:

    model.safetensors   <name> = 4-bit codes, <name>.levels = per-row levels,
                        every other tensor as f16
    lowbit.json         {name: level count}
    bark3.json          release name + container SHA-256 (also read by `bark3`)
    config.json, train-time processor/tokenizer files, copied from a reference

Requirements: numpy, constriction, safetensors, and the official `bark_loader.py`
next to this file (it ships with the model repository).

usage:
    python tools/encode_lowbit.py model.bark ../models/Underdog-Bark-3.0-Plus \
        [--reference ../models/Underdog-Bark-3.0-Pro]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bark_loader  # noqa: E402  (official decoder, shipped with the weights)

KPAD = 4
REFERENCE_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(raw_path: Path):
    """Yield (name, codes/levels or None, weights) with the container's offsets."""
    raw = raw_path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    head = json.loads(raw[8:8 + n])
    if head["format"] not in ("bark-levels-v2", "bark-levels-v3"):
        raise SystemExit(f"{raw_path}: unknown container format {head['format']!r}")
    p = 8 + n
    for e in head["index"]:
        shape = e["shape"]
        if e["k"] == "levels2":
            k, rows, cols = e["K"], shape[0], int(np.prod(shape[1:]))
            lev = np.frombuffer(
                bark_loader._unz(raw[p:p + e["lev"]], e.get("zl")), np.float16
            ).reshape(rows, k)
            p += e["lev"]
            meta = raw[p:p + e["meta"]]
            p += e["meta"]
            nb = (cols * 4 + 7) // 8
            bits = np.unpackbits(np.frombuffer(meta[:nb], np.uint8))[:cols * 4].reshape(cols, 4)
            cls = np.packbits(
                np.concatenate([np.zeros((cols, 4), np.uint8), bits], 1), axis=1
            )[:, 0]
            counts = np.frombuffer(meta[nb:], np.uint32).reshape(
                bark_loader.NC, 2 * k + 1
            ).astype(np.float64)
            words = np.frombuffer(raw[p:p + e["b"]], np.uint32).copy()
            p += e["b"]
            codes = bark_loader._decode_codes(cls, counts, words, rows, cols)
            yield e["n"], (k, rows, cols, lev, codes), bark_loader._levels_to_weights(lev, codes).reshape(shape)
        elif e["k"].startswith("ansint"):
            bits = int(e["k"][6:])
            scale = np.frombuffer(
                bark_loader._unz(raw[p:p + e["s"]], e.get("zs")), np.float16
            ).astype(np.float32)
            p += e["s"]
            counts = np.frombuffer(raw[p:p + e["meta"]], np.uint32).astype(np.float64)
            p += e["meta"]
            words = np.frombuffer(raw[p:p + e["b"]], np.uint32).copy()
            p += e["b"]
            import constriction

            model = constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)
            sym = constriction.stream.stack.AnsCoder(words).decode(model, int(np.prod(shape)))
            q = (sym - (2 ** (bits - 1) - 1)).astype(np.float32)
            yield e["n"], None, (q.reshape(len(scale), -1) * scale[:, None]).reshape(shape)
        elif e["k"] == "raw":
            dtype = {"torch.int64": np.int64, "torch.int32": np.int32}.get(e["dtype"], np.float16)
            value = np.frombuffer(
                bark_loader._unz(raw[p:p + e["b"]], e.get("zr")), dtype
            ).copy().reshape(shape)
            p += e["b"]
            yield e["n"], None, value
        else:
            raise SystemExit(f"{e['n']}: unsupported entry kind {e['k']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container", type=Path, help="model.bark")
    ap.add_argument("out_dir", type=Path, help="model directory to create")
    ap.add_argument("--reference", type=Path, default=None,
                    help="directory to copy config/tokenizer files from")
    args = ap.parse_args()

    out, levels = {}, {}
    codes_bytes = levels_bytes = 0
    for name, lowbit, weights in walk(args.container):
        if lowbit is None:
            out[name] = weights.astype(np.float16) if weights.dtype.kind == "f" else weights
            continue
        k, rows, cols, lev, codes = lowbit
        packed = np.zeros((rows, (cols + 7) // 8), np.uint32)
        for j in range(8):
            band = codes[:, j::8]
            if band.shape[1]:
                packed[:, :band.shape[1]] |= band.astype(np.uint32) << (4 * j)
        pad = np.zeros((rows, KPAD), np.float16)
        pad[:, :k] = lev
        # The packing must reproduce the loader's values exactly.
        table = np.concatenate([np.zeros((rows, 1), np.float32), pad.astype(np.float32)], 1)
        mag = np.where(codes == 0, 0, np.where(codes <= k, codes, codes - k)).astype(np.int64)
        got = np.where(codes > k, -np.take_along_axis(table, mag, 1), np.take_along_axis(table, mag, 1))
        assert np.array_equal(got, weights.reshape(rows, cols)), name
        out[name] = packed
        out[name + ".levels"] = pad
        levels[name] = k
        codes_bytes += packed.nbytes
        levels_bytes += pad.nbytes

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / "model.safetensors"
    save_file(out, str(dst))
    (args.out_dir / "lowbit.json").write_text(json.dumps(levels, indent=1), encoding="utf-8")
    hist = {}
    for k in levels.values():
        hist[str(k)] = hist.get(str(k), 0) + 1
    info = {
        "source": args.out_dir.name,
        "layout": "low-bit",
        "container": args.container.name,
        "container_sha256": sha256(args.container),
        "levels_k": dict(sorted(hist.items())),
    }
    (args.out_dir / "bark3.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if args.reference:
        for name in REFERENCE_FILES:
            src = args.reference / name
            if src.is_file():
                (args.out_dir / name).write_bytes(src.read_bytes())
    print(f"wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
    print(f"  {len(levels)} low-bit tensors: codes {codes_bytes / 1e6:.0f} MB, levels {levels_bytes / 1e6:.0f} MB")
    print(f"  {len(out) - 2 * len(levels)} other tensors as f16")


if __name__ == "__main__":
    main()
