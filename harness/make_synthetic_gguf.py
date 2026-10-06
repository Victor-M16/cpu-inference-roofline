"""Write a Llama-architecture GGUF with real model shapes and random weights.

Why this exists: dense transformer inference speed depends on tensor shapes and
data types, not on the weight values. A shape-faithful random model therefore
gives the same tokens/s and bytes/token as the real checkpoint, and it can be
generated on machines that cannot reach a model hub (bank networks, CI, air-gapped
boxes). Output quality is meaningless; only use these files for timing.

If you can download real GGUFs, pass them to bench.py directly instead.

    python harness/make_synthetic_gguf.py --shape tinyllama-1.1b --out models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover
    sys.exit("pip install gguf  (or add llama.cpp/gguf-py to PYTHONPATH)")

# Shapes copied from the public config.json of each model.
SHAPES = {
    "tinyllama-1.1b": dict(dim=2048, layers=22, heads=32, kv_heads=4, ffn=5632, vocab=32000),
    "llama-3.2-1b": dict(dim=2048, layers=16, heads=32, kv_heads=8, ffn=8192, vocab=128256),
    "llama-3.2-3b": dict(dim=3072, layers=28, heads=24, kv_heads=8, ffn=8192, vocab=128256),
    "llama-2-7b": dict(dim=4096, layers=32, heads=32, kv_heads=32, ffn=11008, vocab=32000),
}


def fake_vocab(n: int) -> tuple[list[str], list[float], list[int]]:
    """SentencePiece-style vocab: unk/bos/eos, 256 byte tokens, then filler pieces."""
    T = gguf.TokenType
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [T.UNKNOWN, T.CONTROL, T.CONTROL] + [T.BYTE] * 256
    tokens += [f"▁t{i}" for i in range(n - len(tokens))]
    types += [T.NORMAL] * (n - len(types))
    scores = [0.0] * 3 + [0.0] * 256 + [-float(i) for i in range(n - 259)]
    return tokens, scores, [int(t) for t in types]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="tinyllama-1.1b")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    s = SHAPES[args.shape]
    dim, heads, kvh = s["dim"], s["heads"], s["kv_heads"]
    head_dim = dim // heads
    kv_dim = kvh * head_dim

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.shape}-synthetic-F16.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name(f"{args.shape} (synthetic weights, timing only)")
    w.add_context_length(args.ctx)
    w.add_embedding_length(dim)
    w.add_block_count(s["layers"])
    w.add_feed_forward_length(s["ffn"])
    w.add_head_count(heads)
    w.add_head_count_kv(kvh)
    w.add_rope_dimension_count(head_dim)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(s["vocab"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    tokens, scores, types = fake_vocab(s["vocab"])
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)

    rng = np.random.default_rng(0)

    def mat(rows: int, cols: int) -> np.ndarray:
        # numpy shape is (out_features, in_features), matching the HF converter
        return (rng.standard_normal((rows, cols), dtype=np.float32) * 0.02).astype(np.float16)

    def norm() -> np.ndarray:
        return np.ones(dim, dtype=np.float32)

    w.add_tensor("token_embd.weight", mat(s["vocab"], dim))
    for i in range(s["layers"]):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm())
        w.add_tensor(p + "attn_q.weight", mat(dim, dim))
        w.add_tensor(p + "attn_k.weight", mat(kv_dim, dim))
        w.add_tensor(p + "attn_v.weight", mat(kv_dim, dim))
        w.add_tensor(p + "attn_output.weight", mat(dim, dim))
        w.add_tensor(p + "ffn_norm.weight", norm())
        w.add_tensor(p + "ffn_gate.weight", mat(s["ffn"], dim))
        w.add_tensor(p + "ffn_up.weight", mat(s["ffn"], dim))
        w.add_tensor(p + "ffn_down.weight", mat(dim, s["ffn"]))
    w.add_tensor("output_norm.weight", norm())
    w.add_tensor("output.weight", mat(s["vocab"], dim))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(path)


if __name__ == "__main__":
    main()
