"""Run the roofline probe and llama-bench on this machine and write one results JSON.

The JSON holds everything report.py needs: the measured roofs (bandwidth and
compute, per thread count), the intensity sweep, and for every model/thread/depth
combination the measured tokens/s plus the bytes and FLOPs each token costs,
computed from the GGUF's actual tensor sizes.

    python harness/bench.py --label "xeon-cascadelake-4vcpu" \
        --llama-bench ../llama.cpp/build/bin/llama-bench \
        --models models/*.gguf
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import gguf

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ machine info


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return ""


def machine_info() -> dict:
    info: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Linux":
        lscpu = dict(
            (k.strip(), v.strip())
            for k, v in (line.split(":", 1) for line in sh(["lscpu"]).splitlines() if ":" in line)
        )
        info["cpu"] = lscpu.get("Model name", platform.processor())
        for key in ("L1d cache", "L2 cache", "L3 cache", "Thread(s) per core", "Core(s) per socket", "Hypervisor vendor"):
            if key in lscpu:
                info[key.lower().replace("(s)", "s").replace(" ", "_")] = lscpu[key]
        flags = set(lscpu.get("Flags", "").split())
        info["simd"] = sorted(f for f in flags if re.fullmatch(r"(sse4_2|avx|avx2|fma|f16c|avx512\w*|amx\w*|asimd\w*|sve\w*)", f))
        mem = Path("/proc/meminfo").read_text().split("\n")[0].split()
        info["ram_gib"] = round(int(mem[1]) / 2**20, 1)
    elif platform.system() == "Darwin":
        info["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        info["ram_gib"] = round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
    else:
        info["cpu"] = platform.processor()
    return info


# ------------------------------------------------------------------ probe


def build_probe() -> Path:
    src = ROOT / "probe" / "roofline_probe.c"
    exe = ROOT / "probe" / "roofline_probe"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        cc = os.environ.get("CC", "cc")
        flags = ["-O3", "-march=native", "-fopenmp"]
        if platform.system() == "Darwin":  # Apple clang: brew install libomp
            omp = sh(["brew", "--prefix", "libomp"]).strip() or "/opt/homebrew/opt/libomp"
            flags = ["-O3", "-mcpu=native", "-Xpreprocessor", "-fopenmp", f"-I{omp}/include", f"-L{omp}/lib", "-lomp"]
        subprocess.run([cc, *flags, "-o", str(exe), str(src)], check=True)
    return exe


def run_probe(threads: int, mib: int) -> dict:
    exe = build_probe()
    print(f"[probe] measuring roofs with {threads} threads, {mib} MiB arrays", file=sys.stderr)
    out = subprocess.run([str(exe), "--threads", str(threads), "--mib", str(mib)], capture_output=True, text=True, check=True)
    probe = json.loads(out.stdout)
    top = probe["scaling"][-1]
    probe["roof_bw_gbs"] = top["read_gbs"]
    # The dedicated FMA kernel and the top of the sweep both measure the compute
    # ceiling; take whichever the compiler/CPU combination did better on.
    probe["roof_gflops"] = max(top["peak_gflops"], max(p["gflops"] for p in probe["sweep"]))
    return probe


# ------------------------------------------------------------------ model accounting


def model_costs(path: Path) -> dict:
    """Bytes that must cross the memory bus per decoded token, from real tensor sizes."""
    r = gguf.GGUFReader(str(path))
    arch = r.fields["general.architecture"].contents()

    def kv(key: str):
        f = r.fields.get(f"{arch}.{key}")
        return f.contents() if f else None

    total = embd = 0
    params = embd_params = 0
    for t in r.tensors:
        total += int(t.n_bytes)
        params += int(t.n_elements)
        if t.name == "token_embd.weight":
            embd, embd_params = int(t.n_bytes), int(t.n_elements)
    tied = not any(t.name == "output.weight" for t in r.tensors)
    if tied:  # the embedding matrix is also the LM head, so it is fully read each token
        embd = embd_params = 0

    dim = kv("embedding_length")
    heads = kv("attention.head_count")
    kv_heads = kv("attention.head_count_kv") or heads
    layers = kv("block_count")
    head_dim = dim // heads
    return {
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "params": params,
        # token_embd is a row gather in decode (one row per token), not a stream.
        "weight_bytes_per_token": total - embd,
        "matmul_params": params - embd_params,
        "layers": layers,
        "dim": dim,
        "kv_dim": kv_heads * head_dim,
    }


def per_token(costs: dict, depth: float, batch: int) -> tuple[float, float]:
    """(bytes, flops) per token. batch = tokens sharing one pass over the weights."""
    kv_bytes = 2 * costs["layers"] * costs["kv_dim"] * depth * 2  # K and V, f16 cache
    attn_flops = 4 * costs["layers"] * costs["dim"] * depth  # QK^T and AV
    # Weights and the cached K/V are each read once per pass and shared by the batch.
    bytes_ = (costs["weight_bytes_per_token"] + kv_bytes) / batch
    flops = 2 * costs["matmul_params"] + attn_flops
    return bytes_, flops


# ------------------------------------------------------------------ llama-bench


def run_llama_bench(exe: str, model: Path, threads: list[int], depths: list[int], pp: int, tg: int, reps: int) -> list[dict]:
    cmd = [
        exe, "-m", str(model), "-p", str(pp), "-n", str(tg),
        "-t", ",".join(map(str, threads)), "-d", ",".join(map(str, depths)),
        "-r", str(reps), "-o", "json",
    ]
    print("[bench] " + " ".join(cmd), file=sys.stderr)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr[-2000:])
        raise SystemExit(f"llama-bench failed on {model}")
    return json.loads(out.stdout)


RAW_KEYS = ("n_threads", "n_depth", "n_prompt", "n_gen", "n_batch", "n_ubatch", "model_type",
            "avg_ts", "stddev_ts", "build_commit", "cpu_info", "flash_attn", "type_k", "type_v")


def analyse(rows: list[dict], costs: dict, probe: dict) -> list[dict]:
    bw_by_t = {s["threads"]: s["read_gbs"] for s in probe["scaling"]}
    fl_by_t = {s["threads"]: s["peak_gflops"] for s in probe["scaling"]}
    fl_by_t[probe["threads"]] = probe["roof_gflops"]  # sweep ran at full thread count
    results = []
    for row in rows:
        t = row["n_threads"]
        depth = row.get("n_depth", 0)
        if row["n_gen"] > 0:
            phase, n = "decode", row["n_gen"]
            batch = 1
            avg_depth = depth + n / 2
        else:
            phase, n = "prefill", row["n_prompt"]
            batch = min(n, row.get("n_ubatch", 512))
            avg_depth = depth + n / 2
        b, f = per_token(costs, avg_depth, batch)
        tps = row["avg_ts"]
        results.append({
            "phase": phase,
            "threads": t,
            "depth": depth,
            "tokens": n,
            "type": row.get("model_type", ""),
            "tok_s": tps,
            "tok_s_std": row.get("stddev_ts", 0.0),
            "bytes_per_token": b,
            "flops_per_token": f,
            "intensity": f / b,
            "achieved_gbs": b * tps / 1e9,
            "achieved_gflops": f * tps / 1e9,
            # fraction of the roof that applies at this thread count
            "frac_bw_roof": b * tps / 1e9 / bw_by_t.get(t, probe["roof_bw_gbs"]),
            "frac_compute_roof": f * tps / 1e9 / fl_by_t.get(t, probe["roof_gflops"]),
            "roofline_bound_tok_s": min(
                bw_by_t.get(t, probe["roof_bw_gbs"]) * 1e9 / b,
                probe["roof_gflops"] * 1e9 / f,
            ),
        })
    return results


def thread_list(n: int) -> list[int]:
    out, t = [], 1
    while t < n:
        out.append(t)
        t *= 2
    return out + [n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="short machine id, e.g. 'i5-4200u-2013-laptop'")
    ap.add_argument("--note", default="", help="free text: laptop/desktop/VM, power mode, year bought")
    ap.add_argument("--llama-bench", default=os.environ.get("LLAMA_BENCH", "llama-bench"))
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--threads", type=int, nargs="+", help="default: 1,2,4,..,all logical CPUs")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 2048], help="KV-cache depth before each test")
    ap.add_argument("--pp", type=int, default=512, help="prefill (prompt) tokens")
    ap.add_argument("--tg", type=int, default=64, help="decode (generated) tokens")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--probe-mib", type=int, default=256, help="per-array size; keep well above L3")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    threads = args.threads or thread_list(os.cpu_count() or 1)
    probe = run_probe(max(threads), args.probe_mib)
    print(f"[probe] bandwidth roof {probe['roof_bw_gbs']:.1f} GB/s, compute roof {probe['roof_gflops']:.0f} GFLOP/s", file=sys.stderr)

    models = []
    for m in args.models:
        costs = model_costs(m)
        rows = run_llama_bench(args.llama_bench, m, threads, args.depths, args.pp, args.tg, args.reps)
        costs["raw"] = [{k: r.get(k) for k in RAW_KEYS} for r in rows]
        costs["runs"] = analyse(costs["raw"], costs, probe)
        costs["llama_cpp_build"] = rows[0].get("build_commit", "") if rows else ""
        models.append(costs)
        for r in costs["runs"]:
            if r["threads"] == max(threads):
                print(
                    f"  {m.name:42s} {r['phase']:7s} d={r['depth']:<5d} {r['tok_s']:8.1f} tok/s  "
                    f"{r['bytes_per_token'] / 1e6:8.1f} MB/tok  {r['achieved_gbs']:6.1f} GB/s "
                    f"({100 * r['frac_bw_roof']:3.0f}% of BW roof)  AI={r['intensity']:.1f}",
                    file=sys.stderr,
                )

    result = {
        "label": args.label,
        "note": args.note,
        "date": dt.date.today().isoformat(),
        "machine": machine_info(),
        "probe": probe,
        "models": models,
    }
    out = args.out or ROOT / "results" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
