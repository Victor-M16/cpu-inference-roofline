"""Turn results/*.json into figures and a markdown summary.

    python harness/report.py            # reads results/*.json, writes docs/*.svg + results/SUMMARY.md
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Validated categorical slots 1-3 (blue, orange, aqua); neutrals for roofs and grid.
DECODE = "#2a78d6"
PREFILL = "#eb6834"
SWEEP = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
QUANT_ORDER = ["F16", "Q8_0", "Q4_K_M", "Q4_0"]
# Q4_0 and Q4_K_M land almost on top of each other; nudge their labels apart.
LABEL_OFFSET = {"Q4_K_M": (7, -11), "Q4_0": (-6, 7)}

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "svg.fonttype": "none",
})


def quant(model: dict) -> str:
    for q in QUANT_ORDER:
        if f"-{q}." in model["file"] or model["file"].endswith(f"{q}.gguf"):
            return q
    return model["file"].rsplit("-", 1)[-1].removesuffix(".gguf")


def runs(model: dict, phase: str, threads: int | None = None, depth: int = 0) -> list[dict]:
    return [
        r for r in model["runs"]
        if r["phase"] == phase and r["depth"] == depth and (threads is None or r["threads"] == threads)
    ]


def max_threads(res: dict) -> int:
    return max(r["threads"] for m in res["models"] for r in m["runs"])


def sorted_models(res: dict) -> list[dict]:
    return sorted(res["models"], key=lambda m: QUANT_ORDER.index(quant(m)) if quant(m) in QUANT_ORDER else 99)


# ------------------------------------------------------------------ figures


def save(fig, out: Path) -> None:
    fig.savefig(out)  # SVG for the README
    fig.savefig(out.with_suffix(".png"), dpi=160)  # PNG for slides and emails
    plt.close(fig)


def roofline(res: dict, out: Path) -> None:
    p = res["probe"]
    bw, peak = p["roof_bw_gbs"], p["roof_gflops"]
    ridge = peak / bw
    T = max_threads(res)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = [0.05 * 1.15**i for i in range(80)]
    xs = [x for x in xs if x <= 2000]
    ax.plot(xs, [min(bw * x, peak) for x in xs], color=INK, lw=2, label="Roofline (measured roofs)")
    ax.plot([s["flops_per_byte"] for s in p["sweep"]], [s["gflops"] for s in p["sweep"]],
            "o", color=SWEEP, ms=6, mec="white", mew=1.5, label="Probe: intensity sweep")
    ax.axvline(ridge, color=INK2, lw=0.8, ls=":")
    ax.annotate(f"ridge ≈ {ridge:.0f} FLOP/B", (ridge, peak * 0.03), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.annotate(f"slope = {bw:.1f} GB/s", (0.3, bw * 0.3), xytext=(8, -12), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.annotate(f"{peak:.0f} GFLOP/s", (900, peak), xytext=(-60, 4), textcoords="offset points",
                color=INK2, fontsize=8)

    for phase, color, marker in (("decode", DECODE, "o"), ("prefill", PREFILL, "s")):
        first = True
        for m in sorted_models(res):
            for r in runs(m, phase, T):
                ax.plot(r["intensity"], r["achieved_gflops"], marker, color=color, ms=8, mec="white", mew=1.5,
                        label=f"llama.cpp {phase} (batch {'1' if phase == 'decode' else r['tokens']})" if first else None)
                first = False
                ax.annotate(quant(m), (r["intensity"], r["achieved_gflops"]),
                            xytext=LABEL_OFFSET.get(quant(m), (7, -3)), textcoords="offset points",
                            fontsize=7.5, color=INK, ha="right" if quant(m) == "Q4_0" else "left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 2000)
    ax.set_ylim(max(0.5, bw * 0.03), peak * 3)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte from DRAM)")
    ax.set_ylabel("Attained GFLOP/s")
    ax.set_title(f"{res['machine'].get('cpu', res['label'])}, {T} threads", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, out)


def scaling(res: dict, out: Path) -> None:
    """Indexed to 1 thread: decode follows bandwidth, prefill follows compute."""
    sc = {s["threads"]: s for s in res["probe"]["scaling"]}
    m = next((x for x in res["models"] if quant(x) == "Q4_0"), res["models"][0])
    dec = {r["threads"]: r["tok_s"] for r in runs(m, "decode")}
    pre = {r["threads"]: r["tok_s"] for r in runs(m, "prefill")}
    ts = sorted(set(dec) & set(sc))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ts, [sc[t]["read_gbs"] / sc[ts[0]]["read_gbs"] for t in ts], ls=(0, (4, 3)), color=DECODE, lw=1.2,
            label="Probe: DRAM read bandwidth")
    ax.plot(ts, [dec[t] / dec[ts[0]] for t in ts], marker="o", ls="-", color=DECODE, lw=2, ms=8, mec="white", mew=1.5,
            label=f"llama.cpp decode tok/s ({quant(m)})")
    ax.plot(ts, [sc[t]["peak_gflops"] / sc[ts[0]]["peak_gflops"] for t in ts], ls=(0, (4, 3)), color=PREFILL, lw=1.2,
            label="Probe: peak FMA throughput")
    ax.plot(ts, [pre[t] / pre[ts[0]] for t in ts], marker="s", ls="-", color=PREFILL, lw=2, ms=8, mec="white", mew=1.5,
            label=f"llama.cpp prefill tok/s ({quant(m)})")
    ax.set_xticks(ts)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speed-up over 1 thread")
    ax.set_title("Which roof does each phase follow as cores are added?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, out)


def bytes_vs_speed(all_res: list[dict], out: Path) -> None:
    """Decode tok/s against bytes/token, one panel per machine, with the 1/bytes bandwidth curve."""
    n = len(all_res)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0), squeeze=False)
    for ax, res in zip(axes[0], all_res):
        T = max_threads(res)
        bw = res["probe"]["roof_bw_gbs"]
        pts = []
        for m in sorted_models(res):
            for d in sorted({r["depth"] for r in m["runs"]}):
                for r in runs(m, "decode", T, d):
                    pts.append((r["bytes_per_token"] / 1e6, r["tok_s"], quant(m), d))
        xmax = max(p[0] for p in pts) * 1.15
        xs = [xmax * i / 200 for i in range(8, 201)]
        ax.plot(xs, [bw * 1e3 / x for x in xs], color=INK, lw=1.5,
                label=f"Bandwidth bound: {bw:.1f} GB/s ÷ bytes/token")
        for depth, mk in ((0, "o"), (2048, "D")):
            sel = [p for p in pts if p[3] == depth]
            if not sel:
                continue
            ax.plot([p[0] for p in sel], [p[1] for p in sel], mk, color=DECODE, ms=8, mec="white", mew=1.5,
                    alpha=1.0 if depth == 0 else 0.6, label=f"Measured decode, KV depth {depth}", ls="none")
            if depth == 0:
                for x, y, q, _ in sel:
                    ax.annotate(q, (x, y), xytext=LABEL_OFFSET.get(q, (7, 4)), textcoords="offset points",
                                fontsize=7.5, color=INK, ha="right" if q == "Q4_0" else "left")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, max(p[1] for p in pts) * 1.35)
        ax.set_xlabel("MB moved per decoded token (weights + KV cache)")
        ax.set_ylabel("Decode tokens/s")
        ax.set_title(f"{res['label']} ({T} threads)", fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, out)


# ------------------------------------------------------------------ summary


def summary(all_res: list[dict]) -> str:
    lines = ["# Results summary", "", "Generated by `harness/report.py` from `results/*.json`.", ""]
    lines += [
        "| Machine | CPU | Threads | BW roof (GB/s) | Compute roof (GFLOP/s) | Ridge (FLOP/B) |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for res in all_res:
        p = res["probe"]
        lines.append(
            f"| {res['label']} | {res['machine'].get('cpu', '')} | {max_threads(res)} | {p['roof_bw_gbs']:.1f} | "
            f"{p['roof_gflops']:.0f} | {p['roof_gflops'] / p['roof_bw_gbs']:.1f} |"
        )
    lines += [
        "",
        "Decode = batch 1, KV depth 0, all threads. *BW used* is bytes/token × tokens/s as a share of the "
        "probe's read bandwidth; *compute used* is the same for FLOPs.",
        "",
        "| Machine | Format | Model MB | Decode tok/s | MB/token | FLOP/byte | BW used | Compute used "
        "| Decode tok/s @ KV 2048 | Prefill tok/s | Prefill FLOP/byte | Prefill compute used |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for res in all_res:
        T = max_threads(res)
        for m in sorted_models(res):
            d0 = runs(m, "decode", T, 0)
            dk = runs(m, "decode", T, 2048)
            pf = runs(m, "prefill", T, 0)
            if not d0:
                continue
            d0 = d0[0]
            deep = f"{dk[0]['tok_s']:.1f}" if dk else "–"
            pre = (
                f"{pf[0]['tok_s']:.1f} | {pf[0]['intensity']:.0f} | {100 * pf[0]['frac_compute_roof']:.0f}%"
                if pf else "– | – | –"
            )
            lines.append(
                f"| {res['label']} | {quant(m)} | {m['file_bytes'] / 1e6:.0f} | {d0['tok_s']:.1f} | "
                f"{d0['bytes_per_token'] / 1e6:.0f} | {d0['intensity']:.2f} | {100 * d0['frac_bw_roof']:.0f}% | "
                f"{100 * d0['frac_compute_roof']:.0f}% | {deep} | {pre} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    files = sorted((ROOT / "results").glob("*.json"))
    if not files:
        raise SystemExit("no results/*.json yet; run harness/bench.py first")
    all_res = [json.loads(f.read_text()) for f in files]
    # Re-derive bytes/FLOPs from the raw llama-bench rows so accounting fixes apply to old runs.
    sys.path.insert(0, str(ROOT / "harness"))
    from bench import analyse

    for res in all_res:
        for m in res["models"]:
            if "raw" in m:
                m["runs"] = analyse(m["raw"], m, res["probe"])
    DOCS.mkdir(exist_ok=True)
    for res in all_res:
        roofline(res, DOCS / f"roofline-{res['label']}.svg")
        scaling(res, DOCS / f"scaling-{res['label']}.svg")
    bytes_vs_speed(all_res, DOCS / "decode-bytes-vs-speed.svg")
    (ROOT / "results" / "SUMMARY.md").write_text(summary(all_res))
    print("\n".join(str(p.relative_to(ROOT)) for p in sorted(DOCS.glob("*.svg"))))
    print("results/SUMMARY.md")


if __name__ == "__main__":
    main()
