"""Generate benchmark graphs from the CSV files written by `hospital_sim --benchmark`.

Usage:  python benchmarks/plot_results.py [results_dir]
Output: <results_dir>/graphs/*.png

Only measured values from the CSVs are plotted - nothing is synthesised.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "graphs")
os.makedirs(OUT, exist_ok=True)

# Categorical palette (fixed order, validated reference instance - light mode).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(r, k):
    return float(r[k])


kernel = read("kernel_benchmark.csv")
scen = read("scenario_benchmark.csv")
made = []


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)


def series(rows, exp, variant, key, xkey="patients", extra=None):
    pts = [(num(r, xkey), num(r, key)) for r in rows
           if r["experiment"] == exp and r["variant"] == variant and (extra is None or extra(r))]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def label_end(ax, x, y, text, color):
    if x:
        ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)


TIMING = "Kernel = CUDA events (data resident on GPU). E2E = wall clock incl. upload, per-step event download, final download."

if kernel:
    sw = [r for r in kernel if r["experiment"] == "size_sweep"]
    steps = sw[0]["steps"] if sw else "?"

    # 1) patient count vs execution time
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = [("CPU single-thread", "end_to_end_ms", "CPU 1 thread"),
             ("CPU multi-thread (OpenMP)", "end_to_end_ms", "CPU multi-thread"),
             ("CUDA naive (AoS split)", "end_to_end_ms", "CUDA naive (E2E)"),
             ("CUDA optimized (SoA fused)", "end_to_end_ms", "CUDA optimized (E2E)"),
             ("CUDA optimized (SoA fused)", "kernel_ms", "CUDA optimized (kernel only)")]
    for i, (v, k, lab) in enumerate(lines):
        x, y = series(kernel, "size_sweep", v, k)
        ax.plot(x, y, marker="o", color=SLOT[i], label=lab, linestyle="--" if k == "kernel_ms" else "-")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Patients (N)"); ax.set_ylabel(f"Time for {steps} update steps (ms, log)")
    ax.set_title("Patient count vs execution time")
    ax.legend(loc="upper left")
    save(fig, "01_patients_vs_time.png", TIMING)

    # 2) patient count vs speedup
    fig, ax = plt.subplots(figsize=(8, 5))
    sp = [("CUDA optimized (SoA fused)", "speedup_kernel_vs_cpu1", "CUDA optimized, kernel only"),
          ("CUDA optimized (SoA fused)", "speedup_e2e_vs_cpu1", "CUDA optimized, end-to-end"),
          ("CUDA naive (AoS split)", "speedup_e2e_vs_cpu1", "CUDA naive, end-to-end"),
          ("CPU multi-thread (OpenMP)", "speedup_e2e_vs_cpu1", "CPU multi-thread")]
    for i, (v, k, lab) in enumerate(sp):
        x, y = series(kernel, "size_sweep", v, k)
        ax.plot(x, y, marker="o", color=SLOT[i], label=lab)
        label_end(ax, x, y, f"{y[-1]:.1f}x" if y else "", SLOT[i])
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
    ax.text(ax.get_xlim()[0] if False else 110, 1.05, "break-even (1x)", fontsize=8, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Patients (N)"); ax.set_ylabel("Speedup vs CPU single-thread (log)")
    ax.set_title("Patient count vs GPU speedup")
    ax.legend(loc="upper left")
    save(fig, "02_patients_vs_speedup.png", "Speedup = CPU single-thread time / implementation time (same N, same 200 steps). Kernel = CUDA events; E2E = wall clock incl. transfers.")

    # 3) AoS vs SoA (kernel time per step vs N)
    fig, ax = plt.subplots(figsize=(8, 5))
    lay = [("CUDA naive (AoS split)", "AoS, 4 split kernels"), ("CUDA AoS fused", "AoS, fused kernel"),
           ("CUDA SoA split", "SoA, 4 split kernels"), ("CUDA optimized (SoA fused)", "SoA, fused kernel")]
    for i, (v, lab) in enumerate(lay):
        x, y = series(kernel, "size_sweep", v, "kernel_ms")
        y = [t / float(steps) * 1000.0 for t in y]
        ax.plot(x, y, marker="o", color=SLOT[i], label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Patients (N)"); ax.set_ylabel("Kernel time per step (us, log)")
    ax.set_title("AoS vs SoA, split vs fused kernels")
    ax.legend(loc="upper left")
    save(fig, "03_aos_vs_soa.png", "Kernel-only time (CUDA events), block size 256.")

    # 4) block size vs execution time
    bs = [r for r in kernel if r["experiment"] == "block_size" and r["is_gpu"] == "1"]
    if bs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, v in enumerate(["CUDA naive (AoS split)", "CUDA AoS fused", "CUDA optimized (SoA fused)"]):
            x, y = series(kernel, "block_size", v, "kernel_ms", xkey="block_size")
            ax.plot(x, y, marker="o", color=SLOT[i], label=v)
            if y:
                j = min(range(len(y)), key=lambda q: y[q])
                ax.annotate(f"best {int(x[j])}", (x[j], y[j]), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512, 1024]); ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
        ax.set_yscale("log")
        ax.set_xlabel("Threads per block"); ax.set_ylabel(f"Kernel time, {steps} steps (ms, log)")
        ax.set_title(f"Block size vs execution time (N = {int(num(bs[0], 'patients')):,})")
        ax.legend(loc="upper right")
        save(fig, "04_block_size.png", "Kernel-only time (CUDA events).")

    # 5) CPU vs naive vs optimized at selected sizes (grouped bars)
    sizes = sorted({int(num(r, "patients")) for r in sw})
    pick = [s for s in sizes if s in (10000, 100000, 1000000)] or sizes[-3:]
    impl = [("CPU single-thread", "end_to_end_ms", "CPU 1 thread"),
            ("CPU multi-thread (OpenMP)", "end_to_end_ms", "CPU multi-thread"),
            ("CUDA naive (AoS split)", "end_to_end_ms", "CUDA naive (E2E)"),
            ("CUDA optimized (SoA fused)", "end_to_end_ms", "CUDA optimized (E2E)")]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.2
    for i, (v, k, lab) in enumerate(impl):
        vals = []
        for s in pick:
            m = [num(r, k) for r in sw if r["variant"] == v and int(num(r, "patients")) == s]
            vals.append(m[0] if m else 0)
        xs = [j + (i - 1.5) * w for j in range(len(pick))]
        ax.bar(xs, vals, width=w * 0.92, color=SLOT[i], label=lab)
    ax.set_xticks(range(len(pick))); ax.set_xticklabels([f"N = {s:,}" for s in pick])
    ax.set_yscale("log"); ax.set_ylabel(f"Time for {steps} steps (ms, log)")
    ax.set_title("CPU vs naive CUDA vs optimized CUDA")
    ax.legend(loc="upper left")
    save(fig, "05_cpu_naive_optimized.png", TIMING)

    # 6) ordering + active list
    od = [r for r in kernel if r["experiment"] == "ordering" and r["is_gpu"] == "1"]
    al = [r for r in kernel if r["experiment"] == "active_list" and r["is_gpu"] == "1"]
    if od or al:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
        for i, v in enumerate(["CUDA optimized (SoA fused)", "CUDA AoS fused"]):
            vals = [num(r, "kernel_ms") for o in ("random", "zone-sorted") for r in od
                    if r["variant"] == v and r["ordering"] == o]
            a1.bar([j + (i - 0.5) * 0.38 for j in range(len(vals))], vals, width=0.36, color=SLOT[i], label=v)
        a1.set_xticks([0, 1]); a1.set_xticklabels(["random order", "zone/state-sorted"])
        a1.set_ylabel("Kernel time (ms)"); a1.set_title("Zone-based ordering")
        a1.set_ylim(0, a1.get_ylim()[1] * 1.3); a1.legend(loc="upper center", ncol=2, fontsize=8)
        for i, v in enumerate(["CUDA optimized (SoA fused)", "CUDA SoA fused active-list"]):
            x, y = series(kernel, "active_list", v, "kernel_ms", xkey="inactive_fraction")
            a2.plot([100 * q for q in x], y, marker="o", color=SLOT[i + 2],
                    label="full range" if i == 0 else "active-index list")
        a2.set_xlabel("Inactive patients (%)"); a2.set_ylabel("Kernel time (ms)")
        a2.set_title("Skipping inactive patients"); a2.legend(loc="upper right")
        save(fig, "06_zone_and_active.png", "Kernel-only time (CUDA events), largest N.")

if scen:
    # 7) full hybrid scenario: wall time per configuration
    ns = sorted({int(r["patients"]) for r in scen})
    cfgs = []
    for r in scen:
        if r["backend"] not in cfgs and "unavailable" not in r["backend"]:
            cfgs.append(r["backend"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.8 / max(1, len(cfgs))
    for i, c in enumerate(cfgs):
        vals = []
        for n in ns:
            m = [num(r, "wall_ms") for r in scen if r["backend"] == c and int(r["patients"]) == n]
            vals.append(m[0] if m else 0)
        ax.bar([j + (i - (len(cfgs) - 1) / 2) * w for j in range(len(ns))], vals, width=w * 0.92,
               color=SLOT[i % len(SLOT)], label=c)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"N = {n:,}" for n in ns])
    ax.set_yscale("log"); ax.set_ylabel("Wall time for the whole run (ms, log)")
    ax.set_title("Full hybrid simulation (mass-casualty, scaled): end-to-end wall time")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, "07_scenario_wall_time.png",
         "Wall time includes CPU controller (queues, allocation), transfers and synchronisation.")

print("\n".join(made) if made else "no CSV data found in " + RES)
