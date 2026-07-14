#!/usr/bin/env python3
"""
Deep-SemP figure script: contig-length and full-length recovery distributions
for C. elegans (baseline / random / Deep-SemP).

Plot 1 (N50 defense): contig-length distributions, three assemblies overlaid.
Plot 2 (top-N):       longest TOP_N contigs, rank-ordered length curve.
Plot 3 (full-length): per-reference union-coverage density, from BLAST TSVs.

Histograms carry a smoothed KDE overlay (Gaussian kernel). SciPy is used when
present; a NumPy fallback is used otherwise, so the only hard deps stay
matplotlib + numpy.

Run on the cluster where the FASTAs / TSVs live.
Deps: matplotlib, numpy  (scipy optional, for the KDE)
"""

import gzip, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")            # file output, no display needed
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ---------------------------------------------------------------- config
FASTAS = {
    "Baseline":  "/data3/projects/2025_Assembly/eyh/c_elegans/assemblies/baseline_trinity/baseline_filtered_final_coding_only.fasta",
    "Random":    "/data3/projects/2025_Assembly/eyh/c_elegans/postprocess/random_control/05_final_coding.fasta",
    "Deep-SemP": "/data3/projects/2025_Assembly/eyh/c_elegans/postprocess/student_stratB/05_final_coding.fasta",
}
# BLAST TSVs (14-col fmt: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen)
BLASTS = {
    "Baseline":  "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/baseline_blast.tsv",
    "Random":    "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/random_control_blast.tsv",
    "Deep-SemP": "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/student_blast.tsv",
}
ORDER   = ["Baseline", "Random", "Deep-SemP"]
COLORS  = {"Baseline": "#4C72B0", "Random": "#999999", "Deep-SemP": "#C44E52"}

# Plot 1 behaviour: length distribution of ALL contigs, x-axis focused for readability.
XMAX_BP   = 6000     # x-axis view limit (data is NOT clipped; longer contigs just fall off-view)
LOG_X     = False    # set True for log10 length axis instead of linear
BIN_WIDTH = 100      # bp per histogram bin (linear mode)

# Plot 2 behaviour
TOP_N     = 200      # number of longest contigs to show in the rank curve

# Plot 3 behaviour
FL_THRESHOLD = 0.90  # full-length definition line at 90% reference coverage

# KDE / histogram overlay behaviour (applies to Plots 1 and 3)
SHOW_HIST = True     # light histogram behind the KDE line
SHOW_KDE  = True     # smoothed KDE curve
KDE_BW    = None     # bandwidth: None = Scott's rule; a float scales it (smaller = wigglier)

# ---------------------------------------------------------------- helpers
def open_maybe_gz(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)

def contig_lengths(path):
    """Return list of sequence lengths from a FASTA (handles multi-line records)."""
    lengths, cur = [], 0
    with open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur: lengths.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
        if cur: lengths.append(cur)
    return np.array(lengths)

def union_ref_coverage(tsv):
    """Per-subject unioned reference coverage fraction from a 14-col BLAST TSV."""
    ivs, slen = defaultdict(list), {}
    with open(tsv) as fh:
        for r in csv.reader(fh, delimiter="\t"):
            if len(r) != 14:
                continue
            s = r[1]; ss, se, sl = int(r[8]), int(r[9]), int(r[13])
            a, b = min(ss, se), max(ss, se)
            ivs[s].append((a, b)); slen[s] = sl
    def merged(iv):
        iv = sorted(iv); tot = 0; cs, ce = iv[0]
        for a, b in iv[1:]:
            if a <= ce + 1: ce = max(ce, b)
            else: tot += ce - cs + 1; cs, ce = a, b
        return tot + ce - cs + 1
    return np.array([min(merged(v) / slen[s], 1.0) for s, v in ivs.items()])

def n50(L):
    s = np.sort(L)[::-1]; c = np.cumsum(s)
    return s[np.searchsorted(c, c[-1] / 2)]

# ---- KDE (scipy if available, else a small chunked NumPy Gaussian KDE) ----
def _kde_eval(data, grid, bw=None):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    grid = np.asarray(grid, float)
    if data.size < 2 or np.allclose(data, data[0]):
        return np.zeros_like(grid)
    if HAVE_SCIPY:
        return gaussian_kde(data, bw_method=bw)(grid)
    # NumPy fallback: Scott's-rule bandwidth, evaluated in grid chunks to bound memory
    n = data.size
    std = data.std(ddof=1) or 1.0
    h = (bw if bw else 1.0) * std * n ** (-1.0 / 5.0)
    if h <= 0:
        h = 1.0
    out = np.empty_like(grid)
    norm = 1.0 / (n * h * np.sqrt(2 * np.pi))
    for i in range(0, grid.size, 256):
        g = grid[i:i + 256]
        u = (g[:, None] - data[None, :]) / h
        out[i:i + 256] = np.exp(-0.5 * u * u).sum(axis=1) * norm
    return out

def kde_curve(data, grid, bounds=None, bw=None, max_pts=20000, seed=0):
    """
    Gaussian-KDE density on `grid`.
    `bounds=(lo, hi)` reflects mass back across finite edges (either may be None),
    which keeps a bounded quantity like coverage in [0, 1] from leaking past the edge.
    Large samples are subsampled to `max_pts` (shape is unaffected) for speed/memory.
    """
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size > max_pts:
        data = np.random.default_rng(seed).choice(data, max_pts, replace=False)
    y = _kde_eval(data, grid, bw)
    if bounds is not None:
        lo, hi = bounds
        if lo is not None:
            y = y + _kde_eval(data, 2 * lo - np.asarray(grid, float), bw)
        if hi is not None:
            y = y + _kde_eval(data, 2 * hi - np.asarray(grid, float), bw)
    return y

# ---------------------------------------------------------------- load
lengths = {name: contig_lengths(p) for name, p in FASTAS.items()}

# quick top-N console check (kept for inspection)
for name in ORDER:
    top = np.sort(lengths[name])[::-1][:TOP_N]
    print(f"{name:10s} top{TOP_N}: min={top.min()} median={int(np.median(top))} "
          f"max={top.max()} mean={int(top.mean())}")

# ---------------------------------------------------------------- Plot 1: length distribution
fig, ax = plt.subplots(figsize=(7, 4.2))

if LOG_X:
    allL = np.concatenate(list(lengths.values()))
    lo = max(allL.min(), 1)
    bins = np.logspace(np.log10(lo), np.log10(allL.max()), 60)
    grid = np.logspace(np.log10(lo), np.log10(allL.max()), 512)
    for name in ORDER:
        L = lengths[name]
        if SHOW_HIST:
            ax.hist(L, bins=bins, histtype="step", linewidth=1.0, alpha=0.35,
                    color=COLORS[name], density=True,
                    label=None if SHOW_KDE else name)
        if SHOW_KDE:
            # KDE computed in log10 space, drawn against the log x-axis.
            # NOTE: in log mode the histogram (per-bp density) and KDE (per-log-unit
            # density) use different y conventions, so for a strictly normalized
            # comparison view one at a time (toggle SHOW_HIST / SHOW_KDE).
            y = kde_curve(np.log10(L), np.log10(grid), bw=KDE_BW)
            ax.plot(grid, y, color=COLORS[name], lw=1.8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Contig length (bp, log scale)")
else:
    allL = np.concatenate(list(lengths.values()))
    bins = np.arange(300, allL.max() + BIN_WIDTH, BIN_WIDTH)
    grid = np.linspace(300, XMAX_BP, 512)
    for name in ORDER:
        L = lengths[name]
        if SHOW_HIST:
            ax.hist(L, bins=bins, histtype="step", linewidth=1.0, alpha=0.35,
                    color=COLORS[name], density=True,
                    label=None if SHOW_KDE else name)
        if SHOW_KDE:
            y = kde_curve(L, grid, bounds=(0, None), bw=KDE_BW)
            ax.plot(grid, y, color=COLORS[name], lw=1.8, label=name)
    ax.set_xlim(300, XMAX_BP)
    ax.set_xlabel(f"Contig length (bp; view limited to {XMAX_BP})")

# N50 markers
for name in ORDER:
    ax.axvline(n50(lengths[name]), color=COLORS[name], ls=":", lw=1, alpha=0.7)

ax.set_ylabel("Density")
ax.set_title("C. elegans contig length distribution (N50 marked, dotted)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fig_contig_length_distribution.pdf")
fig.savefig("fig_contig_length_distribution.png", dpi=200)
print("wrote fig_contig_length_distribution.{pdf,png}")
for name in ORDER:
    L = lengths[name]
    print(f"  {name:10s} n={len(L):6d}  N50={n50(L):5d}  median={int(np.median(L)):5d}  max={L.max()}")

# ---------------------------------------------------------------- Plot 2: top-N rank curve
fig2, ax2 = plt.subplots(figsize=(7, 4.2))
for name in ORDER:
    top = np.sort(lengths[name])[::-1][:TOP_N]
    ax2.plot(np.arange(1, len(top) + 1), top, color=COLORS[name], lw=1.8, label=name)
ax2.set_xlabel(f"Contig rank (longest {TOP_N})")
ax2.set_ylabel("Contig length (bp)")
ax2.set_title(f"C. elegans top {TOP_N} contigs by length")
ax2.legend(frameon=False)
fig2.tight_layout()
fig2.savefig("fig_top_contig_lengths.pdf")
fig2.savefig("fig_top_contig_lengths.png", dpi=200)
print("wrote fig_top_contig_lengths.{pdf,png}")

# ---------------------------------------------------------------- Plot 3: reference coverage
fig3, ax3 = plt.subplots(figsize=(7, 4.2))
covs = {name: union_ref_coverage(t) for name, t in BLASTS.items()}
bins = np.linspace(0, 1, 51)
grid = np.linspace(0, 1, 512)
for name in ORDER:
    c = covs[name]
    if SHOW_HIST:
        ax3.hist(c, bins=bins, histtype="step", linewidth=1.0, alpha=0.35,
                 color=COLORS[name], density=True,
                 label=None if SHOW_KDE else name)
    if SHOW_KDE:
        y = kde_curve(c, grid, bounds=(0.0, 1.0), bw=KDE_BW)   # reflect at 0 and 1
        ax3.plot(grid, y, color=COLORS[name], lw=1.8, label=name)
ax3.axvline(FL_THRESHOLD, color="k", ls="--", lw=1)
ax3.text(FL_THRESHOLD, ax3.get_ylim()[1]*0.92, "  full-length\n  (\u226590%)",
         fontsize=8, va="top")
ax3.set_xlim(0, 1)
ax3.set_xlabel("Reference coverage (unioned per transcript)")
ax3.set_ylabel("Density")
ax3.set_title("C. elegans per-transcript reference coverage")
ax3.legend(frameon=False)
fig3.tight_layout()
fig3.savefig("fig_reference_coverage_pdf.pdf")
fig3.savefig("fig_reference_coverage_pdf.png", dpi=200)
print("wrote fig_reference_coverage_pdf.{pdf,png}")
for name in ORDER:
    c = covs[name]
    print(f"  {name:10s} n={len(c):6d}  mean={c.mean()*100:.1f}%  >=90%={int((c>=0.9).sum())}")