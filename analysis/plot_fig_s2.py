#!/usr/bin/env python3
"""
Deep-SemP Figure S2: contig length vs. reference coverage (C. elegans).

Two-panel figure for the supplement:
  (A) top-N longest contigs, rank-ordered length curve
  (B) per-transcript unioned reference-coverage density (KDE), with the
      full-length (>=90%) threshold marked.

No in-panel titles (journal convention: describe panels in the caption);
panels are labelled (A) and (B). Console output reports N50, medians, and the
>=90% coverage counts to cite in the caption.

Run on the cluster where the FASTAs / BLAST TSVs live.
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
# BLAST TSVs (14-col: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen)
BLASTS = {
    "Baseline":  "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/baseline_blast.tsv",
    "Random":    "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/random_control_blast.tsv",
    "Deep-SemP": "/data3/projects/2025_Assembly/eyh/c_elegans/blast_results/student_blast.tsv",
}
ORDER   = ["Baseline", "Random", "Deep-SemP"]
COLORS  = {"Baseline": "#4C72B0", "Random": "#999999", "Deep-SemP": "#C44E52"}

TOP_N        = 200      # panel A: number of longest contigs to show
FL_THRESHOLD = 0.90     # panel B: full-length definition line at 90% reference coverage
SHOW_HIST    = False    # light histogram behind the KDE in panel B
SHOW_KDE     = True
KDE_BW       = None      # bandwidth: None = Scott's rule; a float scales it

OUT = "fig_S2_length_vs_coverage"

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
    """Gaussian-KDE density on grid; bounds=(lo,hi) reflects mass at finite edges."""
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

def panel_label(ax, text):
    """Bold (A)/(B) label at the top-left, just outside the axes."""
    ax.text(-0.10, 1.04, text, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")

# ---------------------------------------------------------------- load
lengths = {name: contig_lengths(p) for name, p in FASTAS.items()}
covs    = {name: union_ref_coverage(t) for name, t in BLASTS.items()}

# ---------------------------------------------------------------- figure (2 panels)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- Panel A: top-N rank curve ---
for name in ORDER:
    top = np.sort(lengths[name])[::-1][:TOP_N]
    axA.plot(np.arange(1, len(top) + 1), top, color=COLORS[name], lw=1.8, label=name)
axA.set_xlabel(f"Contig rank (longest {TOP_N})")
axA.set_ylabel("Contig length (bp)")
axA.legend(frameon=False)
panel_label(axA, "A")

# --- Panel B: reference-coverage density (Deep-SemP drawn last = on top) ---
grid = np.linspace(0, 1, 512)
bins = np.linspace(0, 1, 51)
for name in ORDER:                      # ORDER puts Deep-SemP last, so it sits on top
    c = covs[name]
    if SHOW_HIST:
        axB.hist(c, bins=bins, histtype="step", linewidth=1.0, alpha=0.35,
                 color=COLORS[name], density=True,
                 label=None if SHOW_KDE else name)
    if SHOW_KDE:
        y = kde_curve(c, grid, bounds=(0.0, 1.0), bw=KDE_BW)   # reflect at 0 and 1
        axB.plot(grid, y, color=COLORS[name], lw=1.8, label=name)
axB.axvline(FL_THRESHOLD, color="k", ls="--", lw=1)
axB.text(FL_THRESHOLD, axB.get_ylim()[1] * 0.92, "  full-length\n  (≥90%)",
         fontsize=8, va="top")
axB.set_xlim(0, 1)
axB.set_xlabel("Reference coverage (unioned per transcript)")
axB.set_ylabel("Density")
axB.legend(frameon=False, loc="upper left")
panel_label(axB, "B")

fig.tight_layout()
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print(f"wrote {OUT}.{{pdf,png}}")

# ---------------------------------------------------------------- stats for the caption
print("\n-- Panel A (contig length) --")
for name in ORDER:
    L = lengths[name]
    top = np.sort(L)[::-1][:TOP_N]
    print(f"  {name:10s} n={len(L):6d}  N50={int(n50(L)):5d}  median(top{TOP_N})={int(np.median(top)):5d}")
print("\n-- Panel B (reference coverage) --")
for name in ORDER:
    c = covs[name]
    print(f"  {name:10s} n={len(c):6d}  mean={c.mean()*100:.1f}%  >=90% (full-length)={int((c>=FL_THRESHOLD).sum())}")
