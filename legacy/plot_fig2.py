import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── Arial font ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

species    = ["C. elegans (training)", "C. latens", "C. remanei",
              "C. briggsae", "H. sapiens (distant)"]
divergence = [0, 20, 20, 20, 700]
entropies  = [0.7992, 0.8141, 0.8240, 0.8480, 0.9127]
colors     = ["#1D9E75"] * 4 + ["#D85A30"]

slope, intercept, r, p, _ = stats.linregress(divergence, entropies)

fig, ax = plt.subplots(figsize=(5.5, 4.2))

# ── Caenorhabditis shaded region ──
cae_ent = entropies[:-1]
ax.axhspan(min(cae_ent) - 0.004, max(cae_ent) + 0.004,
           alpha=0.12, color='#1D9E75', zorder=1,
           label='Caenorhabditis range')

# ── Regression line ──
x_fit = np.linspace(-10, 760, 300)
ax.plot(x_fit, intercept + slope * x_fit, '--',
        color='#888780', lw=1.2, alpha=0.7, zorder=3,
        label=f'Linear fit ($R^2$={r**2:.3f}, $p$={p:.4f})')

# ── Scatter ──
ax.scatter(divergence, entropies, c=colors, s=110,
           zorder=5, edgecolors='white', linewidth=1.3)

# ── Annotations — elegans moved above point ──
offsets = [(8, 2), (8, -5), (8, 4), (8, 6), (-80, 8)]
for sp, d, e, off in zip(species, divergence, entropies, offsets):
    ax.annotate(sp, (d, e), textcoords="offset points",
                xytext=off, fontsize=8.5, style='italic')

# ── Axes ──
ax.set_xlabel('Divergence from C. elegans (Mya)', fontsize=10)
ax.set_ylabel('Normalized Routing Entropy', fontsize=10)
ax.set_xlim(-30, 780)
ax.set_ylim(0.775, 0.935)
ax.grid(alpha=0.25, linewidth=0.6)
ax.tick_params(labelsize=9)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.7)

plt.tight_layout()
plt.savefig('entropy_vs_phylogeny.pdf', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('entropy_vs_phylogeny.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
print("Saved.")