import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Approximate divergence times from C. elegans (Mya)
# remanei and latens split <5 Mya from each other,
# but both ~20 Mya from elegans
species = ["C. elegans\n(training)", "C. latens", "C. remanei",
           "C. briggsae", "H. sapiens\n(distant)"]
divergence = [0, 20, 20, 20, 700]  # Mya from C. elegans
entropies  = [0.7992, 0.8141, 0.8240, 0.8480, 0.9127]
colors     = ["#1D9E75", "#1D9E75", "#1D9E75", "#1D9E75", "#D85A30"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter plot with regression
ax = axes[0]
ax.scatter(divergence, entropies, c=colors, s=120, zorder=5, edgecolors='white', linewidth=1.5)
for i, (sp, d, e) in enumerate(zip(species, divergence, entropies)):
    ax.annotate(sp, (d, e), textcoords="offset points",
                xytext=(8, -5), fontsize=9)

# Regression line
slope, intercept, r, p, _ = stats.linregress(divergence, entropies)
x_line = np.linspace(0, 750, 100)
ax.plot(x_line, intercept + slope * x_line, 'k--', lw=1.2, alpha=0.5,
        label=f'Linear fit (R²={r**2:.3f}, p={p:.4f})')
ax.set_xlabel('Divergence from C. elegans (Mya)', fontsize=11)
ax.set_ylabel('Normalized Routing Entropy', fontsize=11)
ax.set_title('Routing Entropy vs Phylogenetic Distance', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Right: zoomed in on Caenorhabditis only
ax = axes[1]
sp_nema = species[:-1]
div_nema = divergence[:-1]
ent_nema = entropies[:-1]
col_nema = colors[:-1]

ax.scatter(div_nema, ent_nema, c=col_nema, s=150, zorder=5,
           edgecolors='white', linewidth=1.5)
for sp, d, e in zip(sp_nema, div_nema, ent_nema):
    ax.annotate(sp, (d, e), textcoords="offset points",
                xytext=(5, 5), fontsize=9)

ax.axhspan(min(ent_nema)-0.01, max(ent_nema)+0.01,
           alpha=0.1, color='#1D9E75', label='Caenorhabditis range')
ax.set_xlabel('Divergence from C. elegans (Mya)', fontsize=11)
ax.set_ylabel('Normalized Routing Entropy', fontsize=11)
ax.set_title('Caenorhabditis species (zoomed)', fontsize=12)
ax.set_xlim(-2, 25)
ax.set_ylim(0.78, 0.87)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('Deep-SemP Routing Signal vs Phylogenetic Distance\n'
             'Higher entropy = model less certain = near-random routing',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('entropy_vs_phylogeny.png', dpi=150, bbox_inches='tight')
print("Saved: entropy_vs_phylogeny.png")
print(f"\nLinear regression:")
print(f"  Slope : {slope:.6f} entropy/Mya")
print(f"  R²    : {r**2:.4f}")
print(f"  p     : {p:.6f}")