import matplotlib.pyplot as plt
import numpy as np

species   = ["C. elegans\n(training)", "C. latens", "C. remanei",
             "C. briggsae", "H. sapiens\n(distant)"]
entropies = [0.7992, 0.8141, 0.8240, 0.8480, 0.9127]
colors    = ["#1D9E75", "#1D9E75", "#1D9E75", "#1D9E75", "#D85A30"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(species, entropies, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='Fully uniform (random)')
ax.axhline(np.mean(entropies[:4]), color='#1D9E75', ls=':', lw=1.5,
           label=f'Caenorhabditis mean ({np.mean(entropies[:4]):.3f})')

# Annotate bars
for bar, val in zip(bars, entropies):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Normalized Routing Entropy', fontsize=12)
ax.set_title('Read Routing Distribution Entropy by Species\n'
             'Higher = more uniform = less biological signal', fontsize=12)
ax.set_ylim(0.7, 1.05)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('routing_entropy_by_species.png', dpi=150, bbox_inches='tight')
print("Saved: routing_entropy_by_species.png")