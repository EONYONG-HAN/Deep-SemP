#!/usr/bin/env python3
"""
Replot Deep-SemP Figure S1 (clustering metrics vs K) from cached sweep results.

The AgglomerativeClustering (Ward) sweep is expensive (~10 min per K); the
silhouette / Calinski-Harabasz values below were computed once by cluster_eval.py
and stored here so the figure can be regenerated instantly. No in-panel title
(journal convention: the description lives in the caption).

Deps: matplotlib  (numpy not required)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- cached sweep results (C. elegans ESM-2 embeddings, Ward linkage) ----
K_VALUES          = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
silhouette_scores = [0.0564, 0.0463, 0.0404, 0.0414, 0.0453,
                     0.0466, 0.0488, 0.0485, 0.0474, 0.0483]
ch_scores         = [2862.76, 1703.45, 1261.97, 1022.69, 866.98,
                     757.40, 675.56, 612.77, 562.69, 521.56]

# ---- plot (identical styling to cluster_eval.py, title removed) ----
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Silhouette Score', color=color1, fontsize=12)
ax1.plot(K_VALUES, silhouette_scores, marker='o', color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Selected K=50')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Calinski-Harabasz Score', color=color2, fontsize=12)
ax2.plot(K_VALUES, ch_scores, marker='s', color=color2, linewidth=2, linestyle='-.')
ax2.tick_params(axis='y', labelcolor=color2)

# no plt.title(...) — Fig S1 title belongs in the caption
fig.tight_layout()
plt.savefig("clustering_metrics_evaluation.png", dpi=300)
plt.savefig("clustering_metrics_evaluation.pdf", dpi=300)
print("wrote clustering_metrics_evaluation.{png,pdf}")
