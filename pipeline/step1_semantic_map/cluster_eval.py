import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time

# ==========================================
# Configuration
# ==========================================
# TODO: Replace with your actual ESM-2 embeddings file
EMBEDDINGS_PATH = os.environ.get("DEEPSEMP_EMBEDDINGS", "/data3/projects/2025_Assembly/eyh/c_elegans/embeddings/esm2_embeddings.npy")
K_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def main():
    print("Loading ESM-2 Embeddings...")
    # Assuming embeddings are saved as a numpy array of shape (num_genes, hidden_dim)
    # If it's a torch tensor, use: embeddings = torch.load(EMBEDDINGS_PATH).cpu().numpy()
    embeddings = np.load(EMBEDDINGS_PATH) 
    print(f"Loaded {embeddings.shape[0]} sequences with dimension {embeddings.shape[1]}")

    silhouette_scores = []
    ch_scores = []

    print("\nStarting clustering evaluation sweep...")
    for k in K_VALUES:
        start_time = time.time()
        
        # Using hierarchical clustering to match your original pipeline
        clusterer = AgglomerativeClustering(n_clusters=k)
        labels = clusterer.fit_predict(embeddings)
        
        # Calculate metrics
        # Note: If you have >20k sequences, calculating silhouette score can be slow. 
        # You can add `sample_size=5000` to speed it up if needed.
        sil_score = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        
        silhouette_scores.append(sil_score)
        ch_scores.append(ch_score)
        
        elapsed = time.time() - start_time
        print(f"K={k:3d} | Silhouette: {sil_score:.4f} | Calinski-Harabasz: {ch_score:8.2f} | Time: {elapsed:.1f}s")

    # ==========================================
    # Plotting the Results
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Silhouette Score', color=color1, fontsize=12)
    ax1.plot(K_VALUES, silhouette_scores, marker='o', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Highlight your chosen K=50
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Selected K=50')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Calinski-Harabasz Score', color=color2, fontsize=12)
    ax2.plot(K_VALUES, ch_scores, marker='s', color=color2, linewidth=2, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color2)

    # title removed — Fig S1 title lives in the caption
    fig.tight_layout()
    
    plot_path = "clustering_metrics_evaluation.png"
    plt.savefig(plot_path, dpi=300)
    plt.savefig("clustering_metrics_evaluation.pdf", dpi=300)
    print(f"\nDone! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()


### results save for future figure work...
# Starting clustering evaluation sweep...
# K= 10 | Silhouette: 0.0564 | Calinski-Harabasz:  2862.76 | Time: 593.4s
# K= 20 | Silhouette: 0.0463 | Calinski-Harabasz:  1703.45 | Time: 628.6s
# K= 30 | Silhouette: 0.0404 | Calinski-Harabasz:  1261.97 | Time: 623.0s
# K= 40 | Silhouette: 0.0414 | Calinski-Harabasz:  1022.69 | Time: 612.8s
# K= 50 | Silhouette: 0.0453 | Calinski-Harabasz:   866.98 | Time: 609.5s
# K= 60 | Silhouette: 0.0466 | Calinski-Harabasz:   757.40 | Time: 600.2s
# K= 70 | Silhouette: 0.0488 | Calinski-Harabasz:   675.56 | Time: 590.7s
# K= 80 | Silhouette: 0.0485 | Calinski-Harabasz:   612.77 | Time: 597.1s
# K= 90 | Silhouette: 0.0474 | Calinski-Harabasz:   562.69 | Time: 600.8s
# K=100 | Silhouette: 0.0483 | Calinski-Harabasz:   521.56 | Time: 595.9s