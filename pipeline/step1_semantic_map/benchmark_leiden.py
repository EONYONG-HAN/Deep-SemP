import os
import numpy as np
import pandas as pd
from collections import Counter
import time
import scanpy as sc
import anndata as ad

# ==========================================
# Configuration
# ==========================================
EMBEDDINGS_PATH = os.environ.get("DEEPSEMP_EMBEDDINGS", "/data3/projects/2025_Assembly/eyh/c_elegans/embeddings/esm2_embeddings.npy")
TSV_PATH = os.environ.get("DEEPSEMP_TRANSCRIPTS_TSV", "/data3/projects/2025_Assembly/eyh/c_elegans/windows/transcripts.tsv")
K_TARGET = 50

def calculate_metrics(labels, df_mapping, algo_name):
    df_mapping['bucket_label'] = labels
    
    valid_labels = [lbl for lbl in labels if lbl != -1]
    counts = list(Counter(valid_labels).values())
    
    if len(counts) == 0:
        cv = float('inf')
    else:
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        cv = (std_size / mean_size) * 100 
    
    isoform_counts = df_mapping.groupby('gene_id')['transcript_id'].count()
    multi_isoform_genes = isoform_counts[isoform_counts > 1].index
    
    df_multi = df_mapping[df_mapping['gene_id'].isin(multi_isoform_genes)]
    total_multi_genes = len(multi_isoform_genes)
    
    bucket_counts_per_gene = df_multi.groupby('gene_id')['bucket_label'].nunique()
    intact_genes = (bucket_counts_per_gene == 1).sum()
    
    integrity_rate = (intact_genes / total_multi_genes) * 100 if total_multi_genes > 0 else 0
    
    print(f"--- {algo_name} ---")
    print(f"Number of Clusters Created: {len(set(valid_labels))}")
    if len(counts) > 0:
        print(f"Largest Bucket: {max(counts)} | Smallest Bucket: {min(counts)}")
    print(f"Evenness (CV): {cv:.2f}% (Lower is better = balanced parallel compute)")
    print(f"Isoform Integrity Rate: {integrity_rate:.2f}% (Higher is better = biologically sound)\n")

def find_optimal_resolution(adata, target_k, algo='leiden'):
    """Binary search to find the resolution that yields exactly K_TARGET clusters"""
    min_res, max_res = 0.01, 5.0
    best_labels = None
    best_k = -1
    
    print(f"Hunting for resolution to hit K={target_k} for {algo}...")
    for i in range(15): # Max 15 attempts
        res = (min_res + max_res) / 2
        
        if algo == 'leiden':
            sc.tl.leiden(adata, resolution=res, key_added='cluster')
        else:
            sc.tl.louvain(adata, resolution=res, key_added='cluster')
            
        labels = adata.obs['cluster'].astype(int).values
        k = len(set(labels))
        
        # Track the closest match just in case we can't hit exactly 50
        if abs(k - target_k) < abs(best_k - target_k) or best_k == -1:
            best_k = k
            best_labels = labels.copy()
            
        if k == target_k:
            print(f"  -> Hit exactly K={target_k} at resolution={res:.4f}")
            break
        elif k < target_k:
            min_res = res # Need more clusters, increase resolution
        else:
            max_res = res # Need fewer clusters, decrease resolution
            
    if best_k != target_k:
        print(f"  -> Settling for closest K={best_k}")
        
    return best_labels

def main():
    print("1. Loading Data...")
    df = pd.read_csv(TSV_PATH, sep='\t')
    df_mapping = df[['wb_gene', 'transcript_id']].copy()
    df_mapping.rename(columns={'wb_gene': 'gene_id'}, inplace=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    
    print("2. Building K-Nearest Neighbors Graph (scanpy)...")
    t0 = time.time()
    # Convert to AnnData object, standard for scanpy
    adata = ad.AnnData(embeddings)
    # Build neighborhood graph (using 15 neighbors, standard default)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    print(f"Graph built in {time.time()-t0:.1f}s\n")
    
    # # --- 4. Louvain ---
    # print("Running Louvain...")
    # t0 = time.time()
    # louvain_labels = find_optimal_resolution(adata, K_TARGET, algo='louvain')
    # print(f"Done in {time.time()-t0:.1f}s")
    # calculate_metrics(louvain_labels, df_mapping.copy(), "Louvain")
    
    # # --- 5. Leiden ---
    # print("Running Leiden...")
    # t0 = time.time()
    # leiden_labels = find_optimal_resolution(adata, K_TARGET, algo='leiden')
    # print(f"Done in {time.time()-t0:.1f}s")
    # calculate_metrics(leiden_labels, df_mapping.copy(), "Leiden")
    # --- 5. Leiden (Default Resolution) ---
    print("Running Leiden (res=1.0)...")
    t0 = time.time()
    # Run it natively without the binary search loop
    sc.tl.leiden(adata, resolution=1.0, key_added='cluster')
    leiden_default_labels = adata.obs['cluster'].astype(int).values
    print(f"Done in {time.time()-t0:.1f}s")
    calculate_metrics(leiden_default_labels, df_mapping.copy(), "Leiden (res=1.0)")

if __name__ == "__main__":
    main()