import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from collections import Counter
import time

# ==========================================
# Configuration
# ==========================================
# TODO: Update this to your actual embeddings file
EMBEDDINGS_PATH = os.environ.get("DEEPSEMP_EMBEDDINGS", "/data3/projects/2025_Assembly/eyh/c_elegans/embeddings/esm2_embeddings.npy")
TSV_PATH = os.environ.get("DEEPSEMP_TRANSCRIPTS_TSV", "/data3/projects/2025_Assembly/eyh/c_elegans/windows/transcripts.tsv")
K_TARGET = 50

def calculate_metrics(labels, df_mapping, algo_name):
    df_mapping['bucket_label'] = labels
    
    # 1. Calculate Bucket Evenness (Coefficient of Variation)
    # Exclude noise points (-1) for DBSCAN's CV calculation so it doesn't skew the true bucket sizes
    valid_labels = [lbl for lbl in labels if lbl != -1]
    counts = list(Counter(valid_labels).values())
    
    if len(counts) == 0:
        cv = float('inf')
    else:
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        cv = (std_size / mean_size) * 100 
    
    # 2. Calculate Isoform Integrity
    isoform_counts = df_mapping.groupby('gene_id')['transcript_id'].count()
    multi_isoform_genes = isoform_counts[isoform_counts > 1].index
    
    df_multi = df_mapping[df_mapping['gene_id'].isin(multi_isoform_genes)]
    total_multi_genes = len(multi_isoform_genes)
    
    # A gene is "intact" if all its transcripts share the exact same bucket label
    bucket_counts_per_gene = df_multi.groupby('gene_id')['bucket_label'].nunique()
    intact_genes = (bucket_counts_per_gene == 1).sum()
    
    integrity_rate = (intact_genes / total_multi_genes) * 100 if total_multi_genes > 0 else 0
    
    noise_count = list(labels).count(-1)
    
    print(f"--- {algo_name} ---")
    print(f"Number of Clusters Created: {len(set(valid_labels))}")
    if noise_count > 0:
        print(f"Unclustered/Noise Sequences: {noise_count}")
    if len(counts) > 0:
        print(f"Largest Bucket: {max(counts)} | Smallest Bucket: {min(counts)}")
    print(f"Evenness (CV): {cv:.2f}% (Lower is better = balanced parallel compute)")
    print(f"Isoform Integrity Rate: {integrity_rate:.2f}% (Higher is better = biologically sound)\n")

def main():
    print("1. Loading Data...")
    
    # Load the TSV and build the mapping table on the fly
    df = pd.read_csv(TSV_PATH, sep='\t')
    df_mapping = df[['wb_gene', 'transcript_id']].copy()
    df_mapping.rename(columns={'wb_gene': 'gene_id'}, inplace=True)
    
    # Load Embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    
    if len(df_mapping) != embeddings.shape[0]:
        print(f"[ERROR] Dimension mismatch! TSV has {len(df_mapping)} rows, Embeddings have {embeddings.shape[0]} rows.")
        print("Please ensure the embeddings were generated in the exact order of the TSV.")
        return
        
    print(f"Loaded {len(df_mapping)} transcripts successfully.\n")
    
    # --- 1. Hierarchical Clustering (Deep-SemP Baseline) ---
    print("Running Hierarchical Clustering...")
    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=K_TARGET)
    hc_labels = hc.fit_predict(embeddings)
    print(f"Done in {time.time()-t0:.1f}s")
    calculate_metrics(hc_labels, df_mapping.copy(), "Hierarchical Clustering")
    
    # --- 2. K-Means ---
    print("Running K-Means...")
    t0 = time.time()
    kmeans = KMeans(n_clusters=K_TARGET, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(embeddings)
    print(f"Done in {time.time()-t0:.1f}s")
    calculate_metrics(km_labels, df_mapping.copy(), "K-Means")
    
    # --- 3. DBSCAN ---
    print("Running DBSCAN...")
    t0 = time.time()
    # eps=0.5 and min_samples=5 are standard defaults, but DBSCAN will likely struggle heavily here
    dbscan = DBSCAN(eps=0.5, min_samples=5) 
    db_labels = dbscan.fit_predict(embeddings)
    print(f"Done in {time.time()-t0:.1f}s")
    calculate_metrics(db_labels, df_mapping.copy(), "DBSCAN")

if __name__ == "__main__":
    main()