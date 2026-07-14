# Step 1 — Semantic map construction

This step turns the reference proteome into the **semantic map**: an assignment of
every reference transcript to one of `K = 50` buckets, used as ground-truth labels
for training (Step 2).

## Flow

```
proteome (.faa)
   │  embed_esm2.py            ← ESM-2 mean-pooled embeddings  [TO ADD]
   ▼
esm2_embeddings.npy  ──►  cluster_eval.py            (silhouette / CH sweep, K=10–100 → Fig S1)
                     ──►  benchmark_clustering.py     (Table 1: evenness + isoform integrity)
                     ──►  benchmark_leiden.py         (Leiden/Louvain resolution comparison)
   │
   ▼
polyester simulation (.R)      ← simulated training reads      [TO ADD]
   ▼
simulation_data_*.csv  ──►  Step 2 (train_teacher.py / distill_student.py)
```

## Scripts present

| Script | Purpose | Paper artifact |
|---|---|---|
| `cluster_eval.py` | Agglomerative (Ward) clustering across `K=10–100`; silhouette + Calinski-Harabasz | Fig S1; K=50 selection |
| `benchmark_clustering.py` | Compares K-Means / DBSCAN / Leiden / Hierarchical on evenness (CV) + isoform integrity | Table 1 |
| `benchmark_leiden.py` | Leiden/Louvain resolution search on the embedding graph | Table 1 (Leiden row) |

All three read the embeddings via the `DEEPSEMP_EMBEDDINGS` environment variable
(and `DEEPSEMP_TRANSCRIPTS_TSV` for the isoform-integrity metric). See
`configs/paths.example.sh`.

## Scripts to add (upstream, not yet in repo)

These produce the inputs the scripts above consume. Drop them here when ready:

- **`embed_esm2.py`** — run ESM-2 (`facebook/esm2_t33_650M_UR50D`, 650 M params) over
  the reference proteome, mean-pool residue embeddings to a 1,280-dim vector per
  sequence, and save `esm2_embeddings.npy` (shape `num_seqs × 1280`).
- **`simulate_reads.R`** — polyester simulation of paired-end reads (100 bp) from the
  reference transcriptome under the `illumina5` error profile (0.5% error), tagged
  with the bucket label from the semantic map, written as the training/validation CSVs.

Until added, the tutorial documents the exact model name and parameters so the
step is reproducible.
