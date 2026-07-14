# Step 1 — Semantic map construction

This step turns the reference proteome into the **semantic map**: an assignment of
every reference transcript to one of `K = 50` buckets, used as ground-truth labels
for training (Step 2).

## Flow

```
proteome (.fa)
   │  embed_esm2.py            ← ESM-2 mean-pooled embeddings (facebook/esm2_t33_650M_UR50D)
   ▼
esm2_embeddings.npy  ──►  cluster_eval.py            (silhouette / CH sweep, K=10–100 → Fig S1)
(+ protein_ids.txt)  ──►  benchmark_clustering.py     (Table 1: evenness + isoform integrity)
                     ──►  benchmark_leiden.py         (Leiden/Louvain resolution comparison)
   │
   ▼
cDNA (.fa.gz)
   │  simulate_reads.R         ← polyester, illumina5 profile, 0.5% error, 100 bp, 3+3 reps
   ▼
simulated FASTQ  ──►  (parse + attach bucket labels)  ──►  simulation_data_*.csv
   ▼
Step 2 (train_teacher.py / distill_student.py)
```

## Scripts

| Script | Purpose | Paper artifact |
|---|---|---|
| `embed_esm2.py` | ESM-2 (`esm2_t33_650M_UR50D`) over the proteome; padding-masked mean pooling → `esm2_embeddings.npy` (+ `protein_ids.txt`) | 1,280-dim embeddings |
| `simulate_reads.R` | polyester paired-end simulation, `illumina5` profile, 0.5% error, 100 bp, 3+3 reps, length-proportional coverage | illumina5 training reads |
| `cluster_eval.py` | Agglomerative (Ward) clustering across `K=10–100`; silhouette + Calinski-Harabasz | Fig S1; K=50 selection |
| `benchmark_clustering.py` | Compares K-Means / DBSCAN / Leiden / Hierarchical on evenness (CV) + isoform integrity | Table 1 |
| `benchmark_leiden.py` | Leiden/Louvain resolution search on the embedding graph | Table 1 (Leiden row) |

**Paths** — all scripts read their inputs from environment variables (see
`configs/paths.example.sh`): `DEEPSEMP_PROTEOME_FASTA` + `DEEPSEMP_EMBED_OUTDIR` +
`DEEPSEMP_GPU` for embedding, `DEEPSEMP_CDNA_FASTA` + `DEEPSEMP_SIM_OUTDIR` for
simulation, and `DEEPSEMP_EMBEDDINGS` (+ `DEEPSEMP_TRANSCRIPTS_TSV`) for clustering.
Each falls back to the original server default if the variable is unset.

> **Remaining manual step:** after `simulate_reads.R`, the polyester FASTQ output is
> parsed and each read tagged with its transcript's bucket label to build the
> `simulation_data_illumina5_{train,simval}.csv` files consumed by Step 2. (Original
> filenames: `generate_esm_vectors.py`, `generate_polyester_high_error.R`.)
