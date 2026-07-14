# Deep-SemP tutorial — step-by-step walkthrough

This walks through the full pipeline on the *C. elegans* primary benchmark, and shows
how to apply the trained model to a new species (zero-shot). Each step lists its
**inputs**, the **command**, and the **outputs** it produces.

Before starting:

```bash
conda activate deep-semp
cp configs/paths.example.sh configs/paths.sh   # then edit configs/paths.sh
export DEEPSEMP_CONFIG="$PWD/configs/paths.sh" # all scripts auto-source this
```

All pipeline scripts read paths as `${VAR:-default}`; anything set in `configs/paths.sh`
or the environment overrides the built-in default. Python scripts that read paths
from variables at the top of the file (e.g. the analysis scripts) are noted inline.

---

## Step 1 — Build the semantic map

**Goal:** assign every reference transcript to one of `K = 50` buckets, to serve as
ground-truth labels for training.

**1a. Embed the proteome with ESM-2** *(script to be added — see
`pipeline/step1_semantic_map/README.md`)*

Run ESM-2 `facebook/esm2_t33_650M_UR50D` (650 M params) over the reference proteome,
mean-pool the residue embeddings to one 1,280-dim vector per sequence, and save
`esm2_embeddings.npy` (shape `num_seqs × 1280`).

```bash
export DEEPSEMP_EMBEDDINGS=/path/to/esm2_embeddings.npy
export DEEPSEMP_TRANSCRIPTS_TSV=/path/to/transcripts.tsv   # transcript_id, gene_id
```

**1b. Choose K and validate clustering**

```bash
# Silhouette + Calinski-Harabasz sweep, K=10..100  (→ Fig S1; K=50 chosen)
python pipeline/step1_semantic_map/cluster_eval.py

# Table 1: cluster evenness (CV) + isoform integrity across 4 algorithms
python pipeline/step1_semantic_map/benchmark_clustering.py

# Leiden/Louvain resolution comparison
python pipeline/step1_semantic_map/benchmark_leiden.py
```

**Outputs:** clustering metric plots/tables, and the 31,865-transcript → 50-bucket
**semantic map** used as labels below.

**1c. Simulate training reads** *(script to be added)*

Use polyester to simulate paired-end 100 bp reads from the reference transcriptome
under the `illumina5` error profile (0.5% error), tagged with each read's bucket
label, written as `simulation_data_illumina5_train.csv` and `..._simval.csv`.

---

## Step 2 — Train the teacher, distill the student

**Goal:** a lightweight student model that partitions reads into the 50 buckets.

**2a. Fine-tune the DNABERT-2 teacher**

*Inputs:* training CSV (`$DATA_PATH`). *Output:* teacher checkpoint (`best_model.pt`).

```bash
python pipeline/step2_train_distill/train_teacher.py \
    --data_path "$DATA_PATH" \
    --output_dir results/checkpoints/teacher \
    --num_labels 50 --epochs 10 --lr 2e-5 --batch_size 32 \
    --max_len 100 --stratify_split
```

**2b. Distill into the 8-layer student** (the shipped configuration)

*Inputs:* teacher weights, illumina5 train + simval CSVs.
*Output:* `best_student_model.pt` (d_model 384, 8 layers, 12.8 M params).

```bash
python pipeline/step2_train_distill/distill_student.py \
    --teacher_weights "$TEACHER_WEIGHTS" \
    --data_path       "$DATA_PATH" \
    --simval_path     "$SIMVAL_PATH" \
    --output_dir      results/checkpoints/student \
    --epochs 20 --batch_size 768 \
    --d_model 384 --num_layers 8 --dim_feedforward 1024 --nhead 8 \
    --temperature 2.0 --alpha 0.3 --warmup_ratio 0.05 \
    --masked_pooling --stratify_split --no_hidden_distill
```

`run_distill_grid.sh` wraps this and launches the architecture sweep (6/8/10 layers,
d_model 384/512) used for Supplementary Table S2. The paper's `--temperature 2.0
--alpha 0.3` reflect the grid-search optimum; the 8-layer model is selected on
real-data generalization (diminishing returns beyond 8 layers).

**2c. Evaluate partitioning accuracy** (Tables 2 & 3)

```bash
python pipeline/step2_train_distill/eval_teacher_accuracy.py     # teacher, real reads
python pipeline/step2_train_distill/eval_partition_accuracy.py   # student, real reads
```

> These read their model/data paths from the constants at the top of each file —
> edit them (or export the matching env vars) before running. Accuracy on real reads
> is reported as mean ± SD over five resamples of 10,000 reads.

---

## Step 3 — Partition reads and assemble

**Goal:** route real RNA-seq reads to buckets on 4 GPUs, then assemble each bucket
independently with complexity-adaptive CPU scheduling.

**3a. Parallel partitioning (4 GPUs)**

*Inputs:* `$MODEL_PATH` (student), `$R1`/`$R2` (reads). *Output:* per-bucket FASTQ in
`$OUTPUT_DIR`. The script splits the FASTQ into `N_CHUNKS` (zstd), routes each chunk on
its own GPU via `partition_reads.py`, then merges per-bucket files across chunks.

```bash
bash pipeline/step3_partition_assemble/parallel_route.sh
```

Edit `GPUS=(...)` inside the script for your GPU ids. Per-species wrappers
(`parallel_route_briggsae.sh`, `_remanei.sh`, `_latens.sh`, `_sapiens.sh`) show the
zero-shot application to other species — 151 bp reads are auto-truncated to 100 bp
during tokenization while Trinity uses the full length.

**3b. Compute per-bucket k-mer complexity**

*Inputs:* per-bucket FASTQ (`$BUCKET_DIR`). *Output:* `merged_bucket_complexity.tsv`
(the `cpu_score` per bucket, via Jellyfish `k=25`).

```bash
bash pipeline/step3_partition_assemble/compute_bucket_complexity.sh
```

**3c. Adaptive Trinity assembly**

*Inputs:* per-bucket FASTQ + complexity TSV. *Output:* per-bucket `Trinity_bucket_XX.fasta`.
The look-ahead scheduler assigns 4–32 CPUs per bucket by `cpu_score` (cap 40 cores).

```bash
bash pipeline/step3_partition_assemble/run_trinity_adaptive.sh
```

**3d. (optional) Multi-server scaling analysis** (Table 4)

```bash
python pipeline/step3_partition_assemble/routing_simulation.py   # Monte Carlo, N=1000/server count
```

---

## Step 4 — Post-process and evaluate

**Goal:** merge buckets, remove redundancy, keep coding contigs, and compute the
quality metrics reported in Tables 5 & 6.

**4a. Post-processing + evaluation pipeline**

Merges all buckets → CD-HIT-EST (90% id) → CAP3 → 300 bp length filter → CPC2 coding
filter, then runs BUSCO, GffCompare (structural precision), and Bowtie2 (mapping rate).

```bash
bash pipeline/step4_postprocess_eval/postprocess.sh \
    --fasta_dir "$BASE_OUT_TRINITY" \
    --label deepsemp \
    --outdir results/postprocess/deepsemp \
    --r1 "$R1" --r2 "$R2" \
    --ref_dna "$REF_DNA" --ref_gtf "$REF_GTF"
```

Use `--fasta` instead of `--fasta_dir` to post-process a single merged FASTA (e.g. the
monolithic baseline). `evaluate_checkpoints.sh LABEL OUTDIR` re-runs BUSCO/GffCompare
at the length-filter and CPC2 checkpoints (used for the *H. sapiens* boundary case).

**4b. BLASTn: unique transcripts + full-length recovery**

```bash
bash pipeline/step4_postprocess_eval/run_blast.sh     # edit REF_TRANSCRIPTS + assembly paths / configs
```

**4c. Assembly statistics** (N50, GC, counts → Tables 5/6)

```bash
python pipeline/step4_postprocess_eval/analyze_assemblies.py   # see paths in the script header
```

---

## Reproducing the figures

Scripts in `analysis/` regenerate the manuscript figures from the evaluation outputs
(paths are set as constants near the top of each script):

| Script | Produces |
|---|---|
| `plot_entropy_fig2.py` | Fig 2 — routing entropy vs phylogenetic divergence (Spearman ρ=0.89) |
| `plot_scheduling.py` | CPU-scheduling grid (Fig 1 inset) |
| `contig_length.py` | contig-length / full-length / reference-coverage distributions |
| `extract_all_case.py`, `extract_briggsae_remanei.py` | rescued/missed gene lists (→ Supplementary Data S1) |

---

## Notes & tips

- **Run order matters:** Steps 1→2 build the model; Steps 3→4 apply and evaluate it.
  To just *use* Deep-SemP on new reads, you only need the trained student and Steps 3–4.
- **Zero-shot transfer:** the *C. elegans*-trained student applies to other
  *Caenorhabditis* species without retraining. Routing entropy (see `analysis/`) is a
  pre-assembly diagnostic — values near the ~0.8 nematode range indicate the method
  will help; values approaching 1.0 (e.g. *H. sapiens*, 0.913) indicate near-random
  partitioning and mark the phylogenetic boundary.
- **Read length:** train/tokenize at 100 bp; longer reads (151 bp) are truncated for
  routing but assembled at full length by Trinity.
- **`legacy/`** holds earlier script versions for provenance; use the canonical scripts
  in `pipeline/`.
