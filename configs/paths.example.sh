#!/usr/bin/env bash
# =============================================================================
# Deep-SemP — central path configuration (EXAMPLE)
# -----------------------------------------------------------------------------
# HOW TO USE
#   1. Copy this file:            cp configs/paths.example.sh configs/paths.sh
#   2. Edit the paths below to point at your data / references / tools.
#   3. Before running any pipeline script, either:
#        export DEEPSEMP_CONFIG="$PWD/configs/paths.sh"   # scripts auto-source it
#      or simply:
#        source configs/paths.sh                          # export into your shell
#
# Every pipeline script reads these as ${VAR:-<original default>}, so anything
# you set here overrides the built-in default without editing the scripts.
# The values shown are the original server paths, kept as working examples.
# configs/paths.sh is git-ignored, so your local paths never get committed.
# =============================================================================

# ---- Step 1 inputs: embedding + read simulation -----------------------------
export DEEPSEMP_PROTEOME_FASTA="/path/to/reference/proteome.fa"    # input to embed_esm2.py (amino-acid FASTA)
export DEEPSEMP_EMBED_OUTDIR="/path/to/embeddings"                 # where embed_esm2.py writes esm2_embeddings.npy + protein_ids.txt
export DEEPSEMP_GPU="2"                                            # GPU id for embed_esm2.py (sets CUDA_VISIBLE_DEVICES)
export DEEPSEMP_CDNA_FASTA="/path/to/reference/cdna.fa.gz"         # input to simulate_reads.R (transcriptome cDNA)
export DEEPSEMP_SIM_OUTDIR="/path/to/simulation/polyester_illumina5"  # polyester output dir

# ---- Clustering + training inputs (Step 1 / Step 2) -------------------------
export DEEPSEMP_EMBEDDINGS="/path/to/esm2_embeddings.npy"          # ESM-2 mean-pooled proteome embeddings (num_seqs x 1280)
export DEEPSEMP_TRANSCRIPTS_TSV="/path/to/transcripts.tsv"         # transcript_id, gene_id table (for isoform-integrity metric)
export TEACHER_WEIGHTS="/path/to/teacher/best_model.pt"            # DNABERT-2 teacher checkpoint
export DATA_PATH="/path/to/simulation_data_illumina5_train.csv"    # distillation training CSV (sequence,label)
export SIMVAL_PATH="/path/to/simulation_data_illumina5_simval.csv" # held-out simulated validation CSV
export BASE_OUT="/path/to/output/distilled_models"                 # distillation output root

# ---- Trained student + raw reads (Step 3) -----------------------------------
export MODEL_PATH="/path/to/distilled_models/d384_l8_a03_t2_no_hidden/best_student_model.pt"  # shipped 8-layer student
export R1="/path/to/sample_R1.fastq.gz"                            # paired-end reads to partition
export R2="/path/to/sample_R2.fastq.gz"
export OUTPUT_DIR="/path/to/partitioned_reads"                     # per-bucket FASTQ output
export BUCKET_DIR="/path/to/partitioned_reads"                     # input to complexity + Trinity (== OUTPUT_DIR after merge)
export BASE_OUT_TRINITY="/path/to/assemblies/final_benchmark"      # Trinity per-bucket output root
export COMPLEXITY_TSV="/path/to/assemblies/bucket_complexity/merged_bucket_complexity.tsv"

# ---- References & tools for post-processing / evaluation (Step 4) -----------
export CPC2="/path/to/CPC2_standalone-1.0.1/bin/CPC2.py"           # Coding Potential Calculator 2
export REF_DNA="/path/to/reference/genome_dna.fa.gz"               # reference genome (minimap2 splice-aware)
export REF_GTF="/path/to/reference/annotation.gtf"                 # reference annotation (GffCompare)
export REF_TRANSCRIPTS="/path/to/reference/cdna.fa.gz"             # reference cDNA (BLASTn unique/full-length)

# ---- Runtime knobs (optional overrides) -------------------------------------
export THREADS="${THREADS:-30}"
export NUM_BUCKETS="${NUM_BUCKETS:-50}"
export N_CHUNKS="${N_CHUNKS:-4}"          # GPU chunks for parallel partitioning
# export GPUS="0 1 2 3"                    # GPU ids (edit inside parallel_route.sh if needed)
