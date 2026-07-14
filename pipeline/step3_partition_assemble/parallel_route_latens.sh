#!/usr/bin/env bash
set -euo pipefail

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

# =========================
# User settings
# =========================
N_CHUNKS=4
GPUS=(0 1 2 3)

MODEL_PATH="${MODEL_PATH:-/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/distilled_models/illumina5_grid/d384_l8_a03_t2_no_hidden/best_student_model.pt}"
R1="${R1:-/data3/projects/2025_Assembly/eyh/c_latens/raw_data/c_latens_R1.fastq.gz}"
R2="${R2:-/data3/projects/2025_Assembly/eyh/c_latens/raw_data/c_latens_R2.fastq.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/projects/2025_Assembly/eyh/c_latens/partitioned_reads/student_nh_illumina5_v2}"
CHUNK_DIR="${OUTPUT_DIR}/chunks"
PYTHON_BIN=python
ROUTE_SCRIPT="${ROUTE_SCRIPT:-$(dirname "$0")/partition_reads.py}"

# Model architecture — must match training run
NUM_LABELS=50
NUM_LAYERS=8
D_MODEL=384
BATCH_SIZE=12288
FP16="--fp16"
ZSTD_THREADS=4

# =========================
# Derived settings
# =========================
SAMPLE_NAME=$(basename "${R1}" | sed 's/_R1\.fastq\.gz//')
mkdir -p "${CHUNK_DIR}"
mkdir -p "${OUTPUT_DIR}"

WALL_START=$(date +%s)

echo "=============================================="
echo "  Parallel Read Routing — C. latens"
echo "  Sample     : ${SAMPLE_NAME}"
echo "  N chunks   : ${N_CHUNKS}"
echo "  GPUs       : ${GPUS[*]}"
echo "  Model      : ${MODEL_PATH}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "=============================================="
echo

# =========================
# Step 1: Count total reads
# =========================
echo "[Step 1] Counting reads in R1..."
TOTAL_READS=$(zcat "${R1}" | wc -l | awk '{print $1/4}')
echo "  Total pairs: ${TOTAL_READS}"

READS_PER_CHUNK=$(( (TOTAL_READS + N_CHUNKS - 1) / N_CHUNKS ))
LINES_PER_CHUNK=$(( ( (READS_PER_CHUNK + 3) / 4 ) * 4 * 4 ))
echo "  Pairs per chunk: ~${READS_PER_CHUNK}"
echo "  Lines per chunk: ${LINES_PER_CHUNK}"
echo

# =========================
# Step 2: Split R1 and R2
# =========================
echo "[Step 2] Splitting FASTQ files into ${N_CHUNKS} chunks..."
SPLIT_START=$(date +%s)

echo "  Splitting R1 (zstd)..."
zcat "${R1}" | split \
  --lines="${LINES_PER_CHUNK}" \
  --numeric-suffixes=1 \
  --suffix-length=1 \
  --filter="zstd -T${ZSTD_THREADS} -q > \$FILE.fastq.zst" \
  - "${CHUNK_DIR}/${SAMPLE_NAME}_R1_chunk"

echo "  Splitting R2 (zstd)..."
zcat "${R2}" | split \
  --lines="${LINES_PER_CHUNK}" \
  --numeric-suffixes=1 \
  --suffix-length=1 \
  --filter="zstd -T${ZSTD_THREADS} -q > \$FILE.fastq.zst" \
  - "${CHUNK_DIR}/${SAMPLE_NAME}_R2_chunk"

SPLIT_END=$(date +%s)
echo "  Split done in $(( SPLIT_END - SPLIT_START ))s"

R1_CHUNKS=( "${CHUNK_DIR}/${SAMPLE_NAME}_R1_chunk"*.fastq.zst )
R2_CHUNKS=( "${CHUNK_DIR}/${SAMPLE_NAME}_R2_chunk"*.fastq.zst )
echo "  R1 chunks: ${#R1_CHUNKS[@]}"
echo "  R2 chunks: ${#R2_CHUNKS[@]}"

if [ "${#R1_CHUNKS[@]}" -ne "${#R2_CHUNKS[@]}" ]; then
  echo "ERROR: R1 and R2 chunk counts don't match. Aborting."
  exit 1
fi
echo

# =========================
# Step 3: Run routing in parallel
# =========================
echo "[Step 3] Launching ${N_CHUNKS} routing jobs in parallel..."
ROUTE_START=$(date +%s)

PIDS=()
for i in $(seq 1 "${N_CHUNKS}"); do
  chunk_r1="${CHUNK_DIR}/${SAMPLE_NAME}_R1_chunk${i}.fastq.zst"
  chunk_r2="${CHUNK_DIR}/${SAMPLE_NAME}_R2_chunk${i}.fastq.zst"
  chunk_out="${OUTPUT_DIR}/chunk_${i}"
  gpu_id="${GPUS[$((i-1))]}"
  log_file="${CHUNK_DIR}/route_chunk${i}.log"

  mkdir -p "${chunk_out}"

  echo "  Chunk ${i} | GPU ${gpu_id} | ${chunk_r1}"
  echo "             → ${chunk_out}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${PYTHON_BIN}" "${ROUTE_SCRIPT}" \
    --model_path    "${MODEL_PATH}" \
    --r1            "${chunk_r1}" \
    --r2            "${chunk_r2}" \
    --output_dir    "${chunk_out}" \
    --num_labels    "${NUM_LABELS}" \
    --num_layers    "${NUM_LAYERS}" \
    --d_model       "${D_MODEL}" \
    --batch_size    "${BATCH_SIZE}" \
    --masked_pooling \
    ${FP16} \
    --log_every     500000 \
    > "${log_file}" 2>&1 &

  PIDS+=($!)
  echo "    pid: ${PIDS[-1]}"
done

echo
echo "  Waiting for all routing jobs to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  chunk_num=$((i+1))
  if wait "${pid}"; then
    echo "  [OK] Chunk $((i+1)) finished (pid ${pid})"
  else
    echo "  [FAIL] Chunk $((i+1)) failed (pid ${pid}) — check ${CHUNK_DIR}/route_chunk${chunk_num}.log"
    FAILED=$((FAILED+1))
  fi
done

ROUTE_END=$(date +%s)
echo "  Routing done in $(( (ROUTE_END - ROUTE_START) / 60 ))m $(( (ROUTE_END - ROUTE_START) % 60 ))s"

if [ "${FAILED}" -gt 0 ]; then
  echo "ERROR: ${FAILED} routing job(s) failed. Check logs in ${CHUNK_DIR}/"
  exit 1
fi
echo

# =========================
# Step 4: Merge buckets
# =========================
echo "[Step 4] Merging ${NUM_LABELS} buckets across ${N_CHUNKS} chunks..."
MERGE_START=$(date +%s)

for bucket in $(seq -w 0 $((NUM_LABELS-1))); do
  cat "${OUTPUT_DIR}"/chunk_*/bucket_${bucket}_R1.fastq \
    > "${OUTPUT_DIR}/bucket_${bucket}_R1.fastq"
  cat "${OUTPUT_DIR}"/chunk_*/bucket_${bucket}_R2.fastq \
    > "${OUTPUT_DIR}/bucket_${bucket}_R2.fastq"
done

MERGE_END=$(date +%s)
echo "  Merge done in $(( MERGE_END - MERGE_START ))s"
echo

# =========================
# Step 5: Summary
# =========================
WALL_END=$(date +%s)
WALL_ELAPSED=$(( WALL_END - WALL_START ))

echo "=============================================="
echo "  COMPLETE — C. latens routing"
echo "=============================================="
echo "  Total pairs routed : ${TOTAL_READS}"
echo "  Output directory   : ${OUTPUT_DIR}"
echo ""
echo "  Timing breakdown:"
echo "    Split    : $(( SPLIT_END - SPLIT_START ))s"
echo "    Routing  : $(( ROUTE_END - ROUTE_START ))s"
echo "    Merge    : $(( MERGE_END - MERGE_START ))s"
echo "    Total    : ${WALL_ELAPSED}s  ($(echo "scale=2; ${WALL_ELAPSED}/3600" | bc)h)"
echo ""

echo "  Per-bucket read counts (top 10 largest):"
for bucket in $(seq -w 0 $((NUM_LABELS-1))); do
  f="${OUTPUT_DIR}/bucket_${bucket}_R1.fastq"
  if [ -f "${f}" ]; then
    count=$(wc -l < "${f}")
    pairs=$(( count / 4 ))
    echo "${bucket} ${pairs}"
  fi
done | sort -k2 -rn | head -10 | \
  awk '{printf "    Bucket %s : %'"'"'d pairs\n", $1, $2}'

dist_file="${OUTPUT_DIR}/bucket_distribution_merged.tsv"
echo "bucket	pairs" > "${dist_file}"
for bucket in $(seq -w 0 $((NUM_LABELS-1))); do
  f="${OUTPUT_DIR}/bucket_${bucket}_R1.fastq"
  count=0
  [ -f "${f}" ] && count=$(( $(wc -l < "${f}") / 4 ))
  echo -e "${bucket}\t${count}" >> "${dist_file}"
done
echo "  Distribution saved to: ${dist_file}"

echo ""
read -p "  Delete chunk files to save disk space? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  rm -rf "${CHUNK_DIR}"
  echo "  Chunk directory deleted."
else
  echo "  Chunk files kept at: ${CHUNK_DIR}"
fi

echo "=============================================="