#!/usr/bin/env bash
# =============================================================
# Benchmark FASTQ splitting strategies
# Tests four approaches and reports speed + disk usage
# =============================================================
set -euo pipefail

# =========================
# User settings
# =========================
R1="/data3/projects/2025_Assembly/eyh/c_elegans/real_data/SRR36278209_1.fastq.gz"
BENCH_DIR="/data3/projects/2025_Assembly/eyh/c_elegans/split_benchmark"
N_CHUNKS=4

# How many reads to test with (set to 0 for full file)
# 5M reads ≈ 6.7% of 75M — fast enough to benchmark, representative enough to extrapolate
TEST_READS=5000000

mkdir -p "${BENCH_DIR}"

# =========================
# Helper: extrapolate to full file
# =========================
TOTAL_READS=75200053

extrapolate() {
  local elapsed=$1
  local test_reads=$2
  local total_reads=$3
  echo "scale=2; ${elapsed} * ${total_reads} / ${test_reads}" | bc
}

fmt_time() {
  local secs=$1
  echo "${secs}s  ($(echo "scale=2; ${secs}/60" | bc) min)"
}

echo "=============================================="
echo "  FASTQ Split Strategy Benchmark"
echo "  R1          : ${R1}"
echo "  Test reads  : ${TEST_READS}"
echo "  N chunks    : ${N_CHUNKS}"
echo "  Bench dir   : ${BENCH_DIR}"
echo "=============================================="
echo

# Count lines for splitting
LINES_PER_CHUNK=$(( (TEST_READS / N_CHUNKS) * 4 ))
echo "Lines per chunk: ${LINES_PER_CHUNK}"
echo

# Pre-extract test subset once to avoid repeated decompression overhead
echo "Extracting ${TEST_READS} reads for benchmarking..."
EXTRACT_START=$(date +%s)
# Use awk instead of head to avoid broken pipe with set -e
zcat "${R1}" | awk -v n=$(( TEST_READS * 4 )) 'NR<=n' > "${BENCH_DIR}/test_input.fastq"
EXTRACT_END=$(date +%s)
INPUT_SIZE=$(du -sh "${BENCH_DIR}/test_input.fastq" | cut -f1)
echo "  Done in $(( EXTRACT_END - EXTRACT_START ))s | Size: ${INPUT_SIZE}"
echo

# Cleanup function
cleanup_chunks() {
  rm -f "${BENCH_DIR}"/chunk_* "${BENCH_DIR}"/test_chunk* || true
}

# =========================
# Strategy 1: gzip compressed chunks (original approach)
# =========================
echo "----------------------------------------------"
echo "Strategy 1: split + gzip recompression"
echo "  (original parallel_route.sh approach)"
echo "----------------------------------------------"
cleanup_chunks
T1_START=$(date +%s)

split \
  --lines="${LINES_PER_CHUNK}" \
  --numeric-suffixes=1 \
  --suffix-length=1 \
  --filter='gzip > $FILE.fastq.gz' \
  "${BENCH_DIR}/test_input.fastq" \
  "${BENCH_DIR}/chunk_gz_"

T1_END=$(date +%s)
T1_ELAPSED=$(( T1_END - T1_START ))
T1_EXTRAP=$(extrapolate ${T1_ELAPSED} ${TEST_READS} ${TOTAL_READS})
T1_SIZE=$(du -sh "${BENCH_DIR}"/chunk_gz_*.fastq.gz 2>/dev/null | tail -1 | cut -f1)

echo "  Time (${TEST_READS} reads) : $(fmt_time ${T1_ELAPSED})"
echo "  Extrapolated (75M reads) : $(fmt_time ${T1_EXTRAP%.*})"
echo "  Chunk sizes              : ${T1_SIZE} per chunk (compressed)"
echo

# =========================
# Strategy 2: plain uncompressed chunks
# =========================
echo "----------------------------------------------"
echo "Strategy 2: split uncompressed (no gzip)"
echo "  Fastest split, uses more disk space"
echo "----------------------------------------------"
cleanup_chunks
T2_START=$(date +%s)

split \
  --lines="${LINES_PER_CHUNK}" \
  --numeric-suffixes=1 \
  --suffix-length=1 \
  "${BENCH_DIR}/test_input.fastq" \
  "${BENCH_DIR}/chunk_raw_"

T2_END=$(date +%s)
T2_ELAPSED=$(( T2_END - T2_START ))
T2_EXTRAP=$(extrapolate ${T2_ELAPSED} ${TEST_READS} ${TOTAL_READS})
T2_SIZE=$(du -sh "${BENCH_DIR}"/chunk_raw_* 2>/dev/null | tail -1 | cut -f1)

echo "  Time (${TEST_READS} reads) : $(fmt_time ${T2_ELAPSED})"
echo "  Extrapolated (75M reads) : $(fmt_time ${T2_EXTRAP%.*})"
echo "  Chunk sizes              : ${T2_SIZE} per chunk (uncompressed)"
echo

# =========================
# Strategy 3: pigz parallel compression
# =========================
echo "----------------------------------------------"
echo "Strategy 3: split + pigz parallel compression"
echo "  Compressed output, faster than gzip"
echo "----------------------------------------------"
cleanup_chunks

if command -v pigz &>/dev/null; then
  T3_START=$(date +%s)

  split \
    --lines="${LINES_PER_CHUNK}" \
    --numeric-suffixes=1 \
    --suffix-length=1 \
    --filter='pigz -p 4 > $FILE.fastq.gz' \
    "${BENCH_DIR}/test_input.fastq" \
    "${BENCH_DIR}/chunk_pigz_"

  T3_END=$(date +%s)
  T3_ELAPSED=$(( T3_END - T3_START ))
  T3_EXTRAP=$(extrapolate ${T3_ELAPSED} ${TEST_READS} ${TOTAL_READS})
  T3_SIZE=$(du -sh "${BENCH_DIR}"/chunk_pigz_*.fastq.gz 2>/dev/null | tail -1 | cut -f1)

  echo "  Time (${TEST_READS} reads) : $(fmt_time ${T3_ELAPSED})"
  echo "  Extrapolated (75M reads) : $(fmt_time ${T3_EXTRAP%.*})"
  echo "  Chunk sizes              : ${T3_SIZE} per chunk (compressed)"
else
  echo "  [SKIP] pigz not found — install with: conda install pigz"
  T3_ELAPSED=0
  T3_EXTRAP="N/A"
fi
echo

# =========================
# Strategy 4: zstd compression (fastest modern compressor)
# =========================
echo "----------------------------------------------"
echo "Strategy 4: split + zstd compression"
echo "  Fast compression, good ratio"
echo "----------------------------------------------"
cleanup_chunks

if command -v zstd &>/dev/null; then
  T4_START=$(date +%s)

  split \
    --lines="${LINES_PER_CHUNK}" \
    --numeric-suffixes=1 \
    --suffix-length=1 \
    --filter='zstd -T4 -q > $FILE.fastq.zst' \
    "${BENCH_DIR}/test_input.fastq" \
    "${BENCH_DIR}/chunk_zstd_"

  T4_END=$(date +%s)
  T4_ELAPSED=$(( T4_END - T4_START ))
  T4_EXTRAP=$(extrapolate ${T4_ELAPSED} ${TEST_READS} ${TOTAL_READS})
  T4_SIZE=$(du -sh "${BENCH_DIR}"/chunk_zstd_*.fastq.zst 2>/dev/null | tail -1 | cut -f1)

  echo "  Time (${TEST_READS} reads) : $(fmt_time ${T4_ELAPSED})"
  echo "  Extrapolated (75M reads) : $(fmt_time ${T4_EXTRAP%.*})"
  echo "  Chunk sizes              : ${T4_SIZE} per chunk (zstd)"
else
  echo "  [SKIP] zstd not found — install with: conda install zstd"
  T4_ELAPSED=0
  T4_EXTRAP="N/A"
fi
echo

# =========================
# Summary
# =========================
echo "=============================================="
echo "  SUMMARY"
echo "=============================================="
printf "%-35s | %-15s | %-20s | %s\n" "Strategy" "Test time" "Est. full (75M)" "Output"
printf "%-35s-+-%-15s-+-%-20s-+-%s\n" \
  "$(printf '%0.s-' {1..35})" "---------------" "--------------------" "-------"
printf "%-35s | %-15s | %-20s | %s\n" \
  "1. gzip recompression" \
  "$(fmt_time ${T1_ELAPSED})" \
  "~$(fmt_time ${T1_EXTRAP%.*})" \
  "compressed"
printf "%-35s | %-15s | %-20s | %s\n" \
  "2. uncompressed" \
  "$(fmt_time ${T2_ELAPSED})" \
  "~$(fmt_time ${T2_EXTRAP%.*})" \
  "raw (~60GB/file)"
[ "${T3_ELAPSED}" -gt 0 ] && printf "%-35s | %-15s | %-20s | %s\n" \
  "3. pigz parallel gzip" \
  "$(fmt_time ${T3_ELAPSED})" \
  "~$(fmt_time ${T3_EXTRAP%.*})" \
  "compressed"
[ "${T4_ELAPSED}" -gt 0 ] && printf "%-35s | %-15s | %-20s | %s\n" \
  "4. zstd parallel" \
  "$(fmt_time ${T4_ELAPSED})" \
  "~$(fmt_time ${T4_EXTRAP%.*})" \
  "compressed"

echo
echo "Note: routing script needs updating to read .zst files if strategy 4 is chosen"
echo "      uncompressed (strategy 2) requires ~120GB temp disk space for both R1+R2"
echo

# Cleanup
echo "Cleaning up benchmark files..."
rm -f "${BENCH_DIR}"/chunk_* "${BENCH_DIR}/test_input.fastq"
echo "Done."
echo "=============================================="