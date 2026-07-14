#!/usr/bin/env bash
set -euo pipefail

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

BUCKET_DIR="${BUCKET_DIR:-/data3/projects/2025_Assembly/eyh/c_briggsae/partitioned_reads/student_nh_illumina5_merged}"
OUT_DIR="${OUT_DIR:-/data3/projects/2025_Assembly/eyh/c_briggsae/assemblies/bucket_complexity}"
NUM_BUCKETS=50
KMER_SIZE=25
THREADS=4
MAX_PARALLEL=8
MIN_SIZE=100000

mkdir -p "${OUT_DIR}"

OUTPUT_TSV="${OUT_DIR}/merged_bucket_complexity.tsv"
echo -e "bucket\tsize_bytes\tsize_human\ttotal_kmers\tdistinct_kmers\tunique_kmers\tcomplexity_ratio\tunique_ratio\tdifficulty\tcpu_score" \
    > "${OUTPUT_TSV}"

compute_complexity() {
    local idx=$1
    local pad=$(printf "%02d" ${idx})
    local r1="${BUCKET_DIR}/bucket_${pad}_R1.fastq"
    local jf="${OUT_DIR}/bucket_${pad}.jf"

    [ ! -f "${r1}" ] && return
    local sz=$(stat -c%s "${r1}")
    [ "${sz}" -lt "${MIN_SIZE}" ] && return

    local sz_human=$(du -sh "${r1}" | cut -f1)

    jellyfish count \
        -m "${KMER_SIZE}" \
        -s 500M \
        -t "${THREADS}" \
        -C \
        -o "${jf}" \
        "${r1}" 2>/dev/null

    local stats=$(jellyfish stats "${jf}")
    local total=$(echo "${stats}"    | awk '/^Total/{print $2}')
    local distinct=$(echo "${stats}" | awk '/^Distinct/{print $2}')
    local unique=$(echo "${stats}"   | awk '/^Unique/{print $2}')

    local complexity_ratio=$(echo "scale=4; ${distinct} / ${total}" | bc)
    local unique_ratio=$(echo "scale=4; ${unique} / ${distinct}" | bc)

    local difficulty=$(echo "scale=4; \
        ( (${complexity_ratio}^2) * 40 ) + \
        ( ${unique_ratio} * 3 ) + \
        ( l(${sz}/1000000+1)/l(2) * 0.3 )" | bc -l 2>/dev/null || echo "N/A")

    local cpu_score=$(echo "scale=4; \
        (${complexity_ratio} * 20) + \
        (l(${sz}/1000000+1)/l(2) * 0.8) + \
        (${unique_ratio} * 2)" | bc -l 2>/dev/null || echo "0")

    echo -e "${pad}\t${sz}\t${sz_human}\t${total}\t${distinct}\t${unique}\t${complexity_ratio}\t${unique_ratio}\t${difficulty}\t${cpu_score}" \
        >> "${OUTPUT_TSV}"

    echo "  Bucket ${pad} | size=${sz_human} | complexity=${complexity_ratio} | cpu_score=${cpu_score}"
    rm -f "${jf}"
}

echo "Computing k-mer complexity for C. briggsae (${NUM_BUCKETS} buckets)..."
for i in $(seq 0 $((NUM_BUCKETS-1))); do
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
        sleep 5
    done
    compute_complexity "${i}" &
done
wait

echo
echo "=============================================="
echo "  COMPLEXITY SUMMARY (sorted by cpu_score)"
echo "=============================================="
printf "  %-8s %-8s %-12s %-12s %-12s %-10s\n" \
    "Bucket" "Size" "Complexity" "Unique%" "CPU_Score" "Suggestion"
printf "  %-8s %-8s %-12s %-12s %-12s %-10s\n" \
    "------" "----" "----------" "-------" "---------" "----------"

tail -n +2 "${OUTPUT_TSV}" | sort -t$'\t' -k10 -rn | \
awk -F'\t' '{
    cpus = 4
    if ($10+0 > 12)     cpus = 32
    else if ($10+0 > 9) cpus = 20
    else if ($10+0 > 6) cpus = 12
    else if ($10+0 > 4) cpus = 8
    printf "  %-8s %-8s %-12s %-12s %-12s %d CPUs\n", \
        $1, $3, $7, $8, $10, cpus
}'

echo
echo "Saved: ${OUTPUT_TSV}"