#!/usr/bin/env bash
set -euo pipefail

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

BUCKET_DIR="${BUCKET_DIR:-/data3/projects/2025_Assembly/eyh/c_briggsae/partitioned_reads/student_nh_illumina5_merged}"
BASE_OUT="${BASE_OUT:-/data3/projects/2025_Assembly/eyh/c_briggsae/assemblies/final_benchmark_merged}"
COMPLEXITY_TSV="${COMPLEXITY_TSV:-/data3/projects/2025_Assembly/eyh/c_briggsae/assemblies/bucket_complexity/merged_bucket_complexity.tsv}"
NUM_BUCKETS=50
MAX_CORES=40
MIN_SIZE=100000

mkdir -p "${BASE_OUT}"

run_bucket() {
    local bucket_idx=$1
    local cpus=$2
    local out_dir=$3
    local timing_log=$4

    local pad=$(printf "%02d" ${bucket_idx})
    local r1="${BUCKET_DIR}/bucket_${pad}_R1.fastq"
    local r2="${BUCKET_DIR}/bucket_${pad}_R2.fastq"
    local mem=$((cpus * 3))G
    local final="${out_dir}/Trinity_bucket_${pad}.fasta"
    local trinity_out="${out_dir}/trinity_bucket_${pad}"
    local log="${out_dir}/log_bucket_${pad}.txt"

    [ -f "${final}" ] && { echo "  [SKIP] Bucket ${pad}"; return; }
    [ ! -f "${r1}" ]  && { echo "  [SKIP] Bucket ${pad} — file missing"; return; }
    local sz=$(stat -c%s "${r1}")
    [ "${sz}" -lt "${MIN_SIZE}" ] && { echo "  [SKIP] Bucket ${pad} — too small"; return; }

    local t_start=$(date +%s)
    local t_start_str=$(date '+%F %T')
    echo "  [RUN]  Bucket ${pad} | CPUs=${cpus} | mem=${mem} | $(du -sh ${r1} | cut -f1)"

    Trinity \
        --seqType    fq \
        --left       "${r1}" \
        --right      "${r2}" \
        --CPU        "${cpus}" \
        --max_memory "${mem}" \
        --output     "${trinity_out}" \
        --full_cleanup \
        > "${log}" 2>&1

    local t_end=$(date +%s)
    local t_end_str=$(date '+%F %T')
    local elapsed_min=$(echo "scale=2; (${t_end} - ${t_start}) / 60" | bc)

    if [ -f "${trinity_out}.Trinity.fasta" ]; then
        mv "${trinity_out}.Trinity.fasta" "${final}"
        rm -rf "${trinity_out}"
        echo "  [OK]   Bucket ${pad} — ${elapsed_min} min (${cpus} CPUs) "
        echo -e "${pad}\t${cpus}\t${t_start_str}\t${t_end_str}\t${elapsed_min}\tSUCCESS" \
            >> "${timing_log}"
    else
        echo "  [FAIL] Bucket ${pad} — check ${log}"
        echo -e "${pad}\t${cpus}\t${t_start_str}\t${t_end_str}\t${elapsed_min}\tFAILED" \
            >> "${timing_log}"
    fi
}

# ── Strategy B ─────────────────────────────────────────────────
out_dir="${BASE_OUT}/strategyB_complexity_aware_40cpu"
timing_log="${out_dir}/timing.tsv"
mkdir -p "${out_dir}"
echo -e "bucket\tcpus\tstart\tend\telapsed_min\tstatus" > "${timing_log}"

echo "=============================================="
echo "  STRATEGY B: Complexity-Aware Parallel"
echo "  C. briggsae — Student Model Routing"
echo "  Max total cores: ${MAX_CORES}"
echo "=============================================="

[ ! -f "${COMPLEXITY_TSV}" ] && { echo "ERROR: Run complexity script first."; exit 1; }

declare -A BUCKET_CPUS
declare -A BUCKET_ORDER

while IFS=$'\t' read -r bucket sz_bytes sz_human total distinct unique \
                        complexity unique_ratio difficulty cpu_score; do
    [ "${bucket}" = "bucket" ] && continue
    idx=$((10#${bucket}))

    cpus=4
    score=$(echo "${cpu_score}" | awk '{printf "%.4f", $1}')
    if   (( $(echo "${score} > 12" | bc -l) )); then cpus=32
    elif (( $(echo "${score} > 9"  | bc -l) )); then cpus=20
    elif (( $(echo "${score} > 6"  | bc -l) )); then cpus=12
    elif (( $(echo "${score} > 4"  | bc -l) )); then cpus=8
    fi
    [ "${cpus}" -gt "${MAX_CORES}" ] && cpus=${MAX_CORES}

    BUCKET_CPUS[${idx}]=${cpus}
    BUCKET_ORDER[${idx}]=${cpu_score}
done < "${COMPLEXITY_TSV}"

for idx in "${!BUCKET_ORDER[@]}"; do
    echo "${BUCKET_ORDER[$idx]} ${idx}"
done | sort -rn | awk '{print $2}' > /tmp/sorted_by_complexity_briggsae.txt

wall_start=$(date +%s)
running_cores=0
declare -A job_pids
declare -A job_cores_map
count=0

while IFS= read -r i; do
    pad=$(printf "%02d" ${i})
    f="${BUCKET_DIR}/bucket_${pad}_R1.fastq"
    [ ! -f "${f}" ] && continue
    sz=$(stat -c%s "${f}")
    [ "${sz}" -lt "${MIN_SIZE}" ] && continue

    cpus=${BUCKET_CPUS[$i]:-4}

    while true; do
        for pid in "${!job_pids[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                freed=${job_cores_map[${pid}]}
                running_cores=$((running_cores - freed))
                unset job_pids[${pid}]
                unset job_cores_map[${pid}]
            fi
        done
        if [ $((running_cores + cpus)) -le "${MAX_CORES}" ]; then
            break
        fi
        sleep 10
    done

    count=$((count+1))
    echo "  [${count}] Bucket ${pad} | CPUs=${cpus} | cores_used=$((running_cores+cpus)) | $(date '+%H:%M:%S')"
    run_bucket "${i}" "${cpus}" "${out_dir}" "${timing_log}" &
    pid=$!
    job_pids[${pid}]=1
    job_cores_map[${pid}]=${cpus}
    running_cores=$((running_cores + cpus))
done < /tmp/sorted_by_complexity_briggsae.txt

wait
wall_end=$(date +%s)
wall_elapsed=$((wall_end - wall_start))

echo
echo "=============================================="
echo "  COMPLETE — C. briggsae Strategy B"
echo "  Wall time: $(echo "scale=2; ${wall_elapsed}/3600" | bc) hours"
echo "=============================================="

tail -n +2 "${timing_log}" | awk -F'\t' '
    BEGIN{sum=0;n=0;max=0}
    $6=="SUCCESS"{sum+=$5; n++; if($5+0>max) max=$5+0}
    END{
        printf "  Buckets assembled : %d\n", n
        printf "  Total CPU-minutes : %.1f\n", sum
        printf "  Slowest bucket    : %.2f min\n", max
    }'