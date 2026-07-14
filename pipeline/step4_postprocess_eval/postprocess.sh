#!/usr/bin/env bash
# =============================================================
# Post-processing + evaluation pipeline — C. briggsae
# =============================================================
set -euo pipefail

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

# =========================
# Tool paths
# =========================
CPC2="${CPC2:-/home/EYH/CPC2_standalone-1.0.1/bin/CPC2.py}"
THREADS=30
BUSCO_LINEAGE="nematoda_odb10"
MIN_LEN=300
CDHIT_ID=0.90

# =========================
# Parse arguments
# =========================
FASTA=""
FASTA_DIR=""
LABEL=""
OUTDIR=""
R1=""
R2=""
REF_DNA=""
REF_GTF=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fasta)     FASTA="$2";     shift 2 ;;
        --fasta_dir) FASTA_DIR="$2"; shift 2 ;;
        --label)     LABEL="$2";     shift 2 ;;
        --outdir)    OUTDIR="$2";    shift 2 ;;
        --r1)        R1="$2";        shift 2 ;;
        --r2)        R2="$2";        shift 2 ;;
        --ref_dna)   REF_DNA="$2";   shift 2 ;;
        --ref_gtf)   REF_GTF="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate required args
if [ -z "${LABEL}" ];   then echo "ERROR: --label required";           exit 1; fi
if [ -z "${R1}" ] || [ -z "${R2}" ]; then echo "ERROR: --r1 and --r2 required"; exit 1; fi
if [ -z "${REF_DNA}" ] || [ -z "${REF_GTF}" ]; then echo "ERROR: --ref_dna and --ref_gtf required"; exit 1; fi
if [ -z "${FASTA}" ] && [ -z "${FASTA_DIR}" ]; then echo "ERROR: --fasta or --fasta_dir required"; exit 1; fi

OUTDIR="${OUTDIR:-./postprocess_${LABEL}}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  Post-processing + Evaluation — C. briggsae"
echo "  Label    : ${LABEL}"
echo "  Output   : ${OUTDIR}"
echo "  Threads  : ${THREADS}"
echo "  Started  : $(date '+%F %T')"
echo "=============================================="
echo

WALL_START=$(date +%s)
RESULTS_TSV="${OUTDIR}/evaluation_results.tsv"
echo -e "label\tcheckpoint\tmetric\tvalue" > "${RESULTS_TSV}"

record() {
    local checkpoint=$1 metric=$2 value=$3
    echo -e "${LABEL}\t${checkpoint}\t${metric}\t${value}" >> "${RESULTS_TSV}"
}

# =========================
# Step 1: Merge bucket FASTAs
# prefix each contig with bucket ID to avoid duplicate names
# e.g. >TRINITY_DN44_c0_g1_i1 → >00_TRINITY_DN44_c0_g1_i1
# =========================
RAW_FASTA="${OUTDIR}/01_combined_raw.fasta"

if [ -n "${FASTA}" ]; then
    echo "[Step 1] Using single FASTA: ${FASTA}"
    cp "${FASTA}" "${RAW_FASTA}"
else
    echo "[Step 1] Merging bucket FASTAs with bucket ID prefix..."
    N=$(ls "${FASTA_DIR}"/Trinity_bucket_*.fasta 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "ERROR: No Trinity_bucket_*.fasta found in ${FASTA_DIR}"
        exit 1
    fi

    > "${RAW_FASTA}"  # empty the file first
    for f in "${FASTA_DIR}"/Trinity_bucket_*.fasta; do
        bucket=$(basename "${f}" .fasta | grep -oP '\d+$')
        awk -v b="${bucket}" '
            /^>/ { sub(/^>/, ">"b"_") }
            { print }
        ' "${f}" >> "${RAW_FASTA}"
    done

    echo "  Merged ${N} bucket FASTAs with bucket ID prefix"

    # Verify no duplicates remain
    DUPS=$(grep '^>' "${RAW_FASTA}" | awk '{print $1}' | sort | uniq -d | wc -l)
    echo "  Duplicate ID check: ${DUPS} duplicates found"
    if [ "${DUPS}" -gt 0 ]; then
        echo "  [WARN] Duplicate IDs still present — check bucket FASTA naming"
    fi
fi

RAW_COUNT=$(grep -c "^>" "${RAW_FASTA}")
echo "  Raw contigs: ${RAW_COUNT}"
record "raw" "contig_count" "${RAW_COUNT}"

# =========================
# Step 2: CD-HIT-EST (90%)
# =========================
CDHIT_FASTA="${OUTDIR}/02_cdhit90.fasta"
echo
echo "[Step 2] CD-HIT-EST (identity=${CDHIT_ID})..."
T2=$(date +%s)

cd-hit-est \
    -i  "${RAW_FASTA}" \
    -o  "${CDHIT_FASTA}" \
    -c  "${CDHIT_ID}" \
    -n  8 \
    -M  0 \
    -T  "${THREADS}" \
    > "${OUTDIR}/02_cdhit.log" 2>&1

CDHIT_COUNT=$(grep -c "^>" "${CDHIT_FASTA}")
echo "  After CD-HIT-EST: ${CDHIT_COUNT} contigs ($(($(date +%s)-T2))s)"
record "cdhit90" "contig_count" "${CDHIT_COUNT}"

# =========================
# Step 3: CAP3 elongation
# =========================
CAP3_FASTA="${OUTDIR}/03_cap3_combined.fasta"
echo
echo "[Step 3] CAP3 elongation..."
T3=$(date +%s)

cd "${OUTDIR}"
cap3 "02_cdhit90.fasta" -p 90 -o 40 > "03_cap3.log" 2>&1
cat "02_cdhit90.fasta.cap.contigs" "02_cdhit90.fasta.cap.singlets" > "03_cap3_combined.fasta"
cd - > /dev/null

CAP3_COUNT=$(grep -c "^>" "${CAP3_FASTA}")
echo "  After CAP3: ${CAP3_COUNT} contigs ($(($(date +%s)-T3))s)"
record "cap3" "contig_count" "${CAP3_COUNT}"

# =========================
# Step 4: Length filter >= 300bp
# =========================
FILTERED_FASTA="${OUTDIR}/04_filtered_300bp.fasta"
echo
echo "[Step 4] Length filter (>= ${MIN_LEN}bp)..."

awk -v min="${MIN_LEN}" '
    /^>/ {
        if (id != "" && length(seq) >= min) print id "\n" seq
        id=$0; seq=""
    }
    !/^>/ { seq = seq $0 }
    END   { if (id != "" && length(seq) >= min) print id "\n" seq }
' "${CAP3_FASTA}" > "${FILTERED_FASTA}"

FILTERED_COUNT=$(grep -c "^>" "${FILTERED_FASTA}")
echo "  After length filter: ${FILTERED_COUNT} contigs"
record "length_filter" "contig_count" "${FILTERED_COUNT}"

# Verify no duplicates after CAP3
DUPS=$(grep '^>' "${FILTERED_FASTA}" | awk '{print $1}' | sort | uniq -d | wc -l)
echo "  Duplicate ID check: ${DUPS} duplicates"
if [ "${DUPS}" -gt 0 ]; then
    echo "  [WARN] Unexpected duplicates after CAP3 — check pipeline"
fi

# =========================
# EVAL CHECKPOINT A
# =========================
echo
echo "=============================================="
echo "  CHECKPOINT A: After length filter"
echo "=============================================="

eval_checkpoint() {
    local fasta=$1
    local checkpoint_label=$2
    local work_dir="${OUTDIR}/eval_${checkpoint_label}"
    mkdir -p "${work_dir}"

    local count=$(grep -c "^>" "${fasta}")
    echo "  Input contigs: ${count}"
    record "${checkpoint_label}" "contig_count" "${count}"

    # --- Bowtie2 ---
    echo "  [A] Bowtie2 alignment..."
    local idx="${work_dir}/bt2_idx"
    bowtie2-build "${fasta}" "${idx}" > "${work_dir}/bt2_build.log" 2>&1
    bowtie2 \
        -p "${THREADS}" \
        -x "${idx}" \
        -1 "${R1}" -2 "${R2}" \
        --no-unal \
        -S /dev/null \
        2> "${work_dir}/bt2_report.txt"

    local align_rate=$(grep "overall alignment rate" "${work_dir}/bt2_report.txt" | \
        grep -oP '[0-9.]+(?=%)')
    echo "  Alignment rate: ${align_rate}%"
    record "${checkpoint_label}" "bowtie2_alignment_rate_pct" "${align_rate}"

    # --- BUSCO ---
    echo "  [B] BUSCO (${BUSCO_LINEAGE})..."
    busco \
        -i  "${fasta}" \
        -l  "${BUSCO_LINEAGE}" \
        -o  "busco_${checkpoint_label}" \
        -m  tran \
        --cpu "${THREADS}" \
        --out_path "${work_dir}" \
        -f \
        > "${work_dir}/busco.log" 2>&1

    local busco_summary="${work_dir}/busco_${checkpoint_label}/short_summary.specific.${BUSCO_LINEAGE}.busco_${checkpoint_label}.txt"
    if [ -f "${busco_summary}" ]; then
        local complete=$(grep "Complete BUSCOs"           "${busco_summary}" | grep -oP '\d+(?= Complete)')
        local single=$(grep "Complete and single"         "${busco_summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local duplicated=$(grep "Complete and duplicated" "${busco_summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local fragmented=$(grep "Fragmented"              "${busco_summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local missing=$(grep "Missing"                    "${busco_summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        echo "  BUSCO: C=${complete} S=${single} D=${duplicated} F=${fragmented} M=${missing}"
        record "${checkpoint_label}" "busco_complete"   "${complete}"
        record "${checkpoint_label}" "busco_single"     "${single}"
        record "${checkpoint_label}" "busco_duplicated" "${duplicated}"
        record "${checkpoint_label}" "busco_fragmented" "${fragmented}"
        record "${checkpoint_label}" "busco_missing"    "${missing}"
        cat "${busco_summary}"
    else
        echo "  [WARN] BUSCO summary not found — check ${work_dir}/busco.log"
    fi

    # --- GffCompare ---
    echo "  [C] GffCompare structural precision..."
    local ref_dna_unzipped="${work_dir}/ref_dna.fa"
    if [[ "${REF_DNA}" == *.gz ]]; then
        gunzip -c "${REF_DNA}" > "${ref_dna_unzipped}"
    else
        ln -sf "${REF_DNA}" "${ref_dna_unzipped}"
    fi

    minimap2 \
        -ax splice -t "${THREADS}" -uf \
        "${ref_dna_unzipped}" "${fasta}" \
        > "${work_dir}/mapped.sam" 2>"${work_dir}/minimap2.log"

    samtools view -bS "${work_dir}/mapped.sam" | \
        samtools sort -@ "${THREADS}" -o "${work_dir}/mapped.bam"
    samtools index "${work_dir}/mapped.bam"

    stringtie \
        "${work_dir}/mapped.bam" \
        -o "${work_dir}/assembled.gtf" \
        -p "${THREADS}" \
        > "${work_dir}/stringtie.log" 2>&1

    gffcompare \
        -r "${REF_GTF}" \
        -o "${work_dir}/gffcmp" \
        "${work_dir}/assembled.gtf" \
        > "${work_dir}/gffcompare.log" 2>&1

    local stats_file="${work_dir}/gffcmp.stats"
    if [ -f "${stats_file}" ]; then
        local sensitivity=$(grep "Sensitivity" "${stats_file}" | head -1 | \
            grep -oP '[0-9.]+' | head -1)
        local precision=$(grep "Precision" "${stats_file}" | head -1 | \
            grep -oP '[0-9.]+' | head -1)
        echo "  GffCompare: Sensitivity=${sensitivity}% Precision=${precision}%"
        record "${checkpoint_label}" "gffcmp_sensitivity" "${sensitivity}"
        record "${checkpoint_label}" "gffcmp_precision"   "${precision}"
        cat "${stats_file}"
    else
        echo "  [WARN] GffCompare stats not found — check ${work_dir}/gffcompare.log"
    fi

    rm -f "${work_dir}/mapped.sam" "${ref_dna_unzipped}"
}

eval_checkpoint "${FILTERED_FASTA}" "A_after_length_filter"

# =========================
# Step 5: CPC2 coding filter
# =========================
CPC2_OUT="${OUTDIR}/05_cpc2_results"
CODING_IDS="${OUTDIR}/05_coding_ids.txt"
CODING_FASTA="${OUTDIR}/05_final_coding.fasta"

echo
echo "[Step 5] CPC2 coding potential filter..."
T5=$(date +%s)

python "${CPC2}" \
    -i "${FILTERED_FASTA}" \
    -o "${CPC2_OUT}" \
    > "${OUTDIR}/05_cpc2.log" 2>&1

awk '$8 == "coding" {print $1}' "${CPC2_OUT}.txt" > "${CODING_IDS}"

awk 'NR==FNR { a[">"$1]; next }
     /^>/    { f = ($1 in a) }
     f        { print }' \
    "${CODING_IDS}" "${FILTERED_FASTA}" > "${CODING_FASTA}"

CODING_COUNT=$(grep -c "^>" "${CODING_FASTA}")
echo "  Coding contigs: ${CODING_COUNT} / $(grep -c "^>" "${FILTERED_FASTA}") ($(($(date +%s)-T5))s)"
record "cpc2" "contig_count" "${CODING_COUNT}"

# =========================
# EVAL CHECKPOINT B
# =========================
echo
echo "=============================================="
echo "  CHECKPOINT B: After CPC2 coding filter"
echo "=============================================="

eval_checkpoint "${CODING_FASTA}" "B_after_cpc2"

# =========================
# Final summary
# =========================
WALL_END=$(date +%s)
WALL_ELAPSED=$((WALL_END - WALL_START))

echo
echo "=============================================="
echo "  PIPELINE COMPLETE — ${LABEL}"
echo "  Wall time : $(echo "scale=2; ${WALL_ELAPSED}/3600" | bc) hours"
echo "=============================================="
echo
echo "Contig counts through pipeline:"
printf "  %-25s %s\n" "Stage" "Count"
printf "  %-25s %s\n" "-----" "-----"
for stage in "raw" "cdhit90" "cap3" "length_filter" "cpc2"; do
    count=$(grep -P "^${LABEL}\t${stage}\tcontig_count" "${RESULTS_TSV}" | \
        awk -F'\t' '{print $4}')
    printf "  %-25s %s\n" "${stage}" "${count:-N/A}"
done

echo
echo "Evaluation results saved to: ${RESULTS_TSV}"
echo "Full log: ${LOG}"