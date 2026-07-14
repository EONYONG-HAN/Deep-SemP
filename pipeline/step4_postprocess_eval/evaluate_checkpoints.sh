#!/usr/bin/env bash
set -euo pipefail

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

# ── Config ─────────────────────────────────────────────────────
LABEL=$1
OUTDIR=$2

CPC2="${CPC2:-/home/EYH/CPC2_standalone-1.0.1/bin/CPC2.py}"
REF_DNA="${REF_DNA:-/data3/projects/2025_Assembly/eyh/h_sapiens/reference/h_sapiens_dna.fa.gz}"
REF_GTF="${REF_GTF:-/data3/projects/2025_Assembly/eyh/h_sapiens/reference/h_sapiens.gtf}"
R1="${R1:-/data3/projects/2025_Assembly/eyh/h_sapiens/raw_data/h_sapiens_R1.fastq.gz}"
R2="${R2:-/data3/projects/2025_Assembly/eyh/h_sapiens/raw_data/h_sapiens_R2.fastq.gz}"

THREADS=30
BUSCO_LINEAGE="mammalia_odb10"

FILTERED_FASTA="${OUTDIR}/04_filtered_300bp.fasta"
DEDUP_FASTA="${OUTDIR}/04_filtered_300bp_dedup.fasta"
FASTA_B="${OUTDIR}/05_final_coding.fasta"
RESULTS_TSV="${OUTDIR}/evaluation_results.tsv"

[ -f "${REF_GTF}" ]        || { echo "ERROR: GTF not found";           exit 1; }
[ -f "${FILTERED_FASTA}" ] || { echo "ERROR: filtered FASTA not found"; exit 1; }

record() {
    echo -e "${LABEL}\t${1}\t${2}\t${3}" >> "${RESULTS_TSV}"
}

run_busco() {
    local fasta=$1 checkpoint=$2 work_dir=$3
    echo "  [BUSCO] ${checkpoint} (${BUSCO_LINEAGE})..."
    busco -i "${fasta}" -l "${BUSCO_LINEAGE}" \
        -o "busco_${checkpoint}" -m tran \
        --cpu "${THREADS}" --out_path "${work_dir}" -f \
        > "${work_dir}/busco.log" 2>&1
    local summary="${work_dir}/busco_${checkpoint}/short_summary.specific.${BUSCO_LINEAGE}.busco_${checkpoint}.txt"
    if [ -f "${summary}" ]; then
        cat "${summary}"
        local complete=$(grep "Complete BUSCOs"           "${summary}" | grep -oP '\d+(?= Complete)')
        local single=$(grep "Complete and single-copy"    "${summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local duplicated=$(grep "Complete and duplicated" "${summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local fragmented=$(grep "Fragmented"              "${summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        local missing=$(grep "Missing"                    "${summary}" | grep -oP '^\s*\d+' | tr -d ' ')
        echo "  BUSCO: C=${complete} S=${single} D=${duplicated} F=${fragmented} M=${missing}"
        record "${checkpoint}" "busco_complete"   "${complete}"
        record "${checkpoint}" "busco_single"     "${single}"
        record "${checkpoint}" "busco_duplicated" "${duplicated}"
        record "${checkpoint}" "busco_fragmented" "${fragmented}"
        record "${checkpoint}" "busco_missing"    "${missing}"
    else
        echo "  [WARN] BUSCO summary not found — check ${work_dir}/busco.log"
    fi
}

run_gffcompare() {
    local fasta=$1 checkpoint=$2 work_dir=$3
    echo "  [GffCompare] ${checkpoint}..."
    local ref_dna_unzipped="${work_dir}/ref_dna.fa"
    gunzip -c "${REF_DNA}" > "${ref_dna_unzipped}"
    minimap2 -ax splice -t "${THREADS}" -uf \
        "${ref_dna_unzipped}" "${fasta}" \
        > "${work_dir}/mapped.sam" 2>"${work_dir}/minimap2.log"
    samtools view -bS "${work_dir}/mapped.sam" | \
        samtools sort -@ "${THREADS}" -o "${work_dir}/mapped.bam"
    samtools index "${work_dir}/mapped.bam"
    stringtie "${work_dir}/mapped.bam" \
        -o "${work_dir}/assembled.gtf" \
        -p "${THREADS}" > "${work_dir}/stringtie.log" 2>&1
    gffcompare -r "${REF_GTF}" \
        -o "${work_dir}/gffcmp" \
        "${work_dir}/assembled.gtf" > "${work_dir}/gffcompare.log" 2>&1
    local stats="${work_dir}/gffcmp.stats"
    if [ -f "${stats}" ]; then
        cat "${stats}"
        local sensitivity=$(grep "Sensitivity" "${stats}" | head -1 | grep -oP '[0-9.]+' | head -1)
        local precision=$(grep "Precision"    "${stats}" | head -1 | grep -oP '[0-9.]+' | head -1)
        echo "  GffCompare: Sensitivity=${sensitivity}% Precision=${precision}%"
        record "${checkpoint}" "gffcmp_sensitivity" "${sensitivity}"
        record "${checkpoint}" "gffcmp_precision"   "${precision}"
    else
        echo "  [WARN] GffCompare stats not found"
    fi
    rm -f "${work_dir}/mapped.sam" "${ref_dna_unzipped}"
}

# ── Dedup ──────────────────────────────────────────────────────
echo "[Dedup] Renaming duplicate IDs..."
awk '/^>/ {
    id = $1; rest = ""
    for (i=2; i<=NF; i++) rest = rest " " $i
    count[id]++
    if (count[id] > 1) id = id "_dup" count[id]
    print id rest; next
}
{ print }' "${FILTERED_FASTA}" > "${DEDUP_FASTA}"
echo "  Contigs: $(grep -c '^>' ${DEDUP_FASTA})"
DUPS=$(grep '^>' "${DEDUP_FASTA}" | awk '{print $1}' | sort | uniq -d | wc -l)
echo "  Remaining duplicates: ${DUPS}"

# ── Checkpoint A ───────────────────────────────────────────────
echo "=============================================="
echo "  CHECKPOINT A: After length filter"
echo "=============================================="
WORK_A="${OUTDIR}/eval_A_after_length_filter"
mkdir -p "${WORK_A}"
run_busco      "${DEDUP_FASTA}" "A_after_length_filter" "${WORK_A}"
run_gffcompare "${DEDUP_FASTA}" "A_after_length_filter" "${WORK_A}"

# ── CPC2 — always rerun, remove stale output first ─────────────
echo "[Step 5] CPC2 coding potential filter..."
CPC2_OUT="${OUTDIR}/05_cpc2_results"
CODING_IDS="${OUTDIR}/05_coding_ids.txt"

# Force rerun by removing previous outputs
rm -f "${CPC2_OUT}.txt" "${CODING_IDS}" "${FASTA_B}"

python "${CPC2}" -i "${DEDUP_FASTA}" -o "${CPC2_OUT}" > "${OUTDIR}/05_cpc2.log" 2>&1

if [ ! -f "${CPC2_OUT}.txt" ]; then
    echo "ERROR: CPC2 failed — check ${OUTDIR}/05_cpc2.log"
    exit 1
fi

awk '$8 == "coding" {print $1}' "${CPC2_OUT}.txt" > "${CODING_IDS}"
awk 'NR==FNR { a[">"$1]; next }
     /^>/    { f = ($1 in a) }
     f        { print }' \
    "${CODING_IDS}" "${DEDUP_FASTA}" > "${FASTA_B}"

CODING_COUNT=$(grep -c "^>" "${FASTA_B}")
echo "  Coding contigs: ${CODING_COUNT} / $(grep -c '^>' ${DEDUP_FASTA})"
record "cpc2" "contig_count" "${CODING_COUNT}"

if [ "${CODING_COUNT}" -eq 0 ]; then
    echo "ERROR: No coding contigs — check CPC2 output"
    exit 1
fi

# ── Checkpoint B ───────────────────────────────────────────────
echo "=============================================="
echo "  CHECKPOINT B: After CPC2 coding filter"
echo "=============================================="
WORK_B="${OUTDIR}/eval_B_after_cpc2"
mkdir -p "${WORK_B}"
run_busco      "${FASTA_B}" "B_after_cpc2" "${WORK_B}"
run_gffcompare "${FASTA_B}" "B_after_cpc2" "${WORK_B}"

echo "=============================================="
echo "  DONE — ${LABEL}"
echo "  Results: ${RESULTS_TSV}"
echo "=============================================="