#!/bin/bash

# --- Deep-SemP: load user paths if provided (see configs/paths.example.sh) ---
if [ -n "${DEEPSEMP_CONFIG:-}" ] && [ -f "${DEEPSEMP_CONFIG}" ]; then
    # shellcheck disable=SC1090
    source "${DEEPSEMP_CONFIG}"
fi

# =============================================================
# BLAST evaluation — C. briggsae
# Tests: single sample (baseline + student) + merged (baseline + student)
# =============================================================
THREADS=30

# Reference CDS from WormBase
REF_TRANSCRIPTS="${REF_TRANSCRIPTS:-/data3/projects/2025_Assembly/eyh/c_briggsae/reference/c_briggsae_cdna.fa.gz}"

# Single sample assemblies
SINGLE_BASELINE="${SINGLE_BASELINE:-/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/baseline/05_final_coding.fasta}"
SINGLE_STUDENT="${SINGLE_STUDENT:-/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/student_stratB/05_final_coding.fasta}"

# Merged assemblies
MERGED_BASELINE="${MERGED_BASELINE:-/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/merged_baseline/05_final_coding.fasta}"
MERGED_STUDENT="${MERGED_STUDENT:-/data3/projects/2025_Assembly/eyh/c_briggsae/postprocess/merged_student_stratB/05_final_coding.fasta}"

# Output directory
OUTDIR="${OUTDIR:-/data3/projects/2025_Assembly/eyh/c_briggsae/blast_results}"
mkdir -p "${OUTDIR}"
cd "${OUTDIR}"

echo "=== 1. Building BLAST Database ==="
# Decompress if needed
gunzip -c "${REF_TRANSCRIPTS}" > c_briggsae_cdna.fa
makeblastdb -in c_briggsae_cdna.fa -dbtype nucl -out c_briggsae_ref_db
echo "  Database built"

blast_query() {
    local fasta=$1
    local outfile=$2
    local label=$3
    echo "=== BLASTing ${label} ==="
    blastn -query "${fasta}" \
           -db c_briggsae_ref_db \
           -num_threads "${THREADS}" \
           -evalue 1e-5 \
           -max_target_seqs 1 \
           -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen" \
           -out "${outfile}"
    echo "  Done: ${outfile}"
}

echo "=== 2. Running BLAST for all assemblies ==="
blast_query "${SINGLE_BASELINE}" "single_baseline_blast.tsv"  "Single Baseline"
blast_query "${SINGLE_STUDENT}"  "single_student_blast.tsv"   "Single Student"
blast_query "${MERGED_BASELINE}" "merged_baseline_blast.tsv"  "Merged Baseline"
blast_query "${MERGED_STUDENT}"  "merged_student_blast.tsv"   "Merged Student"

# Clean up uncompressed reference
rm -f c_briggsae_cdna.fa

echo "=== BLAST Complete ==="
echo "Results saved to: ${OUTDIR}"