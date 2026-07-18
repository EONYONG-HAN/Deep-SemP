#!/usr/bin/env bash
# Create a small (~few MB) paired-end toy dataset for the quickstart by
# subsampling a full C. elegans FASTQ. Run once, then commit the toy files.
#
# Usage:
#   R1=/path/SRR36278209_1.fastq.gz R2=/path/SRR36278209_2.fastq.gz \
#     N=20000 bash tutorial/make_toy_data.sh
set -euo pipefail

R1="${R1:?set R1 to a full R1 FASTQ(.gz)}"
R2="${R2:?set R2 to a full R2 FASTQ(.gz)}"
N="${N:-20000}"          # number of read PAIRS to sample (~few MB gzipped)
SEED="${SEED:-42}"
OUT="${OUT:-tutorial/data}"
mkdir -p "${OUT}"

if command -v seqtk >/dev/null 2>&1; then
    echo "Subsampling ${N} read pairs with seqtk (seed ${SEED})..."
    seqtk sample -s"${SEED}" "${R1}" "${N}" | gzip > "${OUT}/toy_R1.fastq.gz"
    seqtk sample -s"${SEED}" "${R2}" "${N}" | gzip > "${OUT}/toy_R2.fastq.gz"
else
    echo "seqtk not found; falling back to the first ${N} read pairs (deterministic)."
    L=$(( N * 4 ))
    zcat -f "${R1}" | head -n "${L}" | gzip > "${OUT}/toy_R1.fastq.gz"
    zcat -f "${R2}" | head -n "${L}" | gzip > "${OUT}/toy_R2.fastq.gz"
fi

echo "Wrote:"
ls -lh "${OUT}"/toy_R1.fastq.gz "${OUT}"/toy_R2.fastq.gz
echo "Commit these two files so users get the toy data with the repo."
