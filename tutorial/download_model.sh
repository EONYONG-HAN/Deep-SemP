#!/usr/bin/env bash
# Download the trained Deep-SemP student model from the COBI lab page.
# The model is hosted off-GitHub (too large for the repo); it is git-ignored here.
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://cobi.knu.ac.kr/tools/deepsemp/best_student_model.pt}"
OUT_DIR="${1:-models}"
OUT="${OUT_DIR}/best_student_model.pt"

mkdir -p "${OUT_DIR}"
echo "Downloading student model:"
echo "  from : ${MODEL_URL}"
echo "  to   : ${OUT}"

if command -v wget >/dev/null 2>&1; then
    wget -c -O "${OUT}" "${MODEL_URL}"
elif command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "${OUT}" "${MODEL_URL}"
else
    echo "ERROR: need wget or curl installed." >&2
    exit 1
fi

echo "Done. Model saved to ${OUT}"
