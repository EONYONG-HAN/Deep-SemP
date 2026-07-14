#!/usr/bin/env bash
set -euo pipefail

# =========================
# User settings
# =========================
GPU_ID=2
MAX_JOBS=3

PYTHON_BIN=python
SCRIPT=distill_deepsemp.py

TEACHER_WEIGHTS="/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/best_model.pt"
DATA_PATH="/data3/projects/2025_Assembly/eyh/c_elegans/training_data/simulation_data_full.csv"

BASE_OUT="/data3/projects/2025_Assembly/eyh/models/checkpoints/track_a_direct_full/distilled_models/weekend_grid"
mkdir -p "${BASE_OUT}"

COMMON_ARGS=(
  --teacher_weights "${TEACHER_WEIGHTS}"
  --data_path "${DATA_PATH}"
  --epochs 20
  --batch_size 128
  --d_model 384
  --nhead 8
  --num_layers 6
  --alpha 0.3
  --temperature 3.0
  --masked_pooling
  --stratify_split
  --warmup_ratio 0.05
)

# =========================
# Define runs
# Format:
# run_name|extra args...
# =========================
RUNS=(
  "run_dm384_l6_a03_t3_mask|"
  "run_dm384_l4_a03_t3_mask|--num_layers 4"
  "run_dm384_l6_a05_t3_mask|--alpha 0.5"
  "run_dm384_l6_a03_t2_mask|--temperature 2.0"
  "run_dm384_l6_a03_t4_mask|--temperature 4.0"
)

# =========================
# Launch function
# =========================
launch_run() {
  local run_name="$1"
  shift
  local extra_args=("$@")

  local out_dir="${BASE_OUT}/${run_name}"
  local log_file="${BASE_OUT}/${run_name}.log"

  mkdir -p "${out_dir}"

  echo "[$(date '+%F %T')] START ${run_name}"
  echo "  out_dir : ${out_dir}"
  echo "  log     : ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${out_dir}" \
    --log_file "${log_file}" \
    "${extra_args[@]}" \
    > "${log_file}.launcher_stdout" 2>&1 &

  local pid=$!
  echo "  pid     : ${pid}"
}

# =========================
# Main scheduler
# =========================
echo "===== Weekend Distillation Grid ====="
echo "GPU_ID   : ${GPU_ID}"
echo "MAX_JOBS : ${MAX_JOBS}"
echo "BASE_OUT : ${BASE_OUT}"
echo

pids=()

for entry in "${RUNS[@]}"; do
  run_name="${entry%%|*}"
  extra="${entry#*|}"

  # shellcheck disable=SC2206
  extra_args=( $extra )

  # Wait until fewer than MAX_JOBS background jobs are running
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 20
  done

  launch_run "${run_name}" "${extra_args[@]}"
done

echo
echo "All jobs submitted. Waiting for completion..."
wait
echo "[$(date '+%F %T')] All runs finished."