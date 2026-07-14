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
GPU_ID=3
MAX_JOBS=2

PYTHON_BIN=python
SCRIPT="${SCRIPT:-$(dirname "$0")/distill_student.py}"

TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/best_model.pt}"
DATA_PATH="${DATA_PATH:-/data3/projects/2025_Assembly/eyh/c_elegans/training_data/simulation_data_illumina5_train.csv}"
SIMVAL_PATH="${SIMVAL_PATH:-/data3/projects/2025_Assembly/eyh/c_elegans/training_data/simulation_data_illumina5_simval.csv}"

BASE_OUT="${BASE_OUT:-/data3/projects/2025_Assembly/eyh/c_elegans/models/checkpoints/track_a_direct_full/distilled_models/illumina5_grid}"
mkdir -p "${BASE_OUT}"

COMMON_ARGS=(
  --teacher_weights "${TEACHER_WEIGHTS}"
  --data_path       "${DATA_PATH}"
  --simval_path     "${SIMVAL_PATH}"
  --epochs          20
  --batch_size      768
  --nhead           8
  --temperature     2.0
  --alpha           0.3
  --masked_pooling
  --stratify_split
  --warmup_ratio    0.05
)

# =========================
# Define runs
# Format: run_name|extra args
#
# Runs 1-2: RESUME from epoch 15 checkpoint
#           (killed during epoch 16, resume_checkpoint.pt saved at ep15)
# Runs 3-5: FRESH start (not yet started)
# =========================
RUNS=(
  # --- RESUME: epoch 15 → 20 (weights only, scheduler restarts at lower LR) ---
  "d384_l8_a03_t2_no_hidden|--d_model 384 --num_layers 8 --dim_feedforward 1024 --no_hidden_distill --load_checkpoint ${BASE_OUT}/d384_l8_a03_t2_no_hidden/best_student_model.pt --start_epoch 15 --lr 1e-4"
  "d384_l8_a03_t2_hidden_b01|--d_model 384 --num_layers 8 --dim_feedforward 1024 --beta 0.1 --load_checkpoint ${BASE_OUT}/d384_l8_a03_t2_hidden_b01/best_student_model.pt --start_epoch 15 --lr 1e-4"

  # --- FRESH: full 20 epochs ---
  "d384_l8_a03_t2_hidden_b05|--d_model 384 --num_layers 8 --dim_feedforward 1024 --beta 0.5"
  "d512_l8_a03_t2_no_hidden|--d_model 512 --num_layers 8 --dim_feedforward 2048 --no_hidden_distill"
  "d512_l8_a03_t2_hidden_b01|--d_model 512 --num_layers 8 --dim_feedforward 2048 --beta 0.1"
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

  # Check if this is a resume run
  local resume_note=""
  for arg in "${extra_args[@]}"; do
    if [[ "${arg}" == *"resume_checkpoint.pt" ]]; then
      if [ -f "${arg}" ]; then
        resume_note=" [RESUME from checkpoint]"
      else
        echo "[WARN] Checkpoint not found: ${arg}"
        echo "       Falling back to best_student_model.pt if available..."
        # Try fallback to best_student_model.pt
        local fallback="${out_dir}/best_student_model.pt"
        if [ -f "${fallback}" ]; then
          echo "       Using fallback: ${fallback}"
          resume_note=" [RESUME from best_student_model.pt — scheduler will restart]"
        else
          echo "[ERROR] No checkpoint found for ${run_name}. Skipping."
          return 1
        fi
      fi
    fi
  done

  echo "[$(date '+%F %T')] START ${run_name}${resume_note}"
  echo "  out_dir : ${out_dir}"
  echo "  log     : ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --output_dir "${out_dir}" \
    --log_file   "${log_file}" \
    "${extra_args[@]}" \
    > "${log_file}.launcher_stdout" 2>&1 &

  local pid=$!
  echo "  pid     : ${pid}"
}

# =========================
# Verify resume checkpoints exist before starting
# =========================
echo "===== Pre-flight checkpoint check ====="
RESUME_RUNS=("d384_l8_a03_t2_no_hidden" "d384_l8_a03_t2_hidden_b01")
ALL_OK=true

for run_name in "${RESUME_RUNS[@]}"; do
  ckpt="${BASE_OUT}/${run_name}/resume_checkpoint.pt"
  best="${BASE_OUT}/${run_name}/best_student_model.pt"
  if [ -f "${ckpt}" ]; then
    echo "  [OK] ${run_name} — resume_checkpoint.pt found (full state restore)"
  elif [ -f "${best}" ]; then
    echo "  [OK] ${run_name} — best_student_model.pt found (weights only, lr=3e-6 applied)"
  else
    echo "  [ERROR] ${run_name} — no checkpoint found at all!"
    ALL_OK=false
  fi
done

if [ "${ALL_OK}" = false ]; then
  echo "ERROR: Some resume checkpoints are missing. Aborting."
  exit 1
fi
echo

# =========================
# Main loop
# =========================
echo "===== Illumina5 Distillation Grid (Resume + Fresh) ====="
echo "GPU_ID   : ${GPU_ID}"
echo "MAX_JOBS : ${MAX_JOBS}"
echo "BASE_OUT : ${BASE_OUT}"
echo "Runs     : ${#RUNS[@]}"
echo

for entry in "${RUNS[@]}"; do
  run_name="${entry%%|*}"
  extra="${entry#*|}"

  # shellcheck disable=SC2206
  extra_args=( $extra )

  while [ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 30
  done

  launch_run "${run_name}" "${extra_args[@]}"
done

echo
echo "All jobs submitted. Waiting for completion..."
wait
echo "[$(date '+%F %T')] All runs finished."

# =========================
# Summary
# =========================
echo
echo "===== Results Summary ====="
printf "%-40s | %-12s | %-12s | %-12s\n" "Run" "Student Acc" "Agreement" "SimVal Acc"
printf "%-40s-+-%-12s-+-%-12s-+-%-12s\n" "$(printf '%0.s-' {1..40})" "------------" "------------" "------------"

for entry in "${RUNS[@]}"; do
  run_name="${entry%%|*}"
  metrics_file="${BASE_OUT}/${run_name}/best_metrics.txt"
  simval_file="${BASE_OUT}/${run_name}/simval_metrics.txt"

  if [ -f "${metrics_file}" ]; then
    student_acc=$(grep "student_acc" "${metrics_file}" | awk '{print $2}')
    agreement=$(grep "agreement_rate" "${metrics_file}" | awk '{print $2}')
  else
    student_acc="N/A"
    agreement="N/A"
  fi

  if [ -f "${simval_file}" ]; then
    simval_acc=$(grep "student_acc" "${simval_file}" | awk '{print $2}')
  else
    simval_acc="N/A"
  fi

  printf "%-40s | %-12s | %-12s | %-12s\n" "${run_name}" "${student_acc}" "${agreement}" "${simval_acc}"
done