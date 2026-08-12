#!/usr/bin/env bash
set -uo pipefail

ROOT="/mnt/localDisk3/weizian/unitree-g1-work"
LOG_DIR="$ROOT/outputs/job_logs"
LOCK_DIR="$LOG_DIR/h5_waiter.lock"
STATUS="$LOG_DIR/h5_waiter.status"
TRAIN_LOG="$LOG_DIR/h5_velocity_tracking.log"
CONDA_PYTHON="/mnt/localDisk3/weizian/conda_envs/summer/bin/python"

mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "An H5 waiter is already active." >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

echo "waiting $(date --iso-8601=seconds)" > "$STATUS"
while true; do
  candidate="$({
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  } | awk -F, '{gsub(/ /, "", $0); if ($2 < 2048 && $3 < 10) print $1;}' \
    | head -n 1)"
  if [[ -n "$candidate" ]]; then
    break
  fi
  sleep 60
done

echo "started gpu=$candidate $(date --iso-8601=seconds)" > "$STATUS"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$candidate" \
MUJOCO_GL=egl \
XDG_CACHE_HOME="$ROOT/.cache" \
"$CONDA_PYTHON" -u run_course_project.py train \
  --mode height \
  --device cuda:0 \
  --num-envs 64 \
  --iterations 500 \
  --steps-per-env 24 \
  --seed 23 \
  --init-checkpoint outputs/warmstart/lab7_height_to_navigation.pt \
  --navigation-weight 1 \
  --velocity-tracking-weight 2 \
  --smoothness-weight -0.05 \
  --amp-scale 0.5 \
  --learning-rate 0.0001 \
  > "$TRAIN_LOG" 2>&1
exit_code=$?
echo "finished exit=$exit_code gpu=$candidate $(date --iso-8601=seconds)" > "$STATUS"
exit "$exit_code"
