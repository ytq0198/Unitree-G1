#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: train_stage.sh GPU PHASE SEED ITERATIONS TAG [NUM_ENVS] [CHECKPOINT]" >&2
  exit 2
fi

gpu="$1"
phase="$2"
seed="$3"
iterations="$4"
tag="$5"
num_envs="${6:-512}"
checkpoint="${7:-}"

workspace="/mnt/localDisk3/weizian/unitree-g1-work"
experiment="$workspace/locomotion_pretrain"
python_bin="/mnt/localDisk3/weizian/conda_envs/summer/bin/python"
log="$workspace/outputs/job_logs/$tag.log"

cd "$experiment"
export COURSE_LAB4_WORKSPACE="/mnt/localDisk3/weizian/RL learning/lab4"
export CUDA_VISIBLE_DEVICES="$gpu"
export MUJOCO_GL="egl"
export XDG_CACHE_HOME="$workspace/.cache"

args=(
  train
  --mode height
  --phase "$phase"
  --device cuda:0
  --num-envs "$num_envs"
  --iterations "$iterations"
  --steps-per-env 24
  --seed "$seed"
)
if [ -n "$checkpoint" ]; then
  args+=(--checkpoint "$checkpoint")
fi

exec "$python_bin" -u run_lab7.py "${args[@]}" >"$log" 2>&1
