# Unitree G1 Complex-Terrain Navigation

Course project for training a Unitree G1 humanoid to navigate procedurally generated complex terrain with direct 29-DoF joint actions.

## Scope

- AMP-PPO locomotion and navigation
- Body-frame next-waypoint observations
- Height-scan baseline and optional depth policy
- Ten-random-start evaluation
- Smoothness and motion-style constraints

## Environment

- Python 3.11
- `mjlab==1.5.0`
- PyTorch with CUDA
- Conda environment on the training server: `/mnt/localDisk3/weizian/conda_envs/summer`

## Quick Checks

```bash
python -m unittest discover -s tests -v
python run_course_project.py smoke --mode height --device cuda:0
```

## Training And Evaluation

```bash
python run_course_project.py train --mode height --device cuda:0 \
  --num-envs 2048 --iterations 1000 --seed 23

python run_course_project.py eval --mode height --device cuda:0 \
  --checkpoint outputs/rsl_rl/<run>/model_<iteration>.pt \
  --evaluations 10 --eval-steps 5000 --seed 101
```

The evolving design and verified results are recorded in:

- `experiment_design.md`
- `experiment_report.md`

Training checkpoints, videos, TensorBoard logs, and generated submissions are intentionally excluded from Git.
