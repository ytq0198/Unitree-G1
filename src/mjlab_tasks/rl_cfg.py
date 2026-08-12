"""AMP-PPO configuration for direct G1 navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env_cfgs import DEFAULT_MOTION_PATH, DEFAULT_STUDENT_PATH, ObservationMode

_VISION_MODEL = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
_VISION_CNN = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}


@dataclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "src.amp.ppo:AmpPPO"
  motion_path: str = str(DEFAULT_MOTION_PATH)
  student_path: str = str(DEFAULT_STUDENT_PATH)
  step_dt: float = 0.02
  amp_observation_key: str = "amp"
  amp_reward_scale: float = 0.5
  discriminator_learning_rate: float = 1.0e-4
  discriminator_batch_size: int = 256
  discriminator_updates: int = 2
  replay_capacity: int = 50_000


def course_g1_navigation_ppo_runner_cfg(
  observation_mode: ObservationMode = "height",
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  motion_path: str | Path = DEFAULT_MOTION_PATH,
  amp_reward_scale: float = 0.5,
  learning_rate: float = 1.0e-3,
) -> RslRlOnPolicyRunnerCfg:
  obs_groups: dict[str, tuple[str, ...]]
  if observation_mode == "depth":
    actor = RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN,
      class_name=_VISION_MODEL,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    )
    obs_groups = {"actor": ("actor", "depth"), "critic": ("critic",)}
  else:
    actor = RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    )
    obs_groups = {"actor": ("actor",), "critic": ("critic",)}

  algorithm = AmpPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=learning_rate,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.02,
    max_grad_norm=1.0,
    motion_path=str(Path(motion_path).resolve()),
    student_path=str(Path(student_path).resolve()),
    amp_reward_scale=amp_reward_scale,
  )
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=RslRlModelCfg(
      hidden_dims=(256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=algorithm,
    obs_groups=obs_groups,
    experiment_name=f"course_project_navigation_amp_{observation_mode}",
    run_name="course_project",
    logger="tensorboard",
    upload_model=False,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1000,
  )
