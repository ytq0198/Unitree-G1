"""Course-local event helpers that keep small batched mjlab runs robust."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rows_for_envs(tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
  if tensor.shape[0] == 1 and num_envs > 1:
    return tensor.expand(num_envs, *tensor.shape[1:])
  return tensor


def reset_joints_by_offset_batched(
  env,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints while accepting one-row default/limit tensors.

  Some local MuJoCo-Warp builds expose joint limits as a single shared row even
  when the simulation has multiple worlds. The course tasks use this wrapper so
  small classroom batches work consistently on CPU and ``cuda:0``.
  """
  env_ids = resolve_env_ids(env, env_ids).long()

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  default_joint_pos = _rows_for_envs(default_joint_pos, env.num_envs)
  default_joint_vel = _rows_for_envs(default_joint_vel, env.num_envs)
  soft_joint_pos_limits = _rows_for_envs(soft_joint_pos_limits, env.num_envs)

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
