"""Register the Course Project height and depth navigation tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import course_g1_navigation_env_cfg
from .rl_cfg import course_g1_navigation_ppo_runner_cfg

register_mjlab_task(
  task_id="Course-Project-G1-Navigation-AMP-Height",
  env_cfg=course_g1_navigation_env_cfg("height"),
  play_env_cfg=course_g1_navigation_env_cfg("height", play=True),
  rl_cfg=course_g1_navigation_ppo_runner_cfg("height"),
)

register_mjlab_task(
  task_id="Course-Project-G1-Navigation-AMP-Depth",
  env_cfg=course_g1_navigation_env_cfg("depth"),
  play_env_cfg=course_g1_navigation_env_cfg("depth", play=True),
  rl_cfg=course_g1_navigation_ppo_runner_cfg("depth"),
)
