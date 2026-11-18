"""从本地运行目录回放策略（上游 play.py 的离线版）。

  uv run python tools/play_local.py logs/SuperMimic-GenCoord-G1 [--viewer viser]

viser 查看器会在 http://localhost:8080 打开网页，可切换动作、显示 world model 预测的
"幽灵"机器人；有显示器时可用 ``--viewer native``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.utils.torch import configure_torch_backends
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)

from local_run import build_from_run, resolve_run_dir


@dataclass
class Args:
  run_dir: tyro.conf.Positional[str]
  """运行目录，或 logs/<task> 父目录（自动取最新）。"""
  motion_file: str | None = None
  """覆盖训练时的动作文件（支持 glob / 逗号分隔）。"""
  initial_task: Literal["motion", "velocity"] = "motion"
  viewer: Literal["auto", "native", "viser"] = "viser"
  num_steps_rollout: int = 1000000
  device: str | None = None


def main(args: Args) -> None:
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  run_dir = resolve_run_dir(args.run_dir)
  print(f"[INFO] 运行目录: {run_dir}")

  def hook(env_cfg):
    env_cfg.observations["actor"].enable_corruption = True

  env, wm, actor, _, meta = build_from_run(
    run_dir, num_envs=1, device=device, motion_file=args.motion_file, env_cfg_hook=hook
  )
  motion_names = [Path(f).stem for f in meta["motion_files"]]
  viewer = args.viewer
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  common = dict(
    env=env,
    actor=actor,
    dynamics=wm.dynamics,
    representation=wm.representation,
    wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
    num_steps_rollout=args.num_steps_rollout,
    sim_context=wm.sim_context,
    task_specs=actor.task_specs,
    initial_task=args.initial_task,
  )
  if viewer == "native":
    SuperMimicNativeViewer(**common).run()
  else:
    SuperMimicViserViewer(motion_names=motion_names, **common).run()
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
