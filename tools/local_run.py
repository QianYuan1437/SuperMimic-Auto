"""本地运行目录（logs/<task>/<timestamp>/）的读写辅助。

上游 play/benchmark 只能从 W&B 云端拉取 runner 配置与权重。train_local.py 会在
本地日志目录里额外写入 ``run_meta.json``（runner 配置、任务名、动作文件），
这里负责把它读回来并构建 world model 与 actor，从而完全离线地回放/评估。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import supermimic.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  WorldModel,
  build_actor,
  build_world_model,
)

META_FILE = "run_meta.json"


def resolve_run_dir(path: str) -> Path:
  """接受具体的运行目录，或 ``logs/<task>`` 这样的父目录（取最新一次运行）。"""
  p = Path(os.path.expanduser(path))
  if (p / META_FILE).exists():
    return p
  runs = sorted(q.parent for q in p.glob(f"*/{META_FILE}"))
  if not runs:
    raise FileNotFoundError(f"{p} 下没有找到 {META_FILE}（是否用 train_local.py 训练？）")
  return runs[-1]


def load_meta(run_dir: Path) -> dict:
  return json.loads((run_dir / META_FILE).read_text())


def expand_motion_files(spec: str | list[str]) -> list[str]:
  if isinstance(spec, list):
    return spec
  if "," in spec:
    return [os.path.expanduser(s.strip()) for s in spec.split(",")]
  spec = os.path.expanduser(spec)
  if any(c in spec for c in "*?[]"):
    return sorted(glob.glob(spec))
  return [spec]


def build_from_run(
  run_dir: Path,
  *,
  num_envs: int,
  device: str,
  motion_file: str | None = None,
  play: bool = True,
  env_cfg_hook=None,
  render_mode: str | None = None,
) -> tuple[RslRlVecEnvWrapper, WorldModel, torch.nn.Module, SuperMimicRunnerCfg, dict]:
  """按运行目录重建 env、world model 与 actor，并加载权重。"""
  meta = load_meta(run_dir)
  task_id = meta["task_id"]
  runner_cfg = load_runner_cfg({"runner_cfg": meta["runner_cfg"]})
  check_representation_matches(task_id, runner_cfg.representation)

  env_cfg = load_env_cfg(task_id, play=play)
  env_cfg.scene.num_envs = num_envs
  files = expand_motion_files(motion_file or meta["motion_file"])
  if not files:
    raise ValueError(f"没有匹配的动作文件：{motion_file or meta['motion_file']}")
  env_cfg.commands["motion"].motion_file = files
  if env_cfg_hook is not None:
    env_cfg_hook(env_cfg)

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode))
  wm = build_world_model(runner_cfg, env, device)
  wm.load_bundle(torch.load(run_dir / "world_model.pt", map_location=device))
  wm.dynamics.eval()
  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(run_dir / "actor.pt", map_location=device))
  actor.eval()
  meta["motion_files"] = files
  return env, wm, actor, runner_cfg, meta
