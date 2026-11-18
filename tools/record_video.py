"""无头录制策略跟踪视频（EGL 离屏渲染），用于展示页面或远程服务器上检查效果。

  MUJOCO_GL=egl uv run python tools/record_video.py logs/SuperMimic-GenCoord-G1 \
      --clip run1_subject5 --seconds 20 --output outputs/run1.mp4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mediapy as media
import numpy as np
import torch
import tyro
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends

from local_run import build_from_run, expand_motion_files, load_meta, resolve_run_dir
from policy_utils import MotionPolicy


@dataclass
class Args:
  run_dir: tyro.conf.Positional[str]
  clip: str | None = None
  """动作名（文件名去掉 .npz），默认第一段。"""
  start_s: float = 0.0
  seconds: float = 15.0
  width: int = 960
  height: int = 540
  distance: float = 2.0
  """相机到躯干的距离（m）；mjlab 默认 2.8。"""
  elevation: float = -10.0
  output: str | None = None
  device: str | None = None


def main(args: Args) -> None:
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  run_dir = resolve_run_dir(args.run_dir)
  files = expand_motion_files(load_meta(run_dir)["motion_file"])
  if args.clip is not None:
    files = [f for f in files if Path(f).stem == args.clip]
    if not files:
      raise SystemExit(f"训练动作中没有 {args.clip}")
  clip_file = files[0]
  fps = float(np.load(clip_file)["fps"][0])
  start = int(args.start_s * fps)

  def hook(env_cfg):
    cmd = env_cfg.commands["motion"]
    cmd.sampling_mode = "uniform"
    cmd.pose_range, cmd.velocity_range, cmd.joint_position_range = {}, {}, (0.0, 0.0)
    env_cfg.events.pop("push_robot", None)
    env_cfg.episode_length_s = args.seconds + 1.0
    env_cfg.viewer.width, env_cfg.viewer.height = args.width, args.height
    env_cfg.viewer.distance, env_cfg.viewer.elevation = args.distance, args.elevation

  env, wm, actor, runner_cfg, _ = build_from_run(
    run_dir,
    num_envs=1,
    device=device,
    motion_file=clip_file,
    env_cfg_hook=hook,
    render_mode="rgb_array",
  )
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  command._uniform_sampling = lambda env_ids: torch.full_like(env_ids, start)
  obs, _ = env.reset()
  policy = MotionPolicy(env, wm, actor, runner_cfg)

  frames, falls, dones = [], 0, None
  for _ in range(int(args.seconds / env.unwrapped.step_dt)):
    obs, _, dones, _ = env.step(policy(obs, dones))
    if env.unwrapped.termination_manager.terminated.any():
      falls += 1
    frames.append(env.unwrapped.render())
  out = Path(args.output or run_dir / f"{Path(clip_file).stem}.mp4")
  out.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(out), frames, fps=round(1.0 / env.unwrapped.step_dt))
  print(f"[INFO] 视频: {out}（{len(frames)} 帧，提前终止 {falls} 次）")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
