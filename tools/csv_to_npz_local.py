"""把重定向 CSV 转成 mjlab/SuperMimic 使用的 motion NPZ，保存到本地（不上传 W&B）。

上游的 ``mjlab.scripts.csv_to_npz`` 固定会把结果上传到 W&B registry，这里复用它的
``MotionLoader`` 做插值与速度计算，只是把结果写到本地目录，便于离线训练。

用法（在 supermimic 的 uv 环境中）::

  uv run python tools/csv_to_npz_local.py --input-dir data/lafan1_g1_csv \
      --output-dir data/lafan1_g1_npz [--render]
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from mjlab.scripts.csv_to_npz import MotionLoader


@dataclass
class Args:
  input_dir: str
  """包含 *.csv 的目录（每行: base_pos(3), base_quat_xyzw(4), joint_pos(29)）。"""
  output_dir: str
  pattern: str = "*.csv"
  input_fps: float = 30.0
  output_fps: float = 50.0
  device: str = "cuda:0"
  overwrite: bool = False
  render: bool = False
  """额外渲染每段动作的预览视频（<output_dir>/preview/<name>.mp4）。"""


def render_preview(motion: MotionLoader, path: str, fps: float, device: str) -> None:
  import mediapy as media
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
  )
  from mjlab.viewer.offscreen_renderer import OffscreenRenderer
  from mjlab.viewer.viewer_config import ViewerConfig

  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  renderer = OffscreenRenderer(
    model=sim.mj_model,
    cfg=ViewerConfig(
      height=360,
      width=480,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.5,
      elevation=-5.0,
      azimuth=20,
    ),
    scene=scene,
  )
  renderer.initialize()
  robot = scene["robot"]
  root_pose = torch.cat([motion.motion_base_poss, motion.motion_base_rots], dim=1)
  frames = []
  for i in range(motion.output_frames):
    robot.write_root_link_pose_to_sim(root_pose[i : i + 1])
    robot.write_joint_position_to_sim(motion.motion_dof_poss[i : i + 1])
    renderer.update(sim.data)
    frames.append(renderer.render())
  media.write_video(path, frames, fps=fps)


def main(args: Args) -> None:
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
  files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
  if not files:
    raise SystemExit(f"{args.input_dir} 中没有匹配 {args.pattern} 的文件")
  os.makedirs(args.output_dir, exist_ok=True)

  total_frames = 0
  for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(args.output_dir, f"{name}.npz")
    if os.path.exists(out) and not args.overwrite:
      total_frames += np.load(out)["joint_pos"].shape[0]
      continue
    motion = MotionLoader(
      motion_file=path,
      input_fps=args.input_fps,
      output_fps=args.output_fps,
      device=args.device,
    )
    np.savez(out, **motion.arrays())
    total_frames += motion.output_frames
    print(f"[OK] {name}: {motion.output_frames} 帧 -> {out}")
    if args.render:
      os.makedirs(os.path.join(args.output_dir, "preview"), exist_ok=True)
      render_preview(
        motion,
        os.path.join(args.output_dir, "preview", f"{name}.mp4"),
        args.output_fps,
        args.device,
      )

  hours = total_frames / args.output_fps / 3600
  print(f"[DONE] {len(files)} 段动作，共 {total_frames} 帧（{hours:.2f} 小时）")


if __name__ == "__main__":
  main(tyro.cli(Args))
