"""离线评估：把每段动作切成若干个 episode 窗口，逐窗口检查是否跟踪成功。

每个并行环境从某段动作的某个窗口起点开始（无随机初始扰动、无推搡），运行到
episode 超时或动作结束即视为成功，被终止条件（摔倒、偏离过大）提前结束即为失败。
输出成功率、各段动作的失败窗口数和 mjlab 的跟踪误差指标（MPKPE 等）。

  uv run python tools/eval_local.py logs/SuperMimic-GenCoord-G1 --output-file eval.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import tyro
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends

from local_run import build_from_run, expand_motion_files, load_meta, resolve_run_dir
from policy_utils import MotionPolicy


@dataclass
class Args:
  run_dir: tyro.conf.Positional[str]
  motion_file: str | None = None
  """覆盖训练时的动作文件（例如评估未见过的动作）。"""
  stride_s: float | None = None
  """窗口起点间隔（秒），默认等于 episode 长度（10 s），即不重叠地覆盖全部动作。"""
  noise: bool = True
  """是否保留观测噪声（与上游 play/evaluate 一致）。"""
  device: str | None = None
  output_file: str | None = None


def main(args: Args) -> None:
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  run_dir = resolve_run_dir(args.run_dir)

  # 先读取动作长度以确定窗口数（= 并行环境数）。
  meta = load_meta(run_dir)
  files = expand_motion_files(args.motion_file or meta["motion_file"])
  lengths = [int(np.load(f)["joint_pos"].shape[0]) for f in files]
  fps = float(np.load(files[0])["fps"][0])

  probe_ep_s = unitree_g1_flat_tracking_env_cfg().episode_length_s

  def hook(env_cfg):
    # play 配置的回合长度是“无限”，这里改回训练时的长度，使每个窗口独立评估。
    env_cfg.episode_length_s = probe_ep_s
    env_cfg.commands["motion"].sampling_mode = "uniform"
    env_cfg.commands["motion"].pose_range = {}
    env_cfg.commands["motion"].velocity_range = {}
    env_cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    env_cfg.observations["actor"].enable_corruption = args.noise
    env_cfg.events.pop("push_robot", None)

  # 窗口：(clip_idx, global_start_frame)
  stride = int(round((args.stride_s or probe_ep_s) * fps))
  windows: list[tuple[int, int]] = []
  offset = 0
  for ci, n in enumerate(lengths):
    for s in range(0, max(n - 1, 1), stride):
      if n - s < fps:  # 不足 1 秒的尾巴不单独评估
        continue
      windows.append((ci, offset + s))
    offset += n
  num_envs = len(windows)
  print(f"[INFO] {len(files)} 段动作 → {num_envs} 个评估窗口（stride {stride} 帧）")

  env, wm, actor, runner_cfg, meta = build_from_run(
    run_dir,
    num_envs=num_envs,
    device=device,
    motion_file=args.motion_file,
    env_cfg_hook=hook,
  )
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  starts = torch.tensor([w[1] for w in windows], device=command.device)
  # 用固定起点替换均匀采样：每个环境 reset 时都回到自己的窗口起点。
  command._uniform_sampling = lambda env_ids: starts[env_ids]
  obs, _ = env.reset()
  policy = MotionPolicy(env, wm, actor, runner_cfg)

  ee_body_names = (
    unitree_g1_flat_tracking_env_cfg().terminations["ee_body_pos"].params["body_names"]
  )
  sums = {
    k: torch.zeros(num_envs, device=device)
    for k in ["mpkpe", "r_mpkpe", "joint_vel_error", "ee_pos_error", "ee_ori_error"]
  }
  active_steps = torch.zeros(num_envs, device=device)
  done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(num_envs, dtype=torch.bool, device=device)
  dones = None
  step = 0
  while not done_envs.all():
    ref = SimpleNamespace(
      num_envs=command.num_envs,
      device=command.device,
      cfg=command.cfg,
      body_pos_w=command.body_pos_w.clone(),
      body_pos_relative_w=command.body_pos_relative_w.clone(),
      body_quat_relative_w=command.body_quat_relative_w.clone(),
      joint_vel=command.joint_vel.clone(),
    )
    actions = policy(obs, dones)
    obs, _, dones, _ = env.step(actions)
    ref.robot_body_pos_w = command.robot_body_pos_w
    ref.robot_body_quat_w = command.robot_body_quat_w
    ref.robot_joint_vel = command.robot_joint_vel
    r = cast(MotionCommand, ref)
    active = (~done_envs & ~dones.bool()).float()
    active_steps += active
    sums["mpkpe"] += active * compute_mpkpe(r)
    sums["r_mpkpe"] += active * compute_root_relative_mpkpe(r)
    sums["joint_vel_error"] += active * compute_joint_velocity_error(r)
    sums["ee_pos_error"] += active * compute_ee_position_error(r, ee_body_names)
    sums["ee_ori_error"] += active * compute_ee_orientation_error(r, ee_body_names)

    manager = env.unwrapped.termination_manager
    newly_done = dones.bool() & ~done_envs
    success |= newly_done & manager.time_outs.bool() & ~manager.terminated.bool()
    done_envs |= newly_done
    step += 1
    if step % 100 == 0:
      print(f"[INFO] step {step}: {int(done_envs.sum())}/{num_envs} 完成")

  denom = active_steps.clamp(min=1)
  metrics = {k: (v / denom).mean().item() for k, v in sums.items()}
  clip_idx = torch.tensor([w[0] for w in windows], device=device)
  per_clip = {}
  for ci, f in enumerate(files):
    mask = clip_idx == ci
    per_clip[Path(f).stem] = {
      "windows": int(mask.sum()),
      "failed": int((mask & ~success).sum()),
      "mpkpe": (sums["mpkpe"] / denom)[mask].mean().item(),
    }
  result = {
    "run_dir": str(run_dir),
    "num_clips": len(files),
    "num_windows": num_envs,
    "failed_windows": int((~success).sum()),
    "success_rate": success.float().mean().item(),
    "failed_clips": sum(1 for v in per_clip.values() if v["failed"] > 0),
    **metrics,
    "per_clip": per_clip,
  }
  print("\n" + "=" * 60)
  for k, v in result.items():
    if k != "per_clip":
      print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
  for name, v in per_clip.items():
    flag = "❌" if v["failed"] else "✅"
    print(f"  {flag} {name:<28} 失败 {v['failed']}/{v['windows']}  mpkpe {v['mpkpe']:.4f}")
  print("=" * 60)
  out = Path(args.output_file) if args.output_file else run_dir / "eval.json"
  out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
  print(f"[INFO] 结果已保存: {out}")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
