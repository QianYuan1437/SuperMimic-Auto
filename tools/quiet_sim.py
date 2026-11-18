"""关闭 MuJoCo Warp 的“求解器/线搜索迭代次数达到上限”提示。

训练初期机器人频繁摔倒，接触求解器经常跑满迭代次数，每一步都会打印两行提示，
一小时能产生上百 MB 日志，但这对训练没有影响。这里在 ``put_model`` 之后、CUDA graph
捕获之前清掉 ``opt.warn_overflow`` 中这两个标志位；真正的缓冲区溢出警告（nefc、
contact 等）保持开启。设置 ``SM_SOLVER_WARNINGS=1`` 可恢复原样。
"""

from __future__ import annotations

import os

import mujoco_warp as mjwarp
from mjlab.sim.sim import SimulationCfg

_QUIET = mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
_orig_apply_wp_opt = SimulationCfg.apply_wp_opt


def _apply_wp_opt(self: SimulationCfg, wp_opt) -> None:
  _orig_apply_wp_opt(self, wp_opt)
  wp_opt.warn_overflow = int(wp_opt.warn_overflow) & ~int(_QUIET)


if os.environ.get("SM_SOLVER_WARNINGS", "0") != "1":
  SimulationCfg.apply_wp_opt = _apply_wp_opt
