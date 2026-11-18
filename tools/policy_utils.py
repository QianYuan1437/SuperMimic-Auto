"""离线推理：把 env 观测组装成 actor 输入（与上游 viewer/benchmark 一致）。"""

from __future__ import annotations

import torch
from supermimic.representations import extract_actor_goal
from supermimic.tasks.task_spec import build_actor_obs_td


class MotionPolicy:
  """只启用 motion 编码器的推理包装，自动维护 proprio 历史窗口。"""

  def __init__(self, env, wm, actor, runner_cfg) -> None:
    self.env = env
    self.rep = wm.representation
    self.sim_context = wm.sim_context
    self.actor = actor
    self.proprio = runner_cfg.actor_obs == "proprio"
    self.history_length = runner_cfg.history_length
    self.history: torch.Tensor | None = None
    actor.set_slot_mix({"motion": 1.0})

  @torch.inference_mode()
  def __call__(self, obs, dones: torch.Tensor | None = None) -> torch.Tensor:
    rep_state = self.rep.represent(obs["full_state"])
    sim_ctx = self.sim_context.read(self.env.unwrapped)
    sensor = self.rep.compute_actor_sensor(
      rep_state, sim_ctx, sim_actor_obs=obs["actor"] if self.proprio else None
    )
    frame = sensor.unsqueeze(1)
    if self.history_length == 1:
      history = frame
    elif self.history is None:
      history = frame.repeat(1, self.history_length, 1)
    else:
      history = torch.cat([self.history[:, 1:], frame], dim=1)
      if dones is not None:
        reset = dones.bool().view(-1, 1, 1)
        history = torch.where(reset, frame.repeat(1, self.history_length, 1), history)
    self.history = history
    td = build_actor_obs_td({"motion": extract_actor_goal(obs["goal"])}, history)
    return self.actor(td)
