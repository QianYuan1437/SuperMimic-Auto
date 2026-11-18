# 8. 常见问题排查

## 安装

### `torch.cuda.is_available()` 为 False / `CUDA driver version is insufficient`

锁定的 `torch 2.15.0+cu132` 使用 CUDA 13.2，需要驱动 ≥ 580，推荐 ≥ 595。用 `nvidia-smi` 查看右上角的 “CUDA Version”，低于 13.0 必须升级驱动，低于 13.2 遇到报错时也请升级。

### `uv sync` 下载非常慢

`--frozen` 模式按锁文件里的地址直连 PyPI。见 [安装环境 · 国内网络](./install.md#国内网络)，用 `UV_DEFAULT_INDEX=<镜像> uv sync` 改从镜像下载（本机实测从约 1 MB/s 提高到约 3 MB/s，解析出的 143 个包版本与锁文件完全一致）。

### `No interpreter found for Python 3.13`

uv 会自动下载 Python。如果 GitHub 访问受限，设置 `UV_PYTHON_INSTALL_MIRROR`（见安装页）。

### 首次运行时卡在 `Module ... load on device 'cuda:0' took ...`

这是 NVIDIA Warp 在编译 MuJoCo Warp 的 CUDA kernel，首次需要 1~3 分钟，结果缓存在 `~/.cache/warp`，之后启动会很快。

## 数据

### `motion joints [...] are not the robot's [...]`

NPZ 不是用 mjlab 转换器生成的，或者 CSV 的关节顺序不对。用 `scripts/03_convert_motions.sh` 重新转换。

### `No files matched pattern`

传 glob 时要加引号：`MOTION_FILE="data/my_npz/*.npz"`。

### Hugging Face 下载失败

`export HF_ENDPOINT=https://hf-mirror.com` 后重试，脚本会跳过已经下载的文件。

## 训练

### 显存不足（`CUDA out of memory`）

按影响从大到小依次尝试：

```bash
NUM_ENVS=2048 bash scripts/04_train.sh                       # 减少并行环境（同时缩小 replay buffer）
bash scripts/04_train.sh --runner.policy.batch-size 2048     # 减小想象 rollout 的 batch
bash scripts/04_train.sh --runner.buffer-size 500            # 缩短每个环境保留的转移数
bash scripts/04_train.sh --runner.policy.horizon 16          # 缩短 BPTT 视界（保留 critic 时仍有效）
```

### W&B 要求登录 / 网络超时

`train_local.py` 默认设置 `WANDB_MODE=offline`，不需要登录。如果手动设置成了 online，执行 `uv run wandb login`，或者改回 offline。离线记录可以之后用 `uv run wandb sync wandb/offline-run-*` 上传。

### 奖励长期不上升

先用 `PRESET=single` 在单段动作上确认流程正确（几千步内 `Mean reward` 应明显上升）。再检查动作数据：用 `scripts/03_convert_motions.sh --render --overwrite` 生成预览视频，确认参考动作本身合理（没有穿地、翻转）。

### 桌面卡顿

训练会占满 GPU。在同一张卡上接显示器时界面会变慢，属于正常现象；可以通过 `CUDA_VISIBLE_DEVICES` 选择不接显示器的卡。

## 回放与评估

### viser 页面打不开

远程服务器需要端口转发：`ssh -L 8080:localhost:8080 user@server`。

### `run_meta.json` 不存在

离线回放/评估只支持用 `scripts/04_train.sh`（即 `tools/train_local.py`）训练的运行。直接用上游 `train.py` 训练的运行请使用上游 `play.py --wandb-path <entity/project/run>`（需要 W&B 在线）。

### 录像时报 EGL 错误

安装 `sudo apt install libegl1 libegl-dev`，并确认 `MUJOCO_GL=egl`（脚本已默认设置）。
