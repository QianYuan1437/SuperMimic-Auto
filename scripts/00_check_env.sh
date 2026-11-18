#!/usr/bin/env bash
# 检查硬件与基础软件是否满足要求。只读，不修改系统。
source "$(dirname "$0")/common.sh"
set +e
ok=1

log "1/5 NVIDIA 驱动与 GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  # 锁定的 torch 2.15.0+cu132 需要 CUDA 13 驱动（>= 580）；本仓库在 595（CUDA 13.2）上实测
  if (( drv < 580 )); then
    warn "驱动主版本 $drv < 580：CUDA 13 版 torch 无法运行，请升级驱动（见 docs/guide/requirements.md）"
    ok=0
  elif (( drv < 595 )); then
    warn "驱动主版本 $drv < 595：依赖 CUDA 小版本兼容，未实测；如遇 PTX/驱动版本报错请升级到 >= 595"
  fi
else
  warn "找不到 nvidia-smi：没有安装 NVIDIA 驱动"; ok=0
fi

log "2/5 显存（训练默认 4096 个并行环境，建议 >= 24 GB）"
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
[[ -n "$mem" && "$mem" -lt 20000 ]] && warn "显存 ${mem} MiB 偏小，训练时请减小 --num-envs（见 docs/04-train.md）"

log "3/5 uv"
if command -v uv >/dev/null; then uv --version
else warn "未安装 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; ok=0; fi

log "4/5 git / 磁盘"
command -v git >/dev/null || { warn "未安装 git"; ok=0; }
df -h "$ROOT" | tail -1
avail=$(df -Pk "$ROOT" | tail -1 | awk '{print $4}')
(( avail < 20*1024*1024 )) && warn "可用磁盘 < 20 GB（Python 环境约 10 GB）"

log "5/5 EGL 离屏渲染库（录制视频、无头服务器需要）"
ldconfig -p 2>/dev/null | grep -q libEGL.so || warn "未找到 libEGL：sudo apt install libegl1"

if (( ok )); then log "环境检查通过 ✅"; else die "环境检查未通过，请按上面的提示处理"; fi
