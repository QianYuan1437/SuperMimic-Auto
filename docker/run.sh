#!/usr/bin/env bash
# 以 GPU 方式启动容器：data/ 与训练日志挂载到宿主机，8080 端口用于 viser 查看器。
#   bash docker/run.sh                          # 进入交互 shell
#   bash docker/run.sh bash scripts/04_train.sh # 直接训练
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-supermimic-auto}"
mkdir -p "$ROOT/data" "$ROOT/outputs/logs"
tty=(); [[ -t 1 ]] && tty=(-it)
exec docker run --rm "${tty[@]}" --gpus all --shm-size=8g \
  -e WANDB_MODE="${WANDB_MODE:-offline}" -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -v "$ROOT/data:/workspace/data" \
  -v "$ROOT/outputs/logs:/workspace/third_party/supermimic/logs" \
  -v "$ROOT/scripts:/workspace/scripts" -v "$ROOT/tools:/workspace/tools" \
  -p 8080:8080 "$IMAGE" "$@"
