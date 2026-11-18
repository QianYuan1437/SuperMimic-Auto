#!/usr/bin/env bash
# 所有脚本共享的路径与工具函数。被 source，不要直接执行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../configs/upstream.env
source "$ROOT/configs/upstream.env"

TP="$ROOT/third_party"            # 上游代码（supermimic 与 mjlab_research 必须是兄弟目录）
SM_DIR="$TP/supermimic"
MJ_DIR="$TP/mjlab_research"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
CSV_DIR="$DATA_DIR/lafan1_g1_csv"
NPZ_DIR="$DATA_DIR/lafan1_g1_npz"

# 无显示器的服务器上用 EGL 离屏渲染
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# 输出重定向到文件时也实时刷新日志
export PYTHONUNBUFFERED=1

log()  { printf '\033[1;32m[supermimic]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[supermimic]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[supermimic]\033[0m %s\n' "$*" >&2; exit 1; }

need_workspace() {
  [[ -d "$SM_DIR/.venv" ]] || die "未找到 $SM_DIR/.venv，请先运行 scripts/01_setup.sh"
}

# 在 supermimic 的 uv 环境里运行命令（工作目录切到 supermimic，logs/ 会写在那里）
sm_run() {
  need_workspace
  (cd "$SM_DIR" && uv run --frozen "$@")
}
