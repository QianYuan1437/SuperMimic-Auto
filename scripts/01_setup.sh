#!/usr/bin/env bash
# 克隆上游代码到 third_party/（固定到 configs/upstream.env 中的提交），并创建 uv 环境。
# 可重复执行：已存在的仓库只会 fetch + checkout，不会重新克隆。
# 国内网络：UV_MIRROR=https://repo.huaweicloud.com/repository/pypi/simple bash scripts/01_setup.sh
source "$(dirname "$0")/common.sh"

command -v uv >/dev/null || die "请先安装 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
mkdir -p "$TP"

clone_at() {  # <url> <dir> <commit>
  local url=$1 dir=$2 commit=$3
  if [[ ! -d "$dir/.git" ]]; then
    log "克隆 $url"
    git clone "$url" "$dir"
  fi
  if [[ "${UPSTREAM_LATEST:-0}" == 1 ]]; then
    log "UPSTREAM_LATEST=1：$dir 使用上游最新 main"
    git -C "$dir" fetch origin main && git -C "$dir" checkout -q origin/main
  else
    git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch origin
    git -C "$dir" checkout -q "$commit"
  fi
  log "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

clone_at "$SUPERMIMIC_REPO" "$SM_DIR" "$SUPERMIMIC_COMMIT"
clone_at "$MJLAB_REPO" "$MJ_DIR" "$MJLAB_COMMIT"

log "uv sync（首次会下载 Python 3.13、torch 2.15 + CUDA 13 等约 8~10 GB，请耐心等待）"
if [[ "${UPSTREAM_LATEST:-0}" == 1 ]]; then
  (cd "$SM_DIR" && uv sync)
elif [[ -n "${UV_MIRROR:-}" ]]; then
  # --frozen 会直连锁文件里的 PyPI 地址，镜像无效；这里先在镜像上重新解析安装
  # （版本与锁文件一致），再还原官方 uv.lock，最后用 --frozen 校验环境与锁文件一致。
  log "使用 PyPI 镜像 $UV_MIRROR"
  (cd "$SM_DIR" && UV_DEFAULT_INDEX="$UV_MIRROR" uv sync && git checkout -q uv.lock && uv sync --frozen)
else
  (cd "$SM_DIR" && uv sync --frozen)
fi

log "验证 torch / CUDA / mujoco-warp"
sm_run python - <<'PY'
import torch, warp, mujoco, mujoco_warp, mjlab
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch 看不到 GPU，检查驱动版本"
print("GPU:", torch.cuda.get_device_name(0))
print("warp", warp.__version__, "| mujoco", mujoco.__version__)
PY
log "安装完成 ✅  下一步：scripts/02_download_data.sh"
