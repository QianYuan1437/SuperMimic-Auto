# SuperMimic 运行镜像：Ubuntu 24.04 + uv + Python 3.13 + 固定版本的上游代码。
# torch 轮子自带 CUDA 13 运行库，宿主机只需 NVIDIA 驱动 >= 580 与 nvidia-container-toolkit。
#
#   docker build -t supermimic-auto -f docker/Dockerfile .
#   bash docker/run.sh bash scripts/04_train.sh
FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libegl1 libegl-dev libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /workspace
COPY configs/ configs/
COPY scripts/ scripts/
COPY tools/ tools/

# 克隆上游代码并安装依赖（作为镜像层缓存；数据与日志通过卷挂载）
RUN --mount=type=cache,target=/root/.cache/uv \
    bash -c 'source scripts/common.sh && \
      git clone "$SUPERMIMIC_REPO" "$SM_DIR" && git -C "$SM_DIR" checkout -q "$SUPERMIMIC_COMMIT" && \
      git clone "$MJLAB_REPO" "$MJ_DIR" && git -C "$MJ_DIR" checkout -q "$MJLAB_COMMIT" && \
      cd "$SM_DIR" && uv sync --frozen'

EXPOSE 8080
CMD ["bash"]
