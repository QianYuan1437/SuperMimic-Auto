# 7. Docker 部署

如果不想在宿主机上安装 uv 和 Python 环境，可以用 Docker。宿主机需要：

- NVIDIA 驱动 ≥ 580；
- Docker 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（`docker run --gpus all` 可用）。

```bash
# 构建镜像（会在镜像内克隆固定版本的上游代码并安装依赖）
docker build -t supermimic-auto -f docker/Dockerfile .

# 数据准备（在容器内执行，结果写回宿主机的 data/）
bash docker/run.sh bash scripts/02_download_data.sh
bash docker/run.sh bash scripts/03_convert_motions.sh

# 训练（日志与权重写到宿主机的 outputs/logs/）
bash docker/run.sh env PRESET=lafan bash scripts/04_train.sh

# 评估 / 查看器（8080 端口已映射）
bash docker/run.sh bash scripts/06_eval.sh
bash docker/run.sh bash scripts/05_play.sh
```

`docker/run.sh` 挂载了 `data/`、`outputs/logs/`（映射为容器内的 `third_party/supermimic/logs`）以及 `scripts/`、`tools/`，修改脚本不需要重新构建镜像。

::: tip 国内构建
构建时 `uv sync --frozen` 会从 PyPI 下载约 3 GB 轮子，可以先在宿主机配置 Docker 代理，或参考 [安装环境 · 国内网络](./install.md#国内网络) 修改 Dockerfile 中的 `uv sync` 一行。
:::
