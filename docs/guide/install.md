# 2. 安装环境

## 一键安装

```bash
bash scripts/01_setup.sh
```

脚本做了三件事：

1. 把官方代码克隆到 `third_party/`，并切换到 [`configs/upstream.env`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/configs/upstream.env) 里固定的提交：
   - `third_party/supermimic` ← [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic)
   - `third_party/mjlab_research` ← [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research)

   两者**必须是兄弟目录**：supermimic 的 `pyproject.toml` 通过 `../mjlab_research` 以可编辑方式安装 mjlab。
2. 在 `third_party/supermimic/.venv` 中执行 `uv sync --frozen`，严格按照 `uv.lock` 安装 Python 3.13、torch 2.15（CUDA 13.2）、mujoco-warp 4.0.1 等依赖。
3. 运行一段 Python 验证 torch 能看到 GPU，并打印各库版本。

成功时最后几行类似：

```text
torch 2.15.0+cu132 cuda 13.2 available: True
GPU: NVIDIA GeForce RTX 3090
warp 1.19.0 | mujoco 4.0.1
[supermimic] 安装完成 ✅
```

脚本可以重复执行：已经克隆的仓库只会切换提交，已经下载的轮子会命中 uv 缓存。

## 手动安装（等价步骤）

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/HybridRobotics/supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git
git -C supermimic checkout 7e94b073940b7a19b9a82ff01e191cfb35c4ba91
git -C mjlab_research checkout 6ea3e574b08605b79a290a6d464a812a26acd6fd
cd supermimic && uv sync --frozen
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 国内网络

| 下载内容 | 默认来源 | 加速方法 |
| --- | --- | --- |
| uv 本体 | astral.sh | `pip install uv`，或从 GitHub Release 下载二进制 |
| Python 3.13 | GitHub（python-build-standalone） | `export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone` |
| Python 包 | `uv.lock` 中记录的 files.pythonhosted.org 地址 | 见下文 |
| 动作数据 | huggingface.co | `export HF_ENDPOINT=https://hf-mirror.com` |

`uv sync --frozen` 会直接使用锁文件里的下载地址，设置 PyPI 镜像对它**无效**。如果直连 PyPI 太慢，用 `UV_MIRROR` 让安装脚本改走镜像：

```bash
UV_MIRROR=https://repo.huaweicloud.com/repository/pypi/simple bash scripts/01_setup.sh
# 也可用 https://mirrors.aliyun.com/pypi/simple 、https://mirrors.ustc.edu.cn/pypi/simple
```

脚本会先在镜像上重新解析并安装（本机实测解析出的 143 个包版本与官方锁文件**完全相同**，只是下载地址不同），然后还原官方 `uv.lock`，再用 `uv sync --frozen` 校验环境与锁文件一致。本机实测下载速度从直连 PyPI 的约 1 MB/s 提高到约 3 MB/s。

`warp-lang` 在锁文件里来自 `pypi.nvidia.com`，如果它下载失败，在 PyPI 镜像上也能找到同版本轮子。

## 验证安装

```bash
cd third_party/supermimic
uv run python -c "import mjlab, supermimic.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'SuperMimic' in t])"
# ['SuperMimic-GenCoord-G1', 'SuperMimic-MaxCoord-G1']
```

可选：运行官方单元测试（约 1~2 分钟）：

```bash
cd third_party/supermimic && uv run pytest -q
```
