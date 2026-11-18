# 1. 硬件与系统要求

## 最低 / 推荐配置

| 项目 | 最低 | 推荐 | 说明 |
| --- | --- | --- | --- |
| GPU | NVIDIA，16 GB 显存（需减小并行环境数） | 24 GB 及以上（RTX 3090/4090、A5000、L40、A100、H100） | MuJoCo Warp 与 PyTorch 都只支持 NVIDIA CUDA |
| 驱动 | ≥ 580（CUDA 13.0，依赖 CUDA 小版本兼容，未实测） | ≥ 595（CUDA 13.2，本机实测） | 锁定的 `torch 2.15.0+cu132` 轮子自带 CUDA 13.2 运行库，**不需要单独安装 CUDA Toolkit** |
| 系统 | Linux x86_64（Ubuntu 22.04 / 24.04 已验证） | Ubuntu 24.04 | Windows 请用 WSL2；macOS 只能做回放 |
| 内存 | 16 GB | 32 GB 以上 | |
| 磁盘 | 20 GB 可用 | 50 GB | Python 环境约 10 GB，数据与日志另计 |
| 网络 | 能访问 GitHub、PyPI、Hugging Face | — | 国内可用镜像，见 [安装环境](./install.md#国内网络) |

论文的计时都在 **单张 H100 80 GB** 上完成（例如 3 小时达到 SONIC 的 0.90 归一化回报）。消费级显卡也能跑通完整流程，只是速度更慢；本机 RTX 3090 的实测速度见 [复现记录](../results/index.md)。

## 软件栈

| 组件 | 版本（由 `uv.lock` 固定） | 作用 |
| --- | --- | --- |
| Python | 3.13 | 由 uv 自动下载，不影响系统 Python |
| PyTorch | 2.15.0 + CUDA 13.2 | 世界模型、策略网络与 BPTT |
| MuJoCo / MuJoCo Warp | 4.0.1 | GPU 并行刚体仿真 |
| NVIDIA Warp | 1.19 | MuJoCo Warp 的底层 |
| mjlab（HybridRobotics fork） | 1.6.0 @ `6ea3e57` | Isaac Lab 风格的任务管理器 + G1 跟踪任务 |
| supermimic | 0.1.0 @ `7e94b07` | 世界模型、BPTT 策略学习、训练循环 |

上游提交号统一记录在 [`configs/upstream.env`](https://github.com/QianYuan1437/SuperMimic-Auto/blob/main/configs/upstream.env)，想试用上游最新代码可以 `UPSTREAM_LATEST=1 bash scripts/01_setup.sh`（不保证能复现本仓库的结果）。

## 自检

```bash
bash scripts/00_check_env.sh
```

脚本只读取信息，会检查：GPU 与驱动版本、显存、uv、git、磁盘空间、EGL 离屏渲染库。如果驱动低于 580，必须先升级；580~594 可以先尝试，遇到 `CUDA driver version is insufficient` 或 PTX 相关报错时再升级：

```bash
# Ubuntu 示例
sudo ubuntu-drivers list           # 查看可用驱动
sudo apt install nvidia-driver-595 # 或更新的版本
sudo reboot
```
