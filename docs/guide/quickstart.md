# 快速开始

在一台装好 NVIDIA 驱动（≥ 580）的 Linux 电脑上，按顺序执行下面的命令，即可完成从安装到训练、评估的完整流程。每一步的详细说明见左侧目录。

```bash
# 0. 安装 uv（已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 1. 获取本仓库
git clone https://github.com/QianYuan1437/SuperMimic-Auto.git
cd SuperMimic-Auto

# 2. 检查硬件与软件环境（只读）
bash scripts/00_check_env.sh

# 3. 克隆固定版本的官方代码并安装依赖（首次约 8~10 GB 下载）
bash scripts/01_setup.sh

# 4. 下载 LAFAN1 → Unitree G1 的 40 段重定向动作（约 85 MB）
#    国内网络可先：export HF_ENDPOINT=https://hf-mirror.com
bash scripts/02_download_data.sh

# 5. CSV(30 fps) → NPZ(50 fps)
bash scripts/03_convert_motions.sh

# 6. 训练：先用单段动作快速验证（几分钟即可看到奖励上升，Ctrl+C 停止）
PRESET=single ITERS=20000 bash scripts/04_train.sh

# 7. 正式训练：全部 40 段 LAFAN1 动作
PRESET=lafan bash scripts/04_train.sh

# 8. 无头评估（成功率、失败窗口、MPKPE），结果写到运行目录的 eval.json
bash scripts/06_eval.sh

# 9. 录制视频 / 打开交互查看器（浏览器访问 http://localhost:8080）
bash scripts/07_record.sh --clip dance1_subject1 --seconds 20
bash scripts/05_play.sh
```

::: tip 不需要 W&B 账号
官方代码默认从作者私有的 W&B registry 下载动作，并把权重上传到 W&B。本仓库的 `tools/` 把这两步换成了本地文件：动作从 `data/` 读取，权重、配置和指标都写在 `third_party/supermimic/logs/<任务>/<时间戳>/`。如果想同步到 W&B，设置 `WANDB_MODE=online` 并先执行 `uv run wandb login`。
:::

## 流程图

```text
LAFAN1 CSV ──03_convert──▶ NPZ ──04_train──▶ logs/<task>/<time>/
 (Hugging Face)                              ├── world_model.pt   可微世界模型
                                             ├── actor.pt         跟踪策略
                                             ├── run_meta.json    离线回放所需配置
                                             ├── metrics.jsonl    训练曲线
                                             └── *.onnx           (DEPLOY=1 时) 部署模型
                       ┌──────────────┬──────────────┬──────────────┐
                    05_play        06_eval        07_record      08_plot
                   交互查看器     批量评估         视频          曲线图
```

## 接下来

- 显存不足、下载太慢等问题：[常见问题排查](./troubleshooting.md)
- 想上真机：[导出与真机部署](./deploy.md)
- 这套方法为什么快：[论文精读与代码对照](../paper/notes.md)
- 本机（RTX 3090）实际跑出来的数字：[复现记录](../results/index.md)
