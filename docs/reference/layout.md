# 仓库结构

```text
SuperMimic-Auto/
├── README.md                 仓库首页：一页速览 + 链接到详细文档
├── configs/
│   └── upstream.env          上游仓库地址与固定提交号、数据集地址（唯一的版本来源）
├── scripts/                  按步骤编号的 Shell 入口（00 检查 → 08 画图）
│   └── common.sh             公共路径与函数（被其他脚本 source）
├── tools/                    Python 工具：本地化训练/回放/评估/录像，包装上游代码
├── docker/                   Dockerfile 与 run.sh
├── results/                  本机复现产物：评估 JSON、训练曲线、日志摘要
├── docs/                     文档（同时是 GitHub Pages 站点的源文件）
│   ├── index.md              站点首页
│   ├── guide/                部署指南（1~8 步）
│   ├── paper/                论文精读、社区复现
│   ├── results/              本机复现记录
│   ├── reference/            脚本、超参数、仓库结构
│   ├── public/               站点静态资源（视频、图片）
│   └── .vitepress/           VitePress 配置与主题
├── .github/workflows/
│   └── pages.yml             推送 main 后自动构建并发布 GitHub Pages
├── package.json              文档站点依赖（VitePress）
│
├── third_party/   (git 忽略) 01_setup.sh 克隆的上游代码与 .venv
│   ├── supermimic/           └── logs/<任务>/<时间戳>/  训练输出
│   └── mjlab_research/
└── data/          (git 忽略) 02/03 下载与转换的动作数据
```

## 设计原则

- **不修改上游代码**：`third_party/` 完全是官方仓库在固定提交上的克隆，所有本地化改动都在 `tools/` 中以包装或运行时替换的方式实现。升级上游只需修改 `configs/upstream.env`。
- **一份文档两处阅读**：`docs/` 下的 Markdown 在 GitHub 上可以直接阅读，同时由 VitePress 构建成站点，每页都提供“下载 Markdown”按钮。
- **大文件不入库**：数据、权重、Python 环境都由脚本重新生成；`results/` 只保存体积小的评估结果与曲线。

## 本地预览文档站点

```bash
npm install
npm run docs:dev        # http://localhost:5173
npm run docs:build      # 输出到 docs/.vitepress/dist
```
