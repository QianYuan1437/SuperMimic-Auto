import { fileURLToPath } from 'node:url'
import { type DefaultTheme, defineConfigWithTheme } from 'vitepress'
import { type ExportOptions, mdExportPlugin, writeMarkdownExports } from './mdExport'

const base = process.env.VITEPRESS_BASE || '/'

/** GitHub repository of the project: change it here once, every source link (<Src>) follows. */
export const REPO_URL = 'https://github.com/QianYuan1437/SuperMimic-Auto'
export const REPO_BRANCH = 'main'
export const SITE_URL = 'https://qianyuan1437.github.io/SuperMimic-Auto/'

const SRC_DIR = fileURLToPath(new URL('..', import.meta.url))
const MD_EXPORT: ExportOptions = {
  siteUrl: SITE_URL,
  repoUrl: REPO_URL,
  repoBranch: REPO_BRANCH,
  docsPathInRepo: 'docs',
}

export interface ThemeConfig extends DefaultTheme.Config {
  repoUrl: string
  repoBranch: string
  docsPathInRepo: string
}

export default defineConfigWithTheme<ThemeConfig>({
  base,
  lang: 'zh-CN',
  title: 'SuperMimic Auto',
  description:
    'SuperMimic（CoRL 2026 Oral）一键部署与复刻指南：用一个可微世界模型训练 Unitree G1 人形机器人全身动作跟踪，任意一台 NVIDIA GPU 电脑即可复现。',
  cleanUrls: true,
  lastUpdated: true,
  srcExclude: ['README.md'],
  ignoreDeadLinks: [/^https?:\/\/localhost/],
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }]],
  markdown: { math: true },
  themeConfig: {
    repoUrl: REPO_URL,
    repoBranch: REPO_BRANCH,
    docsPathInRepo: MD_EXPORT.docsPathInRepo,
    socialLinks: [{ icon: 'github', link: REPO_URL }],
    logo: '/favicon.svg',
    nav: [
      { text: '快速开始', link: '/guide/quickstart', activeMatch: '/guide/' },
      { text: '论文解读', link: '/paper/notes', activeMatch: '/paper/' },
      { text: '本机结果', link: '/results/', activeMatch: '/results/' },
      { text: '参考', link: '/reference/scripts', activeMatch: '/reference/' },
    ],
    sidebar: [
      {
        text: '部署指南',
        items: [
          { text: '快速开始（10 条命令）', link: '/guide/quickstart' },
          { text: '1. 硬件与系统要求', link: '/guide/requirements' },
          { text: '2. 安装环境', link: '/guide/install' },
          { text: '3. 准备动作数据', link: '/guide/data' },
          { text: '4. 训练', link: '/guide/train' },
          { text: '5. 回放、评估与录像', link: '/guide/evaluate' },
          { text: '6. 导出与真机部署', link: '/guide/deploy' },
          { text: '7. Docker 部署', link: '/guide/docker' },
          { text: '8. 常见问题排查', link: '/guide/troubleshooting' },
        ],
      },
      {
        text: '论文与生态',
        items: [
          { text: '论文精读与代码对照', link: '/paper/notes' },
          { text: '社区复现与扩展', link: '/paper/community' },
        ],
      },
      {
        text: '本机复现结果',
        items: [{ text: 'RTX 3090 复现记录', link: '/results/' }],
      },
      {
        text: '参考',
        items: [
          { text: '脚本与工具一览', link: '/reference/scripts' },
          { text: '训练超参数', link: '/reference/config' },
          { text: '仓库结构', link: '/reference/layout' },
        ],
      },
    ],
    editLink: {
      pattern: `${REPO_URL}/edit/${REPO_BRANCH}/docs/:path`,
      text: '在 GitHub 上编辑此页',
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一页', next: '下一页' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到相关结果',
            resetButtonTitle: '清除',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
    footer: {
      message:
        'SuperMimic 原始代码 © The SuperMimic Authors（Apache-2.0）· 本部署指南 MIT · 每页均可下载 Markdown 原文',
    },
  },
  async buildEnd(siteConfig) {
    const n = await writeMarkdownExports(siteConfig.srcDir, siteConfig.outDir, MD_EXPORT)
    console.log(`  Markdown exports: ${n} pages -> ${siteConfig.outDir}/md/`)
  },
  vite: {
    plugins: [mdExportPlugin(SRC_DIR, base, MD_EXPORT)],
  },
})
