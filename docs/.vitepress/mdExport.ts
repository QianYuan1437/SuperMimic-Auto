/**
 * Portable Markdown export of the documentation pages ("下载 Markdown" button).
 *
 * The page sources are VitePress Markdown with Vue components, `<Src>` links, `:::` containers
 * and page-local <style> blocks. The export keeps the prose and the math ($...$ / $$...$$) as is
 * and rewrites the rest into plain Markdown:
 *   - <Src path="x" />            -> [`x`](repo/blob/main/x)
 *   - <InteractiveDemo ... />     -> a quoted note linking to the live page
 *   - ::: tip Title ... :::       -> > **Title** quoted block
 *   - site-relative links         -> absolute links to the published site
 *   - frontmatter, <script>, <style> are dropped.
 * Output: <outDir>/md/<relativePath>, served in dev by `mdExportPlugin`.
 */
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, posix, relative } from 'node:path'
import type { Plugin } from 'vite'

export interface ExportOptions {
  /** Published site root, ending in '/'. */
  siteUrl: string
  repoUrl: string
  repoBranch: string
  /** Path of the docs directory inside the repository (for "source" links). */
  docsPathInRepo: string
}

const CONTAINER_TITLES: Record<string, string> = {
  tip: '提示',
  info: '说明',
  warning: '注意',
  danger: '警告',
  details: '详情',
}

/** 'methods/dls.md' -> 'methods/dls', 'methods/index.md' -> 'methods/'. */
export function routeOf(relativePath: string): string {
  return relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '')
}

function resolveLink(href: string, relativePath: string, siteUrl: string): string {
  if (/^([a-z]+:|#|mailto:)/i.test(href)) return href
  const [path, hash = ''] = href.split('#')
  if (!path) return href
  const abs = path.startsWith('/')
    ? path.slice(1)
    : posix.normalize(posix.join(posix.dirname(relativePath), path)).replace(/^\.\//, '')
  const clean = abs.replace(/\.md$/, '').replace(/(^|\/)index$/, '$1')
  return siteUrl + clean + (hash ? `#${hash}` : '')
}

function attr(tag: string, name: string): string | undefined {
  return tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1]
}

export function exportMarkdown(src: string, relativePath: string, opts: ExportOptions): string {
  const pageUrl = opts.siteUrl + routeOf(relativePath)
  const repo = opts.repoUrl.replace(/\/$/, '')
  let md = src.replace(/\r\n/g, '\n')

  // frontmatter, scripts, page-local styles
  md = md.replace(/^---\n[\s\S]*?\n---\n/, '')
  md = md.replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>\n?/g, '')

  // <Src> -> repository link
  md = md.replace(/<Src\b[^>]*\/>/g, (tag) => {
    const path = attr(tag, 'path') ?? ''
    const kind = path.endsWith('/') ? 'tree' : 'blob'
    const text = attr(tag, 'label') ?? path + (attr(tag, 'symbol') ? ` · ${attr(tag, 'symbol')}` : '')
    return `[\`${text}\`](${repo}/${kind}/${opts.repoBranch}/${path.replace(/^\//, '')})`
  })

  // <DemoVideo> -> link to the published video file
  md = md.replace(/<DemoVideo\b[^>]*\/>/g, (tag) => {
    const src = (attr(tag, 'src') ?? '').replace(/^\//, '')
    return `[▶ ${attr(tag, 'caption') ?? src}](${opts.siteUrl}${src})`
  })

  // Interactive components (PascalCase tags, self-closing or paired, possibly multi-line)
  const demo = (name: string) =>
    `> **［交互内容：${name}］** 图表 / 三维演示需在网页中运行，请访问 <${pageUrl}>\n`
  md = md.replace(/<([A-Z][A-Za-z0-9]*)\b[^>]*?\/>\n?/g, (_, name: string) => demo(name))
  md = md.replace(/<([A-Z][A-Za-z0-9]*)\b[^>]*>[\s\S]*?<\/\1>\n?/g, (_, name: string) => demo(name))

  // ::: containers -> quoted blocks (outside fenced code)
  const out: string[] = []
  let fence: string | null = null
  let quoting = false
  for (const line of md.split('\n')) {
    const f = line.match(/^\s*(```+|~~~+)/)
    if (f) {
      if (!fence) fence = f[1]
      else if (line.trim().startsWith(fence)) fence = null
    }
    if (!fence) {
      const open = line.match(/^:::+\s*(\w+)\s*(.*)$/)
      if (open && !quoting) {
        const title = open[2].trim() || CONTAINER_TITLES[open[1]] || open[1]
        out.push(`> **${title}**`, '>')
        quoting = true
        continue
      }
      if (/^:::+\s*$/.test(line) && quoting) {
        quoting = false
        continue
      }
    }
    out.push(quoting ? (line ? `> ${line}` : '>') : line)
  }
  md = out.join('\n')

  // site-relative links -> absolute
  md = md.replace(/\]\(([^)\s]+)\)/g, (_, href: string) => `](${resolveLink(href, relativePath, opts.siteUrl)})`)
  md = md.replace(/\bhref="([^"]+)"/g, (_, href: string) => `href="${resolveLink(href, relativePath, opts.siteUrl)}"`)

  md = md.replace(/\n{3,}/g, '\n\n').trim()
  const source = `${repo}/blob/${opts.repoBranch}/${opts.docsPathInRepo}/${relativePath}`
  const header =
    `<!-- 导出自 IK Zoo：${pageUrl}\n     源文件：${source}\n` +
    `     公式为 LaTeX（$…$ / $$…$$）；交互演示与图表请在网页中查看。 -->\n\n`
  return `${header}${md}\n`
}

async function listMarkdown(dir: string, root = dir): Promise<string[]> {
  const out: string[] = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    if (e.name.startsWith('.') || e.name === 'public' || e.name === 'node_modules') continue
    const p = join(dir, e.name)
    if (e.isDirectory()) out.push(...(await listMarkdown(p, root)))
    else if (e.name.endsWith('.md')) out.push(relative(root, p).split('\\').join('/'))
  }
  return out
}

/** Pages that get a download (home pages and pages with `mdDownload: false` are skipped). */
export function exportable(src: string): boolean {
  const fm = src.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? ''
  return !/^layout:\s*home/m.test(fm) && !/^mdDownload:\s*false/m.test(fm)
}

/** Write every exportable page to <outDir>/md/<relativePath> (VitePress buildEnd). */
export async function writeMarkdownExports(srcDir: string, outDir: string, opts: ExportOptions): Promise<number> {
  let n = 0
  for (const rel of await listMarkdown(srcDir)) {
    const src = await readFile(join(srcDir, rel), 'utf8')
    if (!exportable(src)) continue
    const dest = join(outDir, 'md', rel)
    await mkdir(dirname(dest), { recursive: true })
    await writeFile(dest, exportMarkdown(src, rel, opts))
    n++
  }
  return n
}

/** Dev server: serve `<base>md/<relativePath>` on the fly. */
export function mdExportPlugin(srcDir: string, base: string, opts: ExportOptions): Plugin {
  const prefix = `${base}md/`
  return {
    name: 'ik-zoo-md-export',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = decodeURIComponent((req.url ?? '').split('?')[0])
        if (!url.startsWith(prefix) || !url.endsWith('.md')) return next()
        const rel = url.slice(prefix.length)
        if (rel.includes('..')) return next()
        try {
          const src = await readFile(join(srcDir, rel), 'utf8')
          if (!exportable(src)) return next()
          res.setHeader('Content-Type', 'text/markdown; charset=utf-8')
          res.end(exportMarkdown(src, rel, opts))
        } catch {
          next()
        }
      })
    },
  }
}
