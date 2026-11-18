<script setup lang="ts">
/** Top-of-page actions on documentation pages: download / copy the page as Markdown, view its source. */
import { computed, ref } from 'vue'
import { useData, withBase } from 'vitepress'

const { page, frontmatter, theme } = useData()

const enabled = computed(() => frontmatter.value.mdDownload !== false && !!page.value.relativePath)
const mdUrl = computed(() => withBase(`/md/${page.value.relativePath}`))
const fileName = computed(
  () => `ik-zoo-${page.value.relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '').replace(/\/+$/, '').replace(/\//g, '-') || 'index'}.md`,
)
const sourceUrl = computed(() => {
  const repo = String(theme.value.repoUrl ?? '').replace(/\/$/, '')
  return `${repo}/blob/${theme.value.repoBranch ?? 'main'}/${theme.value.docsPathInRepo ?? 'docs'}/${page.value.relativePath}`
})

const copyState = ref<'idle' | 'done' | 'error'>('idle')
let timer = 0

async function copy() {
  try {
    const res = await fetch(mdUrl.value)
    if (!res.ok) throw new Error(String(res.status))
    await navigator.clipboard.writeText(await res.text())
    copyState.value = 'done'
  } catch {
    copyState.value = 'error'
  }
  window.clearTimeout(timer)
  timer = window.setTimeout(() => (copyState.value = 'idle'), 2000)
}
</script>

<template>
  <div v-if="enabled" class="ik-page-actions">
    <a :href="mdUrl" :download="fileName" title="下载本页的 Markdown 原文（公式为 LaTeX）">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path fill="currentColor" d="M8 1.5a.75.75 0 0 1 .75.75v6.69l2.22-2.22a.75.75 0 1 1 1.06 1.06l-3.5 3.5a.75.75 0 0 1-1.06 0l-3.5-3.5a.75.75 0 1 1 1.06-1.06l2.22 2.22V2.25A.75.75 0 0 1 8 1.5ZM2.75 12a.75.75 0 0 1 .75.75v.75h9v-.75a.75.75 0 0 1 1.5 0v1.5a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75v-1.5a.75.75 0 0 1 .75-.75Z" />
      </svg>
      下载 Markdown
    </a>
    <button type="button" title="复制本页的 Markdown 原文" @click="copy">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path v-if="copyState === 'done'" fill="currentColor" d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
        <path v-else fill="currentColor" d="M5.5 1.75A1.75 1.75 0 0 1 7.25 0h6.5c.97 0 1.75.78 1.75 1.75v6.5A1.75 1.75 0 0 1 13.75 10H12.5v-1.5h1.25a.25.25 0 0 0 .25-.25v-6.5a.25.25 0 0 0-.25-.25h-6.5a.25.25 0 0 0-.25.25V3H5.5V1.75ZM2.25 4.5h6.5c.97 0 1.75.78 1.75 1.75v6.5c0 .97-.78 1.75-1.75 1.75h-6.5A1.75 1.75 0 0 1 .5 12.75v-6.5c0-.97.78-1.75 1.75-1.75Zm0 1.5a.25.25 0 0 0-.25.25v6.5c0 .14.11.25.25.25h6.5a.25.25 0 0 0 .25-.25v-6.5A.25.25 0 0 0 8.75 6h-6.5Z" />
      </svg>
      <span aria-live="polite">{{ copyState === 'done' ? '已复制' : copyState === 'error' ? '复制失败' : '复制 Markdown' }}</span>
    </button>
    <a :href="sourceUrl" target="_blank" rel="noreferrer" title="在 GitHub 上查看本页源文件">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path fill="currentColor" d="M5.7 4.3a1 1 0 0 1 0 1.4L3.4 8l2.3 2.3a1 1 0 1 1-1.4 1.4l-3-3a1 1 0 0 1 0-1.4l3-3a1 1 0 0 1 1.4 0Zm4.6 0a1 1 0 0 1 1.4 0l3 3a1 1 0 0 1 0 1.4l-3 3a1 1 0 1 1-1.4-1.4L12.6 8l-2.3-2.3a1 1 0 0 1 0-1.4Z" />
      </svg>
      源文件
    </a>
  </div>
</template>
