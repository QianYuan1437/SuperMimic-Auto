<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

/**
 * Link to a file of the repository: <Src path="ikzoo/solvers/classical.py" symbol="DLS" />.
 * The repository URL / branch come from `themeConfig.repoUrl` / `repoBranch` (config.mts).
 */
const props = defineProps<{
  path: string
  /** Optional symbol (class / function) shown after the path. */
  symbol?: string
  /** Optional custom label (replaces path + symbol). */
  label?: string
}>()

const { theme } = useData()
const href = computed(() => {
  const repo = String(theme.value.repoUrl ?? '').replace(/\/$/, '')
  const branch = theme.value.repoBranch ?? 'main'
  const kind = props.path.endsWith('/') ? 'tree' : 'blob'
  return `${repo}/${kind}/${branch}/${props.path.replace(/^\//, '')}`
})
</script>

<template>
  <a class="src-link" :href="href" target="_blank" rel="noreferrer" :title="href">
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <path
        fill="currentColor"
        d="M5.7 4.3a1 1 0 0 1 0 1.4L3.4 8l2.3 2.3a1 1 0 1 1-1.4 1.4l-3-3a1 1 0 0 1 0-1.4l3-3a1 1 0 0 1 1.4 0Zm4.6 0a1 1 0 0 1 1.4 0l3 3a1 1 0 0 1 0 1.4l-3 3a1 1 0 1 1-1.4-1.4L12.6 8l-2.3-2.3a1 1 0 0 1 0-1.4Z"
      />
    </svg>
    <template v-if="label">{{ label }}</template>
    <template v-else>
      <code>{{ path }}</code><template v-if="symbol"> · <code>{{ symbol }}</code></template>
    </template>
  </a>
</template>

<style scoped>
.src-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  text-decoration: none !important;
  white-space: nowrap;
}
.src-link svg {
  flex: none;
  opacity: 0.75;
}
.src-link code {
  font-size: 0.85em;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 1px 5px;
  border-radius: var(--ik-radius-xs);
}
.src-link:hover code {
  text-decoration: underline;
}
</style>
