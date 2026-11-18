import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import './custom.css'
import './ui.css'
import PageActions from './components/PageActions.vue'
import Src from './components/Src.vue'

export default {
  extends: DefaultTheme,
  // Download / copy-as-Markdown / source actions above every documentation page.
  Layout: () => h(DefaultTheme.Layout, null, { 'doc-before': () => h(PageActions) }),
  enhanceApp({ app }) {
    app.component('Src', Src)
  },
} satisfies Theme
