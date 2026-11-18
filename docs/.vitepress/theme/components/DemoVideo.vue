<script setup lang="ts">
import { withBase } from 'vitepress'

/** Looping demo video from docs/public: <DemoVideo src="/media/x.mp4" caption="..." />. */
defineProps<{ src: string; caption?: string }>()
</script>

<template>
  <figure class="demo-video">
    <video
      :src="withBase(src)"
      :poster="withBase(src.replace(/\.mp4$/, '.jpg'))"
      autoplay
      loop
      muted
      playsinline
      controls
      preload="metadata"
    />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>

<style scoped>
.demo-video {
  margin: 0;
}
.demo-video video {
  width: 100%;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: #000;
  display: block;
}
.demo-video figcaption {
  margin-top: 6px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  text-align: center;
}
</style>
