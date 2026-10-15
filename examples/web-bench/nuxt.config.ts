// https://nuxt.com/docs/api/configuration/nuxt-config
import { createRequire } from 'node:module'

// `@import "tailwindcss"` in app/assets/css/main.css: tailwindcss is a dependency of @nuxt/ui, not
// of this app, so pnpm does not link it into node_modules/ and the import cannot be resolved from
// the app. Resolve Tailwind's stylesheet from @nuxt/ui instead.
const resolveFromApp = createRequire(import.meta.url).resolve
const tailwindStylesheet = createRequire(resolveFromApp('@nuxt/ui/runtime/index.css')).resolve('tailwindcss/index.css')

export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@nuxt/eslint', '@nuxt/ui'],
  // Tailwind CSS + Nuxt UI styles (the templates use Tailwind utility classes)
  css: ['~/assets/css/main.css'],
  ssr: false,
  vite: {
    resolve: {
      // Exact match only: JS imports such as `tailwindcss/colors` (used by @nuxt/ui) resolve as usual
      alias: [{ find: /^tailwindcss$/, replacement: tailwindStylesheet }],
    },
    optimizeDeps: {
      include: [
        '@tensorflow/tfjs',
        '@tensorflow/tfjs-core',
        '@tensorflow/tfjs-converter',
        '@tensorflow/tfjs-backend-webgl',
        'long',
      ],
    },
    define: {
      global: 'globalThis',
    },
  },
})
