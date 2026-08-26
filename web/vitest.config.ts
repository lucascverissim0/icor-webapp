import { configDefaults, defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    exclude: [...configDefaults.exclude, 'e2e/**'],
    globals: true,
    maxWorkers: 1,
    pool: 'threads',
    setupFiles: ['./tests/setup.ts'],
    restoreMocks: true,
  },
})
