import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    globals: true,
    testTimeout: 30000,
    include: ['tests/**/*.test.ts?(x)', 'src/**/*.test.ts?(x)'],
    exclude: ['tests/**/*.visual.spec.ts'],
  },
})
