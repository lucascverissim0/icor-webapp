import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const portOffset = Number(process.env.ICOR_E2E_PORT_OFFSET ?? process.pid % 1000)
process.env.ICOR_E2E_PORT_OFFSET = String(portOffset)
const apiPort = Number(process.env.ICOR_E2E_API_PORT ?? 18000 + portOffset)
const webPort = Number(process.env.ICOR_E2E_WEB_PORT ?? 19000 + portOffset)
const baseURL = `http://127.0.0.1:${webPort}`
const coverageDatabase = fileURLToPath(
  new URL(`../.local/e2e-coverage-${portOffset}.sqlite3`, import.meta.url),
)

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: `uv run --project .. python ../scripts/run_planner_dev.py --api-port ${apiPort} --web-port ${webPort}`,
    cwd: '.',
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      ICOR_COVERAGE_DB: coverageDatabase,
    },
  },
})
