import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'


declare global {
  interface Window {
    axe: {
      run: (context?: Element | Document, options?: object) => Promise<{
        violations: { id: string; impact: string | null }[]
      }>
    }
  }
}

const axePath = fileURLToPath(new URL('../node_modules/axe-core/axe.min.js', import.meta.url))

async function seriousViolations(page: import('@playwright/test').Page) {
  await page.addScriptTag({ path: axePath })
  return page.evaluate(async () => {
    const result = await window.axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] } })
    return result.violations.filter(({ impact }) => impact === 'serious' || impact === 'critical')
  })
}

test('reads cars on the road and evidence depth per Eurocode from the navigation', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/opportunities')
  await page.getByRole('navigation', { name: 'Primary' }).getByRole('link', { name: 'Data coverage' }).click()

  await expect(page).toHaveURL(/\/data-coverage/)
  await expect(page.getByRole('heading', { name: 'Cars on the road at end of 2026' })).toBeVisible()
  await expect(page.getByText('Snapshot e2e-fixture')).toBeVisible()
  const kpis = page.getByRole('region', { name: 'Key figures' })
  await expect(kpis.getByText('Cars in the model')).toBeVisible()
  await expect(kpis.getByText('10,400')).toBeVisible()

  const codes = page.getByRole('table', { name: /Eurocodes with cars/ })
  await expect(codes.getByRole('row', { name: /Q201/ })).toBeVisible()
  await expect(codes.getByRole('row', { name: /Q401/ })).toContainText('Reviewed, no cars')
  await expect(page.getByRole('table', { name: /Official cars on the road/ }).getByRole('rowheader', { name: 'DE' })).toBeVisible()

  expect(await seriousViolations(page)).toEqual([])

  await page.getByRole('combobox', { name: 'Tier' }).selectOption('C')
  await expect(codes.getByRole('row', { name: /Q301/ })).toBeVisible()
  await expect(codes.getByRole('row', { name: /Q201/ })).toHaveCount(0)
  await expect(page).toHaveURL(/tier=C/)
})

test('keeps the data coverage page inside a 390px screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/data-coverage')

  await expect(page.getByRole('heading', { name: 'Cars on the road at end of 2026' })).toBeVisible()
  await expect(page.getByRole('table', { name: /Eurocodes with cars/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  expect(await seriousViolations(page)).toEqual([])
})
