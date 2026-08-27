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

test.describe.configure({ timeout: 90_000 })

function observationMetric(page: import('@playwright/test').Page) {
  return page.locator('.evidence-metrics dd').filter({ hasText: /^542,455$/ })
}

test('reviews the sealed official candidate without forecast claims', async ({ page }) => {
  await page.goto('/evidence')

  await expect(page.getByRole('heading', { name: 'Source evidence', exact: true })).toBeVisible()
  await expect(observationMetric(page)).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('.release-card')).toHaveCount(4)
  await expect(page.getByText(/exact normalized model-family identity/i)).toBeVisible()
  await expect(page.getByText(/registration year is not model year/i)).toBeVisible()
  await expect(page.getByText(/candidate does not feed forecasts/i)).toBeVisible()
  await expect(page.getByRole('definition').filter({ hasText: /^0$/ })).toHaveCount(2)

  await page.getByRole('searchbox', { name: 'Search source labels' }).fill('ALFA ROMEO')
  await page.getByRole('button', { name: 'Apply filters' }).click()
  await expect.poll(() => page.evaluate(
    () => new URL(window.location.href).searchParams.get('search'),
  )).toBe('ALFA ROMEO')
  await expect(page.getByRole('cell', { name: /ALFA ROMEO/ }).first()).toBeVisible()
})

for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`evidence review reflows at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/evidence')
    await expect(observationMetric(page)).toBeVisible({ timeout: 60_000 })
    const overflow = await page.evaluate(() => [...document.querySelectorAll<HTMLElement>('*')]
      .filter((element) => (
        element.closest('.evidence-table-wrap') === null &&
        element.getBoundingClientRect().right > window.innerWidth + 1
      ))
      .map((element) => ({
        className: element.className,
        right: Math.round(element.getBoundingClientRect().right),
        tagName: element.tagName,
      })))
    expect(overflow).toEqual([])
    if (process.env.ICOR_CAPTURE_REVIEW === '1') {
      await page.screenshot({
        fullPage: true,
        path: `../.local/review/evidence-${viewport.width === 390 ? 'mobile' : 'desktop'}.png`,
      })
    }
  })
}

test('evidence review is keyboard reachable and has no serious accessibility violations', async ({ page }) => {
  await page.goto('/evidence')
  await expect(observationMetric(page)).toBeVisible({ timeout: 60_000 })
  await page.keyboard.press('Tab')
  await expect(page.locator(':focus')).toBeVisible()

  await page.addScriptTag({ path: axePath })
  const violations = await page.evaluate(async () => {
    const result = await window.axe.run(document, {
      runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] },
    })
    return result.violations.filter(
      ({ impact }) => impact === 'serious' || impact === 'critical',
    )
  })
  expect(violations).toEqual([])
})
