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

async function selectVehicle(page: import('@playwright/test').Page) {
  await page.getByLabel('Brand').selectOption({ label: 'Aurora Mobility' })
  await page.getByLabel('Model', { exact: true }).selectOption({ label: 'A1 Horizon' })
  await page.getByLabel('Model year').selectOption('2025')
}

test('exact coverage create, edit, and delete refetches committed ranking', async ({ page }) => {
  await page.goto('/opportunities')
  await expect(page.getByText('Where demand and readiness meet')).toBeVisible()
  await selectVehicle(page)
  await page.getByLabel('Exact configuration / SKU').selectOption(
    'demo-aurora-a1-camera-fr-2030',
  )
  await page.getByRole('button', { name: 'Save exact coverage' }).click()

  await expect(page.getByText('Production coverage saved.')).toBeVisible()
  const exactSummary = page.locator('.opportunity-summary > div').filter({
    hasText: 'Exact-covered base',
  })
  await expect(exactSummary.getByRole('definition')).toHaveText('250')

  await page.getByRole('button', { name: /Edit Aurora Mobility A1 Horizon/ }).click()
  await page.getByLabel('Planner note').fill('E2E edit confirmation')
  await page.getByRole('button', { name: 'Update coverage' }).click()
  await expect(page.getByText('Production coverage updated.')).toBeVisible()
  await expect(page.getByText('E2E edit confirmation')).toBeVisible()

  await page.getByRole('button', { name: /Delete Aurora Mobility A1 Horizon/ }).click()
  await expect(page.getByText(/ranking refreshes only after deletion/i)).toBeVisible()
  await page.getByRole('button', { name: /Confirm delete Aurora Mobility/ }).click()
  await expect(page.getByText('Production coverage deleted.')).toBeVisible()
  await expect(exactSummary.getByRole('definition')).toHaveText('0')
})

test('fallback coverage requires confirmation and shows lower precision', async ({ page }) => {
  await page.goto('/opportunities?groupBy=model_year')
  await selectVehicle(page)
  await page.getByLabel('Exact configuration unknown').check()

  const save = page.getByRole('button', { name: 'Save fallback coverage' })
  await expect(save).toBeDisabled()
  await expect(page.getByText(/half readiness weight/i)).toBeVisible()
  await page.getByLabel(/I understand this is lower precision/i).check()
  await save.click()

  await expect(page.getByText('Production coverage saved.')).toBeVisible()
  await expect(page.getByText(/Vehicle-year fallback/)).toBeVisible()
  await page.getByRole('button', { name: /Delete Aurora Mobility A1 Horizon/ }).click()
  await page.getByRole('button', { name: /Confirm delete Aurora Mobility/ }).click()
  await expect(page.getByText('Production coverage deleted.')).toBeVisible()
})

for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`opportunities has no page overflow at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/opportunities')
    await expect(page.getByText('Where demand and readiness meet')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    if (process.env.ICOR_CAPTURE_REVIEW === '1') {
      await page.screenshot({
        fullPage: true,
        path: `../.local/review/opportunities-${viewport.width === 390 ? 'mobile' : 'desktop'}.png`,
      })
    }
  })
}

test('opportunities is keyboard reachable and has no serious accessibility violations', async ({ page }) => {
  await page.goto('/opportunities')
  await expect(page.getByText('Where demand and readiness meet')).toBeVisible()
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
