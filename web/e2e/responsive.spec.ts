import { expect, test } from '@playwright/test'


for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`has no page overflow at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/planner')
    await expect(page.getByRole('heading', { name: 'Search by model year or generation' })).toBeVisible()

    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    if (process.env.ICOR_CAPTURE_REVIEW === '1') {
      await page.screenshot({
        fullPage: true,
        path: `../.local/review/planner-${viewport.width === 390 ? 'mobile' : 'desktop'}.png`,
      })
    }

    await page.getByRole('combobox', { name: 'Brand' }).fill('Volkswagen')
    await page.getByRole('combobox', { name: 'Model', exact: true }).fill('Golf')
    await page.getByRole('combobox', { name: 'Model year' }).selectOption('2020')
    await page.getByRole('button', { name: 'Calculate forecast' }).click()
    await expect(page.getByRole('heading', { name: 'Volkswagen Golf · Golf Mk8' })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    if (process.env.ICOR_CAPTURE_REVIEW === '1') {
      await page.screenshot({
        fullPage: true,
        path: `../.local/review/detail-${viewport.width === 390 ? 'mobile' : 'desktop'}.png`,
      })
    }
  })
}

test('uses the full detail route when 1440px cannot fit three useful columns', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/planner/configurations/demo-aurora-a1-camera-fr-2030')

  await expect(page.getByRole('heading', { name: 'Aurora Mobility A1 Horizon' })).toBeVisible()
  await expect(page.locator('.planner-detail-context')).toBeHidden()
})
