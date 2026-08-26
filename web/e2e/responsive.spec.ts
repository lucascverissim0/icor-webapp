import { expect, test } from '@playwright/test'


for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`has no page overflow at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/planner')
    await expect(page.getByRole('checkbox', { name: 'France' })).toBeVisible()

    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)

    await page.getByRole('button', { name: /View details/ }).first().click()
    await expect(page.getByText('Configuration detail')).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  })
}
