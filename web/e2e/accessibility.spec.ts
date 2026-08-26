import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'


declare global {
  interface Window {
    axe: {
      run: (context?: Element | Document, options?: object) => Promise<{ violations: { id: string; impact: string | null }[] }>
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

test('primary planner routes have no serious accessibility violations', async ({ page }) => {
  await page.goto('/planner')
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeVisible()
  expect(await seriousViolations(page)).toEqual([])

  await page.getByRole('button', { name: /View details/ }).first().click()
  await expect(page.getByText('Configuration detail')).toBeVisible()
  expect(await seriousViolations(page)).toEqual([])

  await page.goto('/planner/configurations/not-a-configuration')
  await expect(page.getByRole('heading', { name: 'Configuration not found' })).toBeVisible()
  expect(await seriousViolations(page)).toEqual([])
})

test('keyboard focus is visible', async ({ page }) => {
  await page.goto('/planner')
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeVisible()

  await page.keyboard.press('Tab')
  const focused = page.locator(':focus')
  await expect(focused).toBeVisible()
  expect(await focused.evaluate((element) => {
    const style = getComputedStyle(element)
    return style.outlineWidth !== '0px' || style.boxShadow !== 'none'
  })).toBe(true)
})
