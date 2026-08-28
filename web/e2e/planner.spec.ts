import { expect, test } from '@playwright/test'


test('filters, selects, deep-links, and restores planner state', async ({ page }) => {
  await page.goto('/planner')
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeVisible()

  await page.getByRole('checkbox', { name: 'France' }).focus()
  await page.keyboard.press('Space')
  await page.getByRole('button', { name: 'Apply filters' }).focus()
  await page.keyboard.press('Enter')
  expect(decodeURIComponent(page.url())).toContain('market=["FR"]')

  await page.getByRole('button', { name: /View details/ }).first().focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Generation opportunity detail')).toBeVisible()
  expect(decodeURIComponent(page.url())).toContain('market=["FR"]')

  await page.goBack()
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeChecked()

  await page.goForward()
  await expect(page.getByText('Generation opportunity detail')).toBeVisible()
})

test('retries a recoverable configurations failure without losing controls', async ({ page }) => {
  let requests = 0
  await page.route('**/api/v1/planner/configurations*', async (route) => {
    requests += 1
    if (requests <= 2) {
      await route.fulfill({
        contentType: 'application/json',
        status: 500,
        body: JSON.stringify({
          code: 'internal_error',
          message: 'Temporary planner failure.',
          correlation_id: 'e2e-retry',
          field_errors: [],
        }),
      })
      return
    }
    await route.continue()
  })

  await page.goto('/planner')
  const retry = page.getByRole('button', { name: 'Retry' })
  await expect(retry).toBeVisible()
  await retry.focus()
  await page.keyboard.press('Enter')

  await expect(page.getByRole('button', { name: /View details/ }).first()).toBeVisible()
  await expect(page.getByRole('checkbox', { name: 'France' })).not.toBeChecked()
})

test('a missing deep link has a safe planner return', async ({ page }) => {
  await page.goto('/planner/configurations/not-a-configuration?market=FR')

  await expect(page.getByRole('heading', { name: 'Opportunity not found' })).toBeVisible()
  await page.getByRole('link', { name: 'Return to planner' }).click()
  await expect(page.getByRole('checkbox', { name: 'France' })).toBeChecked()
})
