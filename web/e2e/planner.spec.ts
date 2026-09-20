import { expect, test } from '@playwright/test'


test('searches a model year and forecasts every requested market', async ({ page }) => {
  await page.goto('/planner')
  await page.getByRole('searchbox', { name: 'Search brand or model' }).fill('Golf')
  await page.getByRole('button', { name: 'Search vehicles' }).click()
  await expect(page.getByRole('combobox', { name: 'Brand' })).toHaveValue('Volkswagen')
  await expect(page.getByRole('combobox', { name: 'Model', exact: true })).toHaveValue('Golf')
  await page.getByRole('combobox', { name: 'Model year' }).selectOption('2020')
  await page.getByRole('button', { name: 'Calculate forecast' }).click()

  await expect(page.getByRole('heading', { name: 'Volkswagen Golf · Golf Mk8' })).toBeVisible()
  for (const market of ['Europe (EU27)', 'Belgium', 'France', 'Spain', 'The Netherlands', 'United Kingdom (GB; England is not separable)', 'Germany', 'Poland']) {
    await expect(page.getByRole('rowheader', { name: market })).toBeVisible()
  }
  await expect(page.getByText(/Future sales cohorts 2026, 2027, 2028 are not added/i)).toBeVisible()
})

test('supports selecting a generation directly', async ({ page }) => {
  await page.goto('/planner')
  await page.getByRole('combobox', { name: 'Brand' }).selectOption('Volkswagen')
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption('Golf')
  await page.getByRole('radio', { name: 'Generation directly' }).check()
  await page.getByRole('combobox', { name: 'Generation' }).selectOption('volkswagen-golf-mk8-europe')
  await page.getByRole('combobox', { name: 'Forecast year' }).selectOption('2031')
  await page.getByRole('button', { name: 'Calculate forecast' }).click()
  await expect(page.getByText('2031 windshield replacement forecast')).toBeVisible()
})

test('a missing deep link has a safe planner return', async ({ page }) => {
  await page.goto('/planner/configurations/not-a-configuration?market=FR')

  await expect(page.getByRole('heading', { name: 'Opportunity not found' })).toBeVisible()
  await page.getByRole('link', { name: 'Return to planner' }).click()
  await expect(page.getByRole('heading', { name: 'Search by model year or generation' })).toBeVisible()
})
