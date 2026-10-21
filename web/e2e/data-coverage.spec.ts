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

function countryRow(page: import('@playwright/test').Page, market: string) {
  return page.getByRole('table', { name: /Official cars on the road/ }).getByRole('row').filter({
    has: page.getByRole('rowheader', { name: market, exact: true }),
  })
}

test('reads the estimated missing cars and the age bands of each market', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/data-coverage')

  const countries = page.getByRole('table', { name: /Official cars on the road/ })
  await expect(countries).toBeVisible()
  // Largest gap first: FR is 9,000 - 2,500 = 6,500 short, DE 11,000 - 6,700 = 4,300.
  await expect(countries.getByRole('rowheader')).toHaveText(['FR', 'DE'])
  await expect(page.getByText(/Estimated missing cars are country totals only/)).toContainText(
    'never assigned to a model or a Eurocode',
  )

  // DE: a 2025 cohort of 2,400 and 300 from 2024 are under two years (2,700); 4,000 from 2022 is 2-4 years.
  const de = countryRow(page, 'DE')
  await expect(de.getByRole('cell', { name: '60.9%', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '4,300', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '10,000', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '67.0%', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '1,000 / 2,700', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '2,500 / 4,000', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '3,000 / 0', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '3,500 / 0', exact: true })).toBeVisible()
  await expect(de.getByRole('cell', { name: '1,000 / 0', exact: true })).toBeVisible()

  // The split of the 4,300 is reachable from the keyboard.
  await de.getByText('Split').focus()
  await page.keyboard.press('Enter')
  await expect(de.getByRole('listitem')).toHaveText([
    'Registered before model-level sources1,600',
    'Registrations without a model name300',
    'Used imports and other2,400',
  ])

  // FR publishes a total only, flagged imputed.
  const fr = countryRow(page, 'FR')
  await expect(fr.getByRole('cell', { name: /9,000/ })).toContainText('i')
  await expect(fr.getByRole('cell', { name: '6,500', exact: true })).toBeVisible()
  await expect(fr.getByText(/No age split published/)).toBeVisible()
  await expect(page.getByText(/Eurostat flags on a figure/)).toContainText('i imputed')

  const notes = page.getByRole('list', { name: 'Notes on the countries' })
  await expect(notes).toContainText('DE: 2 registration years count as "less than 2 years"')
  await expect(notes).not.toContainText('age bands from')

  await page.getByText('DE: age bands and the estimate split').click()
  const bands = page.getByRole('table', { name: /Age bands for DE/ })
  const young = bands.getByRole('row', { name: /^<2 y/ })
  await expect(young.getByRole('cell')).toHaveText(['1,000', '2,700', '0', '0', '0', '0', '1,700'])
  const old = bands.getByRole('row', { name: /^10–19 y/ })
  await expect(old.getByRole('cell')).toHaveText(['3,500', '0', '3,500', '1,200', '300', '2,000', '—'])
  expect(await seriousViolations(page)).toEqual([])
})

test('keeps the data coverage page inside a 390px screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/data-coverage')

  await expect(page.getByRole('heading', { name: 'Cars on the road at end of 2026' })).toBeVisible()
  await expect(page.getByRole('table', { name: /Eurocodes with cars/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  expect(await seriousViolations(page)).toEqual([])

  // Opening the splits and the band detail widens the tables, never the page.
  await countryRow(page, 'DE').getByText('Split').click()
  await page.getByText('DE: age bands and the estimate split').click()
  await expect(page.getByRole('table', { name: /Age bands for DE/ })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  expect(await seriousViolations(page)).toEqual([])
})
