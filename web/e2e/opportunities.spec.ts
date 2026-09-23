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

test('opens model-year opportunities from the app home', async ({ page }) => {
  await page.goto('/')

  await expect(page).toHaveURL(/\/opportunities/)
  await expect(page.getByText('Prioritized model and generation opportunities')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Model years' })).toHaveAttribute(
    'aria-pressed',
    'true',
  )
})

test('a searched ranking shows the same score as an unfiltered one', async ({ page }) => {
  // The user-visible form of the whole-market rule: the number on the card is a
  // property of the vehicle, so narrowing the list must not move it.
  await page.goto('/opportunities?groupBy=model_year')
  const card = page.locator('.opportunity-card').first()
  await expect(card).toBeVisible()
  const heading = (await card.getByRole('heading').first().textContent()) ?? ''
  const unfiltered = await card.locator('.opportunity-score strong').textContent()
  const basis = page.getByRole('note').filter({ hasText: /whole-market/i })
  await expect(basis).toBeVisible()
  const population = (await basis.textContent()) ?? ''

  await page.goto(`/opportunities?groupBy=model_year&q=${encodeURIComponent(heading.split(' ')[0])}`)
  const narrowed = page.locator('.opportunity-card').first()
  await expect(narrowed).toBeVisible()

  await expect(narrowed.locator('.opportunity-score strong')).toHaveText(unfiltered ?? '')
  await expect(page.getByRole('note').filter({ hasText: /whole-market/i })).toHaveText(population)
})

test('opens a reloadable explanation for an individual ranking', async ({ page }) => {
  await page.setViewportSize({ width: 1800, height: 900 })
  await page.goto('/opportunities?groupBy=model_year')
  await page.getByRole('button', { name: /View Aurora Mobility.*2025 details/i }).click()

  await expect(page).toHaveURL(/\/opportunities\/model_year-/)
  await expect(page.locator('.opportunity-detail-context')).toBeVisible()
  await expect(page.locator('.opportunity-card--selected')).toBeVisible()
  await expect(page.getByLabel('Selected opportunity').getByRole('heading', { name: /Aurora Mobility.*A1 Horizon.*2025/i })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Why this opportunity ranks here' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Estimated active fleet' })).toBeVisible()
  await expect(page.getByText('Total estimated fleet').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Market and horizon contributions' })).toBeVisible()
  await page.reload()
  await expect(page.locator('.opportunity-detail-hero').getByText('Demo generation A')).toBeVisible()
  await expect(page.getByRole('link', { name: /Back to opportunity ranking/i })).toBeVisible()
})

test('next, last, and previous pagination follow requested route state', async ({ page }) => {
  await page.route('**/api/v1/opportunities?**', async (route) => {
    const response = await route.fetch()
    const body = await response.json() as Record<string, unknown>
    const requestedPage = Number(new URL(route.request().url()).searchParams.get('page') ?? 1)
    if (requestedPage > 1) {
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
    await route.fulfill({
      response,
      json: { ...body, page: requestedPage, pages: 3, total: 300 },
    })
  })
  await page.goto('/opportunities')
  const pages = page.getByRole('navigation', { name: 'Opportunity pages' })

  await pages.getByRole('button', { name: 'Next page' }).click()
  await expect(page).toHaveURL(/page=2/)
  await expect(pages).toContainText('Loading page 2')
  await expect(pages.getByRole('button', { name: 'Next page' })).toBeDisabled()
  await expect(pages).toContainText('Page 2 of 3')

  await pages.getByRole('button', { name: 'Last page' }).click()
  await expect(page).toHaveURL(/page=3/)
  await expect(pages).toContainText('Page 3 of 3')

  await pages.getByRole('button', { name: 'Previous page' }).click()
  await expect(page).toHaveURL(/page=2/)
  await expect(pages).toContainText('Page 2 of 3')
})

test('exact coverage create, edit, and delete refetches committed ranking', async ({ page }) => {
  await page.goto('/opportunities')
  await expect(page.getByText('Prioritized model and generation opportunities')).toBeVisible()
  await page.getByText('Manage ICOR worked-model coverage').click()
  await selectVehicle(page)
  await page.getByLabel('Exact configuration / SKU').selectOption(
    'demo-aurora-a1-camera-fr-2030',
  )
  await page.getByRole('button', { name: 'Save exact coverage' }).click()

  await expect(page.getByText('Production coverage saved.')).toBeVisible()
  const exactSummary = page.locator('.opportunity-summary > div').filter({
    hasText: 'Exact ICOR coverage',
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
  await page.getByText('Manage ICOR worked-model coverage').click()
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

for (const viewport of [{ width: 390, height: 844 }, { width: 1100, height: 900 }, { width: 1440, height: 900 }]) {
  test(`opportunities has no page overflow at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/opportunities')
    await expect(page.getByText('Prioritized model and generation opportunities')).toBeVisible()
    const firstDetailsButton = page.getByRole('button', { name: /View .* details/i }).first()
    await expect(firstDetailsButton).toBeVisible()
    expect(await firstDetailsButton.evaluate((button) => {
      const bounds = button.getBoundingClientRect()
      return bounds.left >= 0 && bounds.right <= window.innerWidth
    })).toBe(true)
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
  await expect(page.getByText('Prioritized model and generation opportunities')).toBeVisible()
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

test('opportunity workspace uses the requested sharp corners', async ({ page }) => {
  await page.goto('/opportunities')
  await expect(page.getByText('Prioritized model and generation opportunities')).toBeVisible()

  for (const selector of [
    '.score-method',
    '.dataset-coverage',
    '.opportunity-summary > div',
    '.opportunity-ranking',
    '.opportunity-card button',
  ]) {
    const radius = await page.locator(selector).first().evaluate(
      (element) => getComputedStyle(element).borderRadius,
    )
    expect(radius).toBe('0px')
  }
})
