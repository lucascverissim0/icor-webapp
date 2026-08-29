import { expect, test } from '@playwright/test'

const usesRealCandidate = Boolean(process.env.ICOR_E2E_EVIDENCE_CANDIDATE)
const expectedTotal = usesRealCandidate ? '10,506,946' : '600'
const expectedModelCount = usesRealCandidate ? '6,929' : '3'
const expectedTopRegistrations = usesRealCandidate ? '257,883' : '300'

test.describe.configure({ timeout: 90_000 })

test('serves the promoted official EU27 registration ranking', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Official 2024 registrations' })).toBeVisible()
  await expect(page.getByText(expectedTotal)).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText(expectedModelCount, { exact: true })).toBeVisible()
  await expect(page.getByText('SANDERO', { exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText(expectedTopRegistrations, { exact: true })).toBeVisible()
  await expect(page.getByText(/registration year is not model year/i)).toBeVisible()
  await expect(page.getByText(/windshield fitment and replacement forecasts are not inferred/i)).toBeVisible()

  await page.getByRole('searchbox', { name: 'Search make or model' }).fill('Tesla')
  await page.getByRole('button', { name: 'Search registrations' }).click()

  await expect.poll(() => page.evaluate(
    () => new URL(window.location.href).searchParams.get('search'),
  )).toBe('Tesla')
  await expect(page.getByText('MODEL Y', { exact: true })).toBeVisible({ timeout: 60_000 })
})

for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`official registrations reflow at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.goto('/registrations')
    await expect(page.getByText(expectedTotal)).toBeVisible({ timeout: 60_000 })

    const overflow = await page.evaluate(() => [...document.querySelectorAll<HTMLElement>('*')]
      .filter((element) => (
        element.closest('.registration-table-wrap') === null
        && element.getBoundingClientRect().right > window.innerWidth + 1
      ))
      .map((element) => ({
        className: element.className,
        right: Math.round(element.getBoundingClientRect().right),
        tagName: element.tagName,
      })))

    expect(overflow).toEqual([])
  })
}
