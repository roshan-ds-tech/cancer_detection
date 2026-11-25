// Captures the step-by-step screenshots used in the root README (docs/screenshots/).
// Requires the RUNNING app (backend :8000 + Vite :5180) and a local Google Chrome.
// Usage: npm run screenshots   [E2E_BASE=http://localhost:5180] [API_BASE=http://localhost:8000]
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const BASE = process.env.E2E_BASE ?? 'http://localhost:5180'
const API = process.env.API_BASE ?? 'http://localhost:8000'
const OUT = new URL('../../docs/screenshots/', import.meta.url).pathname
mkdirSync(OUT, { recursive: true })

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

async function open(path) {
  await page.goto(BASE + path, { waitUntil: 'networkidle' })
  await page.waitForFunction(() => !document.querySelector('[role=status]'), null, { timeout: 15000 }).catch(() => {})
}
async function at(heading) {
  await page.getByRole('heading', { name: heading, exact: true }).first().scrollIntoViewIfNeeded()
  await page.evaluate(() => window.scrollBy(0, -90))
  await page.waitForTimeout(300)
}
async function shot(name) {
  await page.screenshot({ path: `${OUT}${name}.png` })
  console.log('saved', name)
}

await open('/'); await shot('01-home')
await open('/dataset'); await shot('02-dataset-overview')
await at('Feature distributions by class'); await shot('03-dataset-distributions')
await at('Feature correlations'); await shot('04-dataset-correlations')
await open('/models'); await shot('05-models-selected')
await at('Algorithm comparison'); await shot('06-models-comparison')
await at('Curves'); await shot('07-models-curves')
await at('Threshold explorer'); await shot('08-models-threshold')
await at('Confusion matrices'); await shot('09-models-confusion')

// Prediction Lab: held-out sample explorer
await open('/predict')
const sel = page.getByLabel('Sample ID')
await sel.selectOption(await sel.locator('option').nth(1).getAttribute('value'))
await shot('10-predict-choose-sample')
await page.getByRole('button', { name: 'Run model' }).click()
const result = page.getByTestId('prediction-result')
await result.waitFor({ timeout: 15000 })
await result.scrollIntoViewIfNeeded(); await page.evaluate(() => window.scrollBy(0, -90))
await shot('11-predict-sample-result')
await page.getByRole('button', { name: /Reveal known dataset label/ }).click()
await page.getByText('Why the model produced this output').evaluate((el) => el.scrollIntoView({ block: 'start' }))
await page.evaluate(() => window.scrollBy(0, -330))
await shot('12-predict-sample-explanation')

// Manual input
await page.evaluate(() => window.scrollTo(0, 0))
await page.getByRole('tab', { name: /Manual input/ }).click()
await page.getByRole('button', { name: /Fill with training medians/ }).click()
await shot('13-predict-manual-form')
await page.getByRole('button', { name: 'Run model' }).click()
await result.waitFor({ timeout: 15000 })
await result.scrollIntoViewIfNeeded(); await page.evaluate(() => window.scrollBy(0, -90))
await shot('14-predict-manual-result')
await page.getByLabel('mean area', { exact: true }).fill('90000')
await page.getByRole('button', { name: 'Run model' }).click()
const abstain = page.getByText('The model abstained')
await abstain.waitFor({ timeout: 15000 })
await abstain.evaluate((el) => el.scrollIntoView({ block: 'center' }))
await shot('15-predict-abstain')

// CSV batch: template + one corrupted row, then a schema mismatch
await page.evaluate(() => window.scrollTo(0, 0))
await page.getByRole('tab', { name: /CSV batch/ }).click()
await shot('16-predict-csv-upload')
const rows = (await (await fetch(`${API}/api/predict/template.csv`)).text()).trim().split('\n')
rows.push(rows[1].replace(/^[^,]+,[^,]+/, 'broken,abc'))
await page.getByTestId('csv-input').setInputFiles({ name: 'batch.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')) })
await page.getByRole('button', { name: /Upload & predict/ }).click()
const batch = page.getByText('Batch results')
await batch.waitFor({ timeout: 15000 })
await batch.scrollIntoViewIfNeeded(); await page.evaluate(() => window.scrollBy(0, -120))
await shot('17-predict-csv-results')
await page.getByTestId('csv-input').setInputFiles({ name: 'bad.csv', mimeType: 'text/csv', buffer: Buffer.from('sample_id,patient_age\n1,50') })
await page.getByRole('button', { name: /Upload & predict/ }).click()
const schemaErr = page.getByText(/do not match the model schema/)
await schemaErr.waitFor({ timeout: 15000 })
await schemaErr.scrollIntoViewIfNeeded(); await page.evaluate(() => window.scrollBy(0, -150))
await shot('18-predict-csv-schema-error')

await open('/explain'); await shot('19-explain-global')
const explSel = page.locator('select:not([aria-label])').first()
await explSel.selectOption(await explSel.locator('option').nth(3).getAttribute('value'))
await page.waitForFunction(() => !document.querySelector('[role=status]'), null, { timeout: 15000 }).catch(() => {})
await page.waitForTimeout(500)
await page.getByRole('heading', { name: 'Sample-level explanation', exact: true }).evaluate((el) => el.scrollIntoView({ block: 'start' }))
await page.evaluate(() => window.scrollBy(0, -60))
await shot('20-explain-local')
await open('/experiments'); await shot('21-experiments')
await open('/research'); await shot('22-research')

await page.goto(`${API}/docs`, { waitUntil: 'networkidle' })
await page.waitForTimeout(800)
await shot('23-api-docs')

const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 })
await mobile.goto(BASE + '/predict', { waitUntil: 'networkidle' })
await mobile.screenshot({ path: `${OUT}24-mobile-predict.png` })
console.log('saved 24-mobile-predict')

await browser.close()
