// Takes the screenshots used in the README (saved to docs/screenshots/) and does a quick check
// of every page: no console errors and no sideways scrolling on a phone.
// Start the backend (:8000) and frontend (:5180) first, then run:  npm run screenshots
// Uses your installed Google Chrome.
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const BASE = 'http://localhost:5180'
const OUT = new URL('../../docs/screenshots/', import.meta.url).pathname
mkdirSync(OUT, { recursive: true })

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
const problems = []
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') problems.push(`${page.url()}: ${m.text()}`) })
page.on('pageerror', (e) => problems.push(`${page.url()}: ${e.message}`))

async function open(path) {
  await page.goto(BASE + path, { waitUntil: 'networkidle' })
  await page.waitForFunction(() => !document.querySelector('[role=status]'), null, { timeout: 30000 }).catch(() => {})
}
async function scrollTo(text) {
  await page.getByRole('heading', { name: text, exact: true }).first().evaluate((el) => el.scrollIntoView({ block: 'start' }))
  await page.evaluate(() => window.scrollBy(0, -70))
}
async function shot(name) {
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${OUT}${name}.png` })
  console.log('saved', name)
}

await open('/'); await shot('01-overview')

await open('/dataset'); await shot('02-dataset')
await scrollTo('Feature correlations'); await shot('03-dataset-correlations')

await open('/models')
await page.getByRole('button', { name: 'Run models' }).click()
await page.getByRole('button', { name: 'Run models' }).waitFor({ timeout: 60000 })
await shot('04-models')
await scrollTo('Accuracy comparison'); await shot('05-models-charts')
await scrollTo('Confusion matrices'); await shot('06-models-confusion')

await open('/predict')
await shot('07-predict-form')
await page.getByRole('button', { name: 'Fill typical malignant values' }).click()
await page.getByRole('button', { name: 'Predict', exact: true }).click()
await page.getByTestId('prediction-result').waitFor()
await page.evaluate(() => window.scrollTo(0, 0))
await shot('08-predict-high')
await scrollTo('What influenced this prediction'); await shot('09-predict-explanation')
await page.getByRole('button', { name: 'Fill typical benign values' }).click()
await page.getByRole('button', { name: 'Predict', exact: true }).click()
await page.getByTestId('prediction-result').waitFor()
await page.evaluate(() => window.scrollTo(0, 0))
await shot('10-predict-low')

await page.getByRole('tab', { name: /Test samples/ }).click()
const sel = page.getByLabel('Sample')
await sel.selectOption(await sel.locator('option').nth(5).getAttribute('value'))
await page.getByRole('button', { name: 'Predict', exact: true }).click()
await page.getByTestId('prediction-result').waitFor()
await page.getByRole('button', { name: /Reveal real diagnosis/ }).click()
await page.evaluate(() => window.scrollTo(0, 0))
await shot('11-predict-test-sample')

await page.getByRole('tab', { name: /CSV batch/ }).click()
const rows = (await (await fetch(`${BASE}/api/predict/template.csv`)).text()).trim().split('\n')
rows.push('broken-row,abc,' + rows[1].split(',').slice(2).join(','))
await page.getByTestId('csv-input').setInputFiles({ name: 'samples.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')) })
await page.getByRole('button', { name: /Upload & predict/ }).click()
await page.getByText('Batch results').waitFor()
await shot('12-predict-csv')

await open('/explain'); await shot('13-explain')
await open('/about'); await shot('14-about')

// phone size: screenshot + check that no page scrolls sideways
await page.setViewportSize({ width: 390, height: 844 })
for (const path of ['/', '/dataset', '/models', '/predict', '/explain', '/about']) {
  await open(path)
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  if (overflow > 1) problems.push(`${path} is ${overflow}px too wide on a phone`)
}
await open('/predict'); await shot('15-mobile')

await browser.close()
if (problems.length) { console.log('PROBLEMS:\n' + problems.join('\n')); process.exit(1) }
console.log('All pages OK: no console errors, no horizontal scrolling on mobile')
