// End-to-end smoke test against the RUNNING app (backend :8000 + Vite :5180).
// Uses the locally installed Google Chrome via playwright-core (no browser download).
// Usage: npm run e2e   [E2E_BASE=http://localhost:5180] [E2E_SHOTS=./e2e/screenshots]
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const BASE = process.env.E2E_BASE ?? 'http://localhost:5180'
const SHOTS = process.env.E2E_SHOTS ?? new URL('./screenshots/', import.meta.url).pathname
const DISCLAIMER = 'Educational research prototype only. Not validated for clinical use. This result is not a medical diagnosis.'
const PAGES = ['/', '/dataset', '/models', '/predict', '/explain', '/experiments', '/research']
mkdirSync(SHOTS, { recursive: true })

const results = []
const check = (name, ok, info = '') => { results.push({ name, ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${info ? ` — ${info}` : ''}`) }

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const errors = []
async function newPage(viewport) {
  const page = await browser.newPage({ viewport })
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`[${m.type()}] ${page.url()} ${m.text()}`) })
  page.on('pageerror', (e) => errors.push(`[pageerror] ${page.url()} ${e.message}`))
  page.on('response', (r) => { if (r.status() >= 500) errors.push(`[http ${r.status()}] ${r.url()}`) })
  return page
}

// 1. every page, desktop + mobile
for (const [label, vp] of [['desktop', { width: 1440, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
  const page = await newPage(vp)
  for (const path of PAGES) {
    await page.goto(BASE + path, { waitUntil: 'networkidle' })
    await page.waitForFunction(() => !document.querySelector('[role=status]'), null, { timeout: 15000 }).catch(() => {})
    const disc = await page.getByTestId('disclaimer').first().textContent()
    const alerts = await page.locator('[role=alert]').count()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    const charts = await page.locator('.recharts-surface').count()
    check(`${label} ${path} renders`, disc?.includes(DISCLAIMER) && alerts === 0, `charts=${charts} alerts=${alerts}`)
    if (label === 'mobile') check(`mobile ${path} no horizontal page scroll`, overflow <= 1, `overflow=${overflow}px`)
    await page.screenshot({ path: `${SHOTS}/${label}${path === '/' ? '_home' : path.replace('/', '_')}.png`, fullPage: true })
  }
  await page.close()
}

// 2. prediction flows (real API)
const page = await newPage({ width: 1440, height: 900 })
await page.goto(BASE + '/predict', { waitUntil: 'networkidle' })
const sel = page.getByLabel('Sample ID')
const firstId = await sel.locator('option').nth(1).getAttribute('value')
await sel.selectOption(firstId)
await page.getByRole('button', { name: 'Run model' }).click()
await page.getByTestId('prediction-result').waitFor({ timeout: 15000 })
const known = () => page.getByText(/^Known dataset label:/)  // regex: case-sensitive, avoids the 'Reveal…' button
const hiddenBefore = (await known().count()) === 0
await page.getByRole('button', { name: /Reveal known dataset label/ }).click()
check('sample explorer: label hidden until reveal, then shown', hiddenBefore && (await known().count()) === 1)
check('sample explorer: explanation chart rendered', (await page.getByText('Why the model produced this output').count()) === 1)
await page.screenshot({ path: `${SHOTS}/flow_sample_prediction.png`, fullPage: true })

await page.getByRole('tab', { name: /Manual input/ }).click()
await page.getByRole('button', { name: 'Run model' }).click()
check('manual: empty submit blocked with validation message', (await page.getByText(/30 field\(s\) need attention/).count()) === 1)
await page.getByRole('button', { name: /Fill with training medians/ }).click()
await page.getByRole('button', { name: 'Run model' }).click()
await page.getByTestId('prediction-result').waitFor({ timeout: 15000 })
check('manual: medians produce a prediction', true)
await page.getByLabel('mean area', { exact: true }).fill('90000')
await page.getByRole('button', { name: 'Run model' }).click()
await page.getByText('The model abstained').waitFor({ timeout: 15000 })
check('manual: extreme input -> model abstains', true)
await page.screenshot({ path: `${SHOTS}/flow_manual_abstain.png`, fullPage: true })

await page.getByRole('tab', { name: /CSV batch/ }).click()
const tpl = await (await fetch(BASE + '/api/predict/template.csv')).text()
const bad = tpl.trim().split('\n')
bad.push(bad[1].replace(/^[^,]+,[^,]+/, 'broken,abc'))
await page.getByTestId('csv-input').setInputFiles({ name: 'batch.csv', mimeType: 'text/csv', buffer: Buffer.from(bad.join('\n')) })
await page.getByRole('button', { name: /Upload & predict/ }).click()
await page.getByText('Batch results').waitFor({ timeout: 15000 })
const invalid = await page.getByText(/not a number/).count()
check('csv: batch results + invalid row reported', invalid === 1)
const dl = page.waitForEvent('download')
await page.getByRole('button', { name: /Download CSV/ }).click()
check('csv: results downloadable', (await (await dl).suggestedFilename()) === 'oncosense_predictions.csv')
await page.getByTestId('csv-input').setInputFiles({ name: 'bad.csv', mimeType: 'text/csv', buffer: Buffer.from('sample_id,patient_age\n1,50') })
await page.getByRole('button', { name: /Upload & predict/ }).click()
await page.getByText(/do not match the model schema/).waitFor({ timeout: 15000 })
check('csv: schema mismatch rejected with details', (await page.getByText(/unexpected columns: patient_age/).count()) === 1)
await page.screenshot({ path: `${SHOTS}/flow_csv.png`, fullPage: true })

// 3. charts carry real API data: compare a value shown on the page with the API
await page.goto(BASE + '/models', { waitUntil: 'networkidle' })
const m = await (await fetch(BASE + '/api/models/metrics')).json()
const sel2 = m.models.find((x) => x.key === m.selected_key)
const auc = sel2.cv.roc_auc.mean.toFixed(3)
check('models page shows API CV AUC for selected model', (await page.getByText(new RegExp(`^${auc} ±`)).count()) >= 1, auc)

await browser.close()
// the deliberate bad-CSV upload above yields one expected 422 resource error in the console
const real = errors.filter((e) => !/Download the React DevTools/.test(e) && !/status of 422/.test(e))
check('no console errors/warnings or 5xx responses', real.length === 0, real.slice(0, 5).join(' | '))
const failed = results.filter((r) => !r.ok).length
console.log(`\n${results.length - failed}/${results.length} checks passed`)
process.exit(failed ? 1 : 0)
