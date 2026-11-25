// Takes the screenshots used in the README (saved to docs/screenshots/).
// Start the backend (:8000) and frontend (:5180) first, then run:  npm run screenshots
// Uses your installed Google Chrome.
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright-core'

const BASE = 'http://localhost:5180'
const OUT = new URL('../../docs/screenshots/', import.meta.url).pathname
mkdirSync(OUT, { recursive: true })

const browser = await chromium.launch({ channel: 'chrome', headless: true })
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const shot = async (name, fullPage = false) => {
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${OUT}${name}.png`, fullPage })
  console.log('saved', name)
}

// Predict page
await page.goto(BASE + '/', { waitUntil: 'networkidle' })
await shot('1-predict-empty')
await page.getByRole('button', { name: 'Fill benign example' }).click()
await page.getByRole('button', { name: 'Predict', exact: true }).click()
await page.getByTestId('result').waitFor()
await shot('2-predict-low')
await page.getByRole('button', { name: 'Fill malignant example' }).click()
await page.getByRole('button', { name: 'Predict', exact: true }).click()
await page.getByText('High', { exact: true }).waitFor()
await shot('3-predict-high')

// Models page
await page.goto(BASE + '/models', { waitUntil: 'networkidle' })
await page.getByRole('button', { name: 'Run models' }).click()
await page.getByRole('button', { name: 'Run models' }).waitFor({ timeout: 60000 })
await shot('4-models', true)

// Dataset page
await page.goto(BASE + '/dataset', { waitUntil: 'networkidle' })
await shot('5-dataset', true)

// Phone size
await page.setViewportSize({ width: 390, height: 844 })
await page.goto(BASE + '/', { waitUntil: 'networkidle' })
await shot('6-mobile')

await browser.close()
