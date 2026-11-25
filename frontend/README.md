# OncoSense AI: frontend

React 19 + TypeScript + Vite + Tailwind CSS 4 + Recharts. See the [root README](../README.md) for the full setup.

```bash
npm install
npm run dev     # http://localhost:5180 (proxies /api and /static to the FastAPI backend on :8000)
npm test        # Vitest + Testing Library
npm run e2e     # headless Chrome smoke test against the running app
npm run screenshots  # regenerate the README screenshots in ../docs/screenshots
npm run build   # production build -> dist/
```
