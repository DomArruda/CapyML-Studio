# CapyML Studio

A 100% client-side, local-first machine learning & statistics studio in a **single HTML file**.
Powered by [Pyodide](https://pyodide.org) (WebAssembly Python) with **scikit-learn**, **SciPy**, **pandas** and **NumPy**, plus [Plotly](https://plotly.com/javascript/), [transformers.js](https://huggingface.co/docs/transformers.js) (ONNX), Tailwind CSS, SheetJS/ExcelJS and a shadcn-inspired dark theme.

**No server. No account. No data leaves your browser.**

## Quick start

Open `index.html` in any modern browser (Chrome/Edge/Firefox). That's it.

- First launch downloads the Python runtime + libraries (~50 MB) **once**, then stores them in IndexedDB (`CapyML_WasmCache`) — every launch after that is fully offline (status badge shows `Offline Ready`).
- Optional local runtime: drop a Pyodide distribution in `./pyodide/` next to `index.html` and it will be picked up automatically.
- `Purge cache` in the header wipes the cached binaries if you ever need to.

## Workspaces

| Tab | What it does |
|---|---|
| **Data & Survey** | Load CSV / TSV / JSON / Excel (drag & drop, picker, or paste), dataset summary, right-click a column to coerce its type (number ↔ text ↔ integer), and a guided "What are you trying to figure out?" survey that routes you to the right pre-configured tool |
| **Analytics & Visuals** | Tableau-style visual builder (drag fields onto the X / Y / Color shelves, blue dimensions & green measures A–Z, smart chart-type buttons, per-dimension histogram/box facets, X/Y swap) + interactive correlation matrix (Pearson/Spearman, significance masking) |
| **Modeling & Hypothesis Lab** | Linear (OLS / Ridge / LASSO / ElasticNet), Classification (Logistic / Decision Tree / Gradient Boosting), Non-linear regression (Polynomial / Tree / Boosting), seven hypothesis tests with plain-English interpretation, and in-browser sentiment analysis (DistilBERT SST-2 / Twitter RoBERTa) |
| **CapyFlow Canvas** | Alteryx-style node pipelines: Input → Filter → Transform → Split → Model → Evaluate, drag-and-drop nodes, type-checked ports, zoom/pan, run + inspect any step |

## Export

- **Export Excel** — dataset + every result table (model metrics, coefficients, correlation pairs, test statistics) + PNG snapshots of all rendered charts, in one `.xlsx`
- **Export CSV** — the current dataset

## Built-in samples

Every workspace proves itself with one click: Housing Prices (regression), Customer Churn (classification), A/B Campaign (hypothesis testing), Customer Reviews (sentiment), and an example pipeline on the canvas. All samples are synthetic, generated locally with a fixed seed.

## Verify / develop

The app is a single `index.html` — the Python engine lives in an embedded `<script type="text/python">` block and is executed by Pyodide at runtime.

```bash
# dev-only: run the offline verification harness
# (needs python + scikit-learn/scipy/pandas/numpy, and node on PATH)
python tools/verify.py
```

The harness compiles the embedded Python, runs real sklearn/scipy workloads against it (all model families, all tests, all chart kinds, a full pipeline + cycle detection), `node --check`s every JS block, and cross-checks all referenced DOM ids.

## Notes

- All dynamically rendered HTML passes through [DOMPurify](https://github.com/cure53/DOMPurify) (XSS-safe rendering; Plotly's own internals are left untouched).
- Chart palettes: color-blind-safe **Okabe–Ito** by default (Tableau 10 / Classic / Grayscale available; persisted in localStorage).
- Sentiment models (~27–50 MB) download once from the ONNX CDN and are cached like everything else; scoring is capped at 100 rows in small batches.
- Light/dark theme toggle in the header.
- Everything is for education/exploration: the synthetic samples are demos, and the models are real scikit-learn but trained in your browser tab.

## License

MIT — see [LICENSE](LICENSE).