# How did I make this website

This repository is the source for my personal website: **[Hugo](https://gohugo.io/)** with [Wowchemy / Hugo Blox](https://wowchemy.com/), deployed on **[Netlify](https://www.netlify.com/)**.

Computational pages (`.qmd` under `content/`) are rendered with **[Quarto](https://quarto.org/)** to Hugo-compatible Markdown before `hugo` runs (see `_quarto.yml` and `netlify.toml`). The committed `_freeze/` directory lets Netlify build without a full R installation; after you change R or Python code in a `.qmd`, run `quarto render` locally and commit the updated `_freeze/` and generated `index.md` / `_index.md` outputs.

The teaching-finance post uses a temporary `biblio.bib` (placeholder entries) so citations resolve; replace that file with your real bibliography when you have it.

The site also ships a **Netlify Function** at `netlify/functions/gaa-vote.mjs` that backs the verdict vote on `/gaa/`. It uses [`@netlify/blobs`](https://www.npmjs.com/package/@netlify/blobs) (declared in `package.json`) as a small key/value store, and is wired in via `netlify.toml` (`[functions] directory = 'netlify/functions'`). A clean local build therefore runs `npm install` as well as `hugo`; on Netlify both happen automatically.
