# How did I make this website

This repository is the source for my personal website: **[Hugo](https://gohugo.io/)** with [Wowchemy / Hugo Blox](https://wowchemy.com/), deployed on **[Netlify](https://www.netlify.com/)**.

Computational pages (`.qmd` under `content/`) are rendered with **[Quarto](https://quarto.org/)** to Hugo-compatible Markdown before `hugo` runs (see `_quarto.yml` and `netlify.toml`). The committed `_freeze/` directory lets Netlify build without a full R installation; after you change R or Python code in a `.qmd`, run `quarto render` locally and commit the updated `_freeze/` and generated `index.md` / `_index.md` outputs.

The teaching-finance post uses a temporary `biblio.bib` (placeholder entries) so citations resolve; replace that file with your real bibliography when you have it.
