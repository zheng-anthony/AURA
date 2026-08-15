# AURA project showcase

A lightweight, dependency-free landing page for **AURA — Automated Urban Road Assessment**. The site is plain HTML and CSS, so there is no build command or package installation.

## Preview locally

From the repository root, run:

```bash
python -m http.server 8000 --directory github-pages
```

Then open `http://localhost:8000`.

## Publish with GitHub Pages

The repository includes `.github/workflows/deploy-pages.yml`, which uploads this directory and deploys it with GitHub Actions.

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Run the **Deploy AURA showcase to GitHub Pages** workflow, or push a change to `main` inside `github-pages/`.

The published URL will appear in the workflow summary after deployment.

## Files

- `index.html` — page structure and copy
- `styles.css` — layout, colors, responsive behavior, and illustration
- `favicon.svg` — browser icon
