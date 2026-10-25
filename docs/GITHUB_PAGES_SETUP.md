# GitHub Pages Setup for TranPy Documentation

Quick guide to host your Sphinx documentation on GitHub Pages.

## One-Time Setup (2 minutes)

### 1. Enable GitHub Pages

1. Go to your repository: https://github.com/mahmouddraz/tranpy
2. Click **Settings** (top menu)
3. Click **Pages** (left sidebar)
4. Under "Source", select:
   - Source: **Deploy from a branch**
   - Branch: **gh-pages** (will be created automatically)
   - Folder: **/ (root)**
5. Click **Save**

That's it! Your documentation will be available at:
**https://mahmouddraz.github.io/tranpy**

### 2. Trigger Documentation Build

The documentation will automatically build and deploy when you:
- Push to the `main` branch
- Or manually trigger the workflow

To manually trigger:
1. Go to https://github.com/mahmouddraz/tranpy/actions
2. Click "Deploy Documentation"
3. Click "Run workflow"
4. Click "Run workflow" button

## How It Works

The GitHub Actions workflow (`.github/workflows/docs.yml`):
1. Runs on every push to `main`
2. Installs Sphinx and dependencies
3. Builds HTML documentation
4. Deploys to `gh-pages` branch
5. GitHub Pages serves from `gh-pages` branch

## Verify Deployment

After the first push:
1. Check workflow status: https://github.com/mahmouddraz/tranpy/actions
2. Wait for "Deploy Documentation" to complete (green checkmark)
3. Visit: **https://mahmouddraz.github.io/tranpy**

## Update Documentation

Documentation updates automatically when you:
1. Edit files in `docs/` folder
2. Commit and push to `main`
3. GitHub Actions rebuilds and deploys

```bash
# Example: Update documentation
vim docs/index.rst
git add docs/index.rst
git commit -m "Update documentation"
git push origin main

# Wait ~2 minutes, then view at:
# https://mahmouddraz.github.io/tranpy
```

## Local Preview

Build and preview locally before pushing:

```bash
# Build documentation
.venv/bin/sphinx-build -M html docs docs/_build

# Open in browser
open docs/_build/html/index.html  # Mac
xdg-open docs/_build/html/index.html  # Linux
```

## Troubleshooting

### 404 Error

**Problem**: Documentation page shows 404

**Solutions**:
1. Wait 2-3 minutes after first deployment
2. Check GitHub Actions completed successfully
3. Verify `gh-pages` branch exists
4. Check Settings → Pages shows "Your site is live"

### Documentation Not Updating

**Problem**: Changes don't appear

**Solutions**:
1. Check GitHub Actions workflow completed
2. Clear browser cache (Ctrl+Shift+R)
3. Wait a few minutes (GitHub Pages caches)

### Build Fails

**Problem**: GitHub Actions shows red X

**Solutions**:
1. Check the workflow logs for errors
2. Test build locally: `.venv/bin/sphinx-build -M html docs docs/_build`
3. Fix errors and push again

## Custom Domain (Optional)

To use a custom domain like `docs.tranpy.org`:

1. Add CNAME file to `docs/` folder:
   ```bash
   echo "docs.tranpy.org" > docs/CNAME
   ```

2. Update DNS with your domain provider:
   - Add CNAME record: `docs` → `mahmouddraz.github.io`

3. Update `pyproject.toml`:
   ```toml
   Documentation = "https://docs.tranpy.org"
   ```

## Status

✅ GitHub Actions workflow configured
✅ Auto-deploy on push to main
✅ Documentation URL updated in package
⏳ Pending: Enable GitHub Pages (one-time)

## Next Steps

1. **Push this code** to GitHub
2. **Enable GitHub Pages** in repository settings
3. **Check deployment** at https://mahmouddraz.github.io/tranpy
4. **Update PyPI** with new documentation URL

---

Your documentation will be live at:
**https://mahmouddraz.github.io/tranpy**
