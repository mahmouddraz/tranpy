# TranPy Documentation

This directory contains the Sphinx documentation for TranPy.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or with uv:

```bash
uv pip install -e ".[docs]"
```

### Build HTML Documentation

#### Using Make (Linux/Mac):

```bash
cd docs
make html
```

#### Using Sphinx directly:

```bash
sphinx-build -M html docs docs/_build
```

#### With the virtual environment:

```bash
/Users/mahmouddraz/code/tranpy/.venv/bin/sphinx-build -M html docs docs/_build
```

### View Documentation

After building, open the documentation in your browser:

```bash
# Mac
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html

# Windows
start docs/_build/html/index.html
```

### Clean Build

To remove all built documentation:

```bash
cd docs
make clean
```

Or manually:

```bash
rm -rf docs/_build
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── Makefile             # Build automation
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── api/                 # API reference
│   ├── datasets.rst
│   ├── models.rst
│   ├── explainers.rst
│   ├── simulation.rst
│   ├── data.rst
│   └── utils.rst
└── _build/              # Generated documentation (gitignored)
    └── html/            # HTML output
```

## Adding New Documentation

### Create New Page

1. Create a new `.rst` file in the appropriate directory
2. Add it to the table of contents in `index.rst`

Example:

```rst
.. toctree::
   :maxdepth: 2

   new_page
```

### Document New Module

1. Create a new API reference file in `api/`
2. Use automodule directives:

```rst
New Module
==========

.. automodule:: tranpy.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Add to `api/` toctree in `index.rst`

## Troubleshooting

### Missing Module Errors

If Sphinx can't find tranpy modules:

1. Ensure you're in the project root directory
2. Install tranpy in development mode: `pip install -e .`
3. Check `conf.py` has correct path: `sys.path.insert(0, os.path.abspath('../src'))`

### Build Warnings

Most warnings about missing references can be ignored. To suppress specific warnings, update `conf.py`.

### Theme Issues

If the Read the Docs theme doesn't load:

```bash
pip install sphinx-rtd-theme
```

## Continuous Documentation

### Auto-rebuild on Changes

Install sphinx-autobuild:

```bash
pip install sphinx-autobuild
```

Run:

```bash
sphinx-autobuild docs docs/_build/html
```

Then open http://127.0.0.1:8000 in your browser.

## Publishing Documentation

### Read the Docs

The documentation is ready to be published on Read the Docs:

1. Connect your GitHub repository to Read the Docs
2. Configuration is already in `docs/conf.py`
3. Documentation will auto-build on commits

### GitHub Pages

To publish on GitHub Pages:

```bash
# Build documentation
make html

# Copy to gh-pages branch
git checkout -b gh-pages
cp -r _build/html/* .
git add .
git commit -m "Update documentation"
git push origin gh-pages
```

## Documentation Standards

### Docstring Format

Use Google or NumPy style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Examples:
        >>> my_function(1, "test")
        True
    """
    pass
```

### RST Formatting

- Use proper headers (=, -, ~)
- Add code blocks with language specification
- Use cross-references with :func:, :class:, :mod:
- Include examples in docstrings

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
