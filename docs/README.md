# Documentation sources

Build locally:

```bash
uv sync --group docs
uv run sphinx-build -W -b html docs docs/_build/html
```

`index.md` is the entry page; `architecture.md` and `validation.md` are the main guides;
`api.md` renders the API reference from the docstrings.
