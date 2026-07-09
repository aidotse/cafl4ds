# API Reference

This page is generated **automatically from your code's docstrings** by [mkdocstrings](https://mkdocstrings.github.io/).
At build time it imports the listed modules, reads their docstrings, and renders typed, cross-linked API docs — so the
reference never drifts from the code as long as your docstrings are accurate. This project configures the **Google
docstring style** (`docstring_style: google` in `mkdocs.yaml`).

## Writing docstrings mkdocstrings can render

Write Google-style docstrings — a one-line summary, then `Args:` / `Returns:` / `Raises:` sections. mkdocstrings parses
these into structured API docs:

```python
class Repeat(Transform):
    """Repeats the text a fixed number of times.

    Args:
        times: How many space-separated copies to emit.

    Returns:
        The input repeated ``times`` times, joined by spaces.
    """

    def __init__(self, times: int = 2) -> None:
        self.times = times
```

Type hints are picked up automatically, so you don't repeat types in the docstring.

## Documenting a new module

Add a mkdocstrings **identifier block** — three colons followed by the dotted import path — anywhere on a docs page.
This page already documents the example module:

```text
::: cafl4ds.pipeline
    options:
      show_root_heading: true   # render a heading for the module/class
      show_source: true         # include a collapsible "source" view
      docstring_style: google
```

To document another module, add another block (e.g. `::: cafl4ds.cli`) on this page or any other, and list that page in
the `nav:` of `mkdocs.yaml`.

The live reference for the shipped example module follows:

::: cafl4ds.pipeline
    options:
      show_root_heading: true
      show_source: true
      docstring_style: google

## If your API docs don't appear

mkdocstrings has to *import* the module to introspect it, so a missing block usually means an import problem. Check
that: the dotted path is correct (`cafl4ds.module`, not a file path); the package is installed in the docs environment
(`uv sync` before `uv run poe docs`); and the module imports cleanly on its own. Build with `uv run poe docs --verbose`
to see mkdocstrings' import errors. See the [mkdocstrings docs](https://mkdocstrings.github.io/python/) for the full
option list.
