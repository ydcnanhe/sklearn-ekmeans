# Building the docs locally (Windows)

This guide shows how to build the Sphinx documentation locally on Windows and open the generated site.

## 1) Create/activate a Python environment

Use any environment manager you prefer (venv/conda/mamba). Example with Python's built-in venv:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
```

If you use conda:

```cmd
conda create -n sklekmeans-docs python=3.11 -y
conda activate sklekmeans-docs
```

## 2) Install package and docs dependencies

From the repository root, install the project in editable mode and the docs extras defined in `pyproject.toml`:

```cmd
pip install -e .[docs]
```

If you see an escaping issue, use quotes:

```cmd
pip install -e ".[docs]"
```

## 3) Build the docs

Run Sphinx to generate the HTML site into `_build/html`:

```cmd
sphinx-build -b html doc doc/_build/html
```

Alternatively, use the Makefile wrapper on Windows (requires `make`):

```cmd
make -C doc html
```

## 4) Open the generated site

Open the homepage in your default browser:

```cmd
start doc\_build\html\index.html
```

Or right-click `doc\_build\html\index.html` and open with browser, e.g., Microsoft Edge.

## 5) Clean builds (optional)

To remove previous build artifacts before rebuilding:

```cmd
rmdir /s /q doc\_build
rmdir /s /q doc\auto_examples
rmdir /s /q doc\generated
del doc\sg_execution_times.rst
```

## Tips
- If imports fail during the build, ensure the repository root is on `PYTHONPATH` or that you installed the package with `pip install -e .`.
- If images or styles don't appear, verify static assets under `doc/_static/` and that `html_css_files = ["css/sklekmeans.css"]` is set in `doc/conf.py`.
- To speed up gallery builds while iterating, you can temporarily set `plot_gallery = "False"` in `doc/conf.py`.

## Git ignores (build outputs)

This repository intentionally ignores Sphinx build outputs and gallery artifacts. The following paths are listed in `.gitignore` and should not be committed:

- `doc/_build/` — Sphinx HTML build output
- `doc/generated/` — Sphinx autosummary/numpydoc generated API stubs
- `doc/auto_examples/` — Sphinx-Gallery generated examples and images
- `doc/sg_execution_times.rst` — Sphinx-Gallery execution times report
- `doc_build/` — Local convenience output folder (if used)

If you need to clean these directories, see the Clean builds section above.