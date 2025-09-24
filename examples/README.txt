Examples Gallery
================

This gallery showcases how to use the estimators provided by ``sklekmeans``.

Contents
--------

- ``example.py``: Minimal example demonstrating a basic ``EKMeans`` fit.
- ``plot_imbalanced_ekmeans.py``: Comparison of clustering on an imbalanced dataset.

How to run locally
------------------

You can execute the examples directly, or build the documentation to render the
plots and notebooks via Sphinx-Gallery.

To build the documentation gallery locally:

1. Install the docs dependencies:

   ``pip install -e .[docs]``

2. Build the docs:

   ``sphinx-build -E -b html doc doc\_build\html``

The gallery will appear under ``doc/_build/html/auto_examples``.
