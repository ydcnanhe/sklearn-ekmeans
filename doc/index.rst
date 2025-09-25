#############################################
sklekmeans: a scikit-learn extension
#############################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/ydcnanhe/sklearn-eqkmeans>`__ |
`Issues & Ideas <https://github.com/ydcnanhe/sklearn-eqkmeans/issues>`__ |

This is the documentation for `sklekmeans` to help at extending
`scikit-learn`. It provides some information on how to build your own custom
`scikit-learn` compatible estimators as well as a template to package them.


.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :class-card: intro-card intro-card-title-lg
        :shadow: md
        :link: quick_start
        :link-type: doc
        :text-align: center
        
        .. image:: _static/img/index_getting_started.gif
            :height: 96
            :class: intro-card-img

        Learn how to install, fit, and evaluate EKMeans on your data.

    .. grid-item-card::  User guide
        :class-card: intro-card intro-card-title-lg
        :shadow: md
        :link: user_guide
        :link-type: doc
        :text-align: center
        
        .. image:: _static/img/index_user_guide.gif
            :height: 96
            :class: intro-card-img

        Concepts, guidance, and detailed usage of EKMeans and MiniBatchEKMeans.

    .. grid-item-card::  API reference
        :class-card: intro-card intro-card-title-lg
        :shadow: md
        :link: api
        :link-type: doc
        :text-align: center
        
        .. image:: _static/img/index_api.gif
            :height: 96
            :class: intro-card-img

        Full API reference for sklekmeans estimators, functions, and utilities.

    .. grid-item-card::  Examples
        :class-card: intro-card intro-card-title-lg
        :shadow: md
        :link: auto_examples/index
        :link-type: doc
        :text-align: center
        
        .. image:: _static/img/index_examples.gif
            :height: 96
            :class: intro-card-img

        Practical examples demonstrating clustering on balanced and imbalanced datasets.


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quick_start
    user_guide
    api
    benchmarks
    auto_examples/index

References
==========

.. [1] Y. He. *An Equilibrium Approach to Clustering: Surpassing Fuzzy
    C-Means on Imbalanced Data*, IEEE Transactions on Fuzzy Systems, 2025.
.. [2] Y. He. *Imbalanced Data Clustering Using Equilibrium K-Means*,
    arXiv, 2024.
