# SIMBSIG = SIMilarity Batched Search Integrated Gpu-based


[![License: BSD](https://img.shields.io/github/license/BorgwardtLab/simbsig)](https://opensource.org/licenses/BSD-3-Clause)
[![Version](https://img.shields.io/pypi/v/simbsig)](https://pypi.org/project/simbsig/)
[![PythonVersion](https://img.shields.io/pypi/pyversions/simbsig)]()
[![Documentation Status](https://readthedocs.org/projects/simbsig/badge/?version=latest)](https://simbsig.readthedocs.io/en/latest/?badge=latest)

SIMBSIG is a GPU accelerated software tool for neighborhood queries, KMeans and PCA which mimics the sklearn API.

The algorithm for batchwise data loading and GPU usage follows the principle of [1]. The algorithm for KMeans follows the Mini-batch KMeans described by Scully [2]. The PCA algorithm follows Halko's method [3].
The API matches sklearn in big parts [4,5], such that code dedicated to sklearn can be simply reused by importing SIMBSIG instead of sklearn. Additional features and arguments for scaling have been added, for example all data input can be either array-like or as a h5py file handle [6].

*Eljas Röllin, Michael Adamer, Lucie Bourguignon, Karsten M. Borgwardt*


## Installation

SIMBSIG is a PyPI package which can be installed via `pip`:

```
pip install simbsig
```

You can also clone the repository and install it locally via [Poetry](https://python-poetry.org/) by executing
```bash
poetry install
```
in the repository directory.

## Example

<!-- Python block-->
```python
>>> X = [[0,1], [1,2], [2,3], [3,4]]
>>> y = [0, 0, 1, 1]
>>> from simbsig import KNeighborsClassifier
>>> knn_classifier = KNeighborsClassifier(n_neighbors=3)
>>> knn_classifier.fit(X, y)
KNeighborsClassifier(...)
>>> print(knn_classifier.predict([[0.9, 1.9]]))
[0]
>>> print(knn_classifier.predict_proba([[0.9]]))
[[0.666... 0.333...]]
```

## Tutorials
Tutorial notebooks with toy examples can be found under [tutorials](https://github.com/BorgwardtLab/simbsig/tree/main/tutorials)

## Documentation

The documentation can be found [here](https://simbsig.readthedocs.io/en/latest/index.html).

## Overview of implemented algorithms

| Class | SIMBSIG | sklearn |
| :---: | :--- | :--- |
| NearestNeighbors | fit | fit |
|  | kneighbors | kneighbors |
|  | radius_neighbors | radius_neighbors |
| KNeighborsClassifier | fit | fit |
|  | predict | predict |
|  | predict_proba | predict_proba |
| KNeighborsRegressor | fit | fit |
|  | predict | predict |
| RadiusNeighborsClassifier | fit | fit |
|  | predict | predict |
|  | predict_proba | predict_proba |
| RadiusNeighborsRegressor | fit | fit |
|  | predict | predict |
| KMeans |  fit | fit|
| | predict | predict |
| | fit_predict | fit_predict |
| PCA | fit | fit |
|  | transform | transform |
|  | fit_transform | fit_transform

## Contact

This code is developed and maintained by members of the Department of Biosystems Science and Engineering at ETH Zurich. It available from the GitHub repo of the [Machine Learning and Computational Biology Lab](https://www.bsse.ethz.ch/mlcb) of [Prof. Dr. Karsten Borgwardt](https://www.bsse.ethz.ch/mlcb/karsten.html).

- [Michael Adamer](https://mikeadamer.github.io/) ([GitHub](https://github.com/MikeAdamer))

*References*:

  [1] Gutiérrez, P. D., Lastra, M., Bacardit, J., Benítez, J. M., & Herrera, F. (2016). GPU-SME-kNN: Scalable and memory efficient kNN and lazy learning using GPUs. Information Sciences, 373, 165-182.

  [2] Sculley, D. (2010, April). Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web (pp. 1177-1178).

  [3] Halko, N., Martinsson, P. G., Shkolnisky, Y., & Tygert, M. (2011). An algorithm for the principal component analysis of large data sets. SIAM Journal on Scientific computing, 33(5), 2580-2594.

  [4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.

  [5] Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. arXiv preprint arXiv:1309.0238.

  [6] Collette, A., Kluyver, T., Caswell, T. A., Tocknell, J., Kieffer, J., Scopatz, A., ... & Hole, L. (2021). h5py/h5py: 3.1. 0. Zenodo.
