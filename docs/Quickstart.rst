Quickstart
==========


Installation
------------

SIMBSIG is a PyPI package which can be installed via `pip`:

.. code-block:: bash

   $ pip install simbsig


You can also clone the repository and install it locally via `Poetry <https://python-poetry.org/>`_ by executing

.. code-block:: bash

   $ poetry install


in the repository directory.

Example
-------

The API is very similar to using sklearn. More details can be found under the corresponding documentation of SIMBSIG's modules.

.. code-block:: bash

   >>> X = [[0,1], [1,2], [2,3], [3,4]]
   >>> y = [0, 0, 1, 1]
   >>> from simbsig.neighbors import KNeighborsClassifier
   >>> knn_classifier = KNeighborsClassifier(n_neighbors=3)
   >>> knn_classifier.fit(X, y)
   KNeighborsClassifier(...)
   >>> print(knn_classifier.predict([[0.9, 1.9]]))
   [0]
   >>> print(knn_classifier.predict_proba([[0.9]]))
   [[0.666... 0.333...]]


