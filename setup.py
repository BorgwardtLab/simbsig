# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['simbsig',
 'simbsig.base',
 'simbsig.cluster',
 'simbsig.decomposition',
 'simbsig.neighbors',
 'simbsig.utils']

package_data = \
{'': ['*']}

install_requires = \
['h5py>=3.7.0,<4.0.0',
 'numpy>=1.23.0,<2.0.0',
 'sklearn>=0.0,<0.1',
 'torch>=1.12.0,<2.0.0',
 'tqdm>=4.64.0,<5.0.0']

setup_kwargs = {
    'name': 'simbsig',
    'version': '0.1.0',
    'description': 'A python package for out-of-core similarity search and dimensionality reduction',
    'long_description': None,
    'author': 'Eljas Roellin',
    'author_email': 'roelline@student.ethz.ch',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
