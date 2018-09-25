#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='zimmer',
      version='0.0.1',
      description='Modeling C elegans data from Manuel Zimmer',
      author='Scott Linderman',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['zimmer'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
