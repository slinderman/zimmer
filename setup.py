#!/usr/bin/env python

from distutils.core import setup

setup(name='zimmer',
      version='0.0.1',
      description='Modeling C elegans data from Manuel Zimmer',
      author='Scott Linderman',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['zimmer'],
      )
