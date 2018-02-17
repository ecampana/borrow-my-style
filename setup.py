#!/usr/bin/env python

from codecs import open
from os import path
from distutils.core import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='code', # name pip uses to uninstall, update, etc.. must be the same as package name
      version='1.0',
      description='Data Science Toolkit',
      long_description=long_description,
      author='Emmanuel Contreras-Campana',
      author_email='emmanuelc82@gmail.com',
      url='https://github.com/ecampana/borrow-my-style',
      packages=['code'] # directories in borrow-my-style to install
)
