from setuptools import setup, Extension
from Cython.Build import build_ext # , cythonize
import os
import numpy as np


if os.name == 'posix':
    extra_compile_args = ['-std=c++11', '-fopenmp']
    extra_link_args = ['-fopenmp']
elif os.name == 'nt':
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    raise RuntimeError('unkown platform')

som_ext = [
            Extension('sparse_som.som',
                sources=['sparse_som/som.pyx', '../src/bsom.cpp', '../src/som.cpp'],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                include_dirs = [np.get_include()],
                language='c++')
]


setup(
  name = 'sparse_som',
  packages = ['sparse_som'],
  version = '0.4',
  description = 'Self-Organizing Maps for sparse inputs in python',
  author = 'J. Melka',
  url = 'https://gitlab.com/yoch/sparse-som',
  ext_modules = som_ext, # cythonize(som_ext) don't work
  license = 'GPL3',
  cmdclass = {'build_ext': build_ext},
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft',
    'Programming Language :: C++',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Visualization'
  ],
  install_requires=['numpy', 'scipy']
)
