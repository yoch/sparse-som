import os
import sys
from setuptools import setup, Extension


if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False


if os.name == 'posix':
    extra_compile_args = ['-std=c++11', '-fopenmp']
    extra_link_args = ['-std=c++11', '-fopenmp']
elif os.name == 'nt':
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    raise RuntimeError('unkown platform')


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [Extension('sparse_som.som',
                sources=[
                    'sparse_som/som'+ext,
                    'sparse_som/lib/som.cpp',
                    'sparse_som/lib/bsom.cpp',
                    'sparse_som/lib/data.cpp',
                ],
                extra_objects=[
                    'sparse_som/lib/som.h',
                    'sparse_som/lib/data.h',
                ],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                include_dirs=['sparse_som/'],
                language='c++')]


if USE_CYTHON:
    from Cython.Build import build_ext  # cythonize don't works
else:
    from distutils.command.build_ext import build_ext


class NumpyBuildExt(build_ext):
    "build_ext command for use when numpy headers are needed."
    def run(self):
        import numpy

        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


setup(
  name = 'sparse_som',
  packages = ['sparse_som'],
  version = '0.5.5',
  description = 'Self-Organizing Maps for sparse inputs in python',
  author = 'J. Melka',
  url = 'https://github.com/yoch/sparse-som',
  ext_modules = extensions,
  cmdclass = {'build_ext': NumpyBuildExt},
  #package_data = {'sparse_som': ['*.pyx']},
  license = 'GPL3',
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
