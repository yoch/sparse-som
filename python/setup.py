import io
import os
from setuptools import setup, Extension


DESCRIPTION = 'Self-Organizing Maps for sparse inputs in python'

# README.md sits next to this file and ships in the sdist (see MANIFEST.in);
# fall back to the short description if it is missing.
try:
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
except IOError:
    LONG_DESCRIPTION = DESCRIPTION


try:
    from Cython.Build import build_ext  # cythonize don't works
    USE_CYTHON = True
except:
    from distutils.command.build_ext import build_ext
    USE_CYTHON = False


if os.name == 'posix':
    extra_compile_args = ['-std=c++11', '-fopenmp']
    extra_link_args = ['-std=c++11', '-fopenmp']
elif os.name == 'nt':
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    raise RuntimeError('unsupported platform')


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [
    Extension(
        'sparse_som.som',
        sources=[
            'sparse_som/som'+ext,
            'sparse_som/lib/som.cpp',
            'sparse_som/lib/bsom.cpp',
            'sparse_som/lib/data.cpp',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=['sparse_som/'],
        language='c++',
        #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


class NumpyBuildExt(build_ext):
    "build_ext command for use when numpy headers are needed."
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


setup(
  name = 'sparse_som',
  packages = ['sparse_som'],
  version = '0.6.1',
  description = DESCRIPTION,
  long_description = LONG_DESCRIPTION,
  long_description_content_type = 'text/markdown',
  author = 'J. Melka',
  url = 'https://github.com/yoch/sparse-som',
  project_urls = {
    'Documentation': 'https://sparse-som.readthedocs.io',
    'Source': 'https://github.com/yoch/sparse-som',
    'Issues': 'https://github.com/yoch/sparse-som/issues',
    'Paper': 'https://doi.org/10.5220/0006499500540063',
  },
  keywords = [
    'self-organizing-map', 'som', 'kohonen', 'sparse', 'libsvm',
    'openmp', 'clustering', 'unsupervised-learning',
  ],
  ext_modules = extensions,
  cmdclass = {'build_ext': NumpyBuildExt},
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
