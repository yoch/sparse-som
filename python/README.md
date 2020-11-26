# Cython

## Compilation
python setup.py build_ext --inplace

## Cleanup
python setup.py clean

## Build package
python3 setup.py sdist bdist_wheel


# Build doc
sphinx-build -b html doc/source soc/build


# Running tests
python -m pytest tests.py
