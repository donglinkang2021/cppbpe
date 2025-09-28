from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "bpe_core",
        ["bpe_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="bpe_core",
    version="0.0.1",
    ext_modules=ext_modules,
    packages=[],
)

# python setup.py build_ext --inplace
