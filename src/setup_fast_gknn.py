from distutils.core import Extension
from Cython.Build import cythonize
import numpy
from numpy.distutils.core import setup

ext = Extension(name="fast_gknn",
                sources=["fast_gknn.pyx"],
                include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize(ext))
