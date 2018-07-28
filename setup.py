import os
import subprocess

import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

"""
Run setup with the following command:
```
python setupGpuWrapper.py build_ext --inplace
```
"""

# Determine current directory of this setup file to find our module
CUR_DIR = os.path.dirname(__file__)

# Use pkg-config to determine library locations and include locations
opencv_libs_str = subprocess.check_output(
                      "pkg-config --libs opencv".split()).decode()
opencv_incs_str = subprocess.check_output(
                      "pkg-config --cflags opencv".split()).decode()

# Parse into usable format for Extension call
opencv_libs = [str(lib) for lib in opencv_libs_str.strip().split()]
# remove leading "-I"
#opencv_incs = [str(inc[2:]) for inc in opencv_incs_str.strip().split()]

extensions = [
    Extension('cv2cuda.gpuwrapper',
              sources=[os.path.join(CUR_DIR, 'cv2cuda/gpuwrapper.pyx')],
              language='c++',
              #include_dirs=[np.get_include()] + opencv_incs,
              include_dirs=[np.get_include()],
              extra_link_args=opencv_libs)
]

setup(
    cmdclass={'build_ext': build_ext},
    name="cv2cuda",
    ext_modules=cythonize(extensions)
)
