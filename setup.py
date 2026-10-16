
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "Navigation.pedestrian_traj",
        ["Navigation/pedestrian_traj.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "Navigation.decode_image",
        ["Navigation/decode_image.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "Navigation.bounding_box",
        ["Navigation/bounding_box.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="MonProjetCython",
    packages=["Navigation", "Outputs", "Core"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
