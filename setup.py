from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


PACKAGENAME = "galsampler"
VERSION = "0.0.dev"


ext0 = Extension(
        name="galsampler.cython_kernels.galaxy_selection_kernel",
        sources=["galsampler/cython_kernels/galaxy_selection_kernel.pyx"],
        include_dirs=['numpy'])

ext_modules = [ext0]
cmdclass = dict(build_ext=build_ext)

setup(
    name=PACKAGENAME,
    version=VERSION,
    setup_requires=["pytest-runner"],
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Tools for generating synthetic cosmological data",
    long_description="Tools for generating synthetic cosmological data",
    install_requires=["numpy", "cython"],
    packages=find_packages(),
    url="https://github.com/LSSTDESC/galsampler",
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
