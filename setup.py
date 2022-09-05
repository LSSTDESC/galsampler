import os
from setuptools import setup, find_packages


__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "galsampler", "_version.py"
)
with open(pth, "r") as fp:
    exec(fp.read())


setup(
    name="galsampler",
    version=__version__,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Tools for generating synthetic cosmological data",
    long_description="Tools for generating synthetic cosmological data",
    install_requires=["numpy", "numba", "scipy"],
    packages=find_packages(),
    url="https://github.com/LSSTDESC/galsampler",
)
