from setuptools import setup, find_packages


PACKAGENAME = "galsampler"
VERSION = "0.0.dev"


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
    url="https://github.com/LSSTDESC/galsampler"
)
