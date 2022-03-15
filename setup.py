import re
import sys
from setuptools import find_packages, setup

with open("src/tissue_purifier/__init__.py") as f:
    for line in f:
        match = re.match('^__version__ = "(.*)"$', line)
        if match:
            __version__ = match.group(1)
            break

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="tissue_purifier",
    version=__version__,
    description="A Python library for the analysis of biological tissue and cellular micron-environments based on self supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url="https://github.com/broadinstitute/tissue_purifier",
    author="Luca Dalessio",
    author_email="dalessioluca@gmail.com",
    install_requires=[
        "torch>=1.10",
    ],
    extras_require={
        "test": [
            "isort>=5.9",
            "flake8",
            "pytest>=5.0",
        ],
    },
    python_requires=">=3.9",
    keywords="self supervised learning, tissue analysis",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.9",
    ],
)
