from setuptools import setup, find_packages

setup(
    name='tissue_purifier',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    license='MIT',
    author='Luca Dalessio',
    author_email='ldalessi@broadinstitute.org',
    description='Tissue analysis in Python',
    extras_require=dict(tests=['pytest']),
)
