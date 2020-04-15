from setuptools import setup

setup(
    name='RDMM',
    version='0.1.0',
    packages=['RDMM'],
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='This package enables redescription model mining within the pysubgroup package',
    install_requires=[
              'pandas','scipy','numpy','matplotlib', 'numba', 'seaborn'
          ],
    python_requires='>=3.5'
)
