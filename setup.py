from setuptools import setup

setup(
    name='model_mining',
    version='0.1.0',
    packages=['', 'test'],
    package_dir={'': 'model_mining', 'test': "model_mining.tests"},
    url='',
    license='',
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='This package enables redescription model mining within the pysubgroup package',
    install_requires=[
              'pandas','scipy','numpy','matplotlib', 'numba', 'seaborn'
          ],
    python_requires='>=3.5'
)
