from setuptools import setup
  
setup(
    name='evopolicy',
    version='0.1',
    description='Evolution Strategies as a Scalabale Alternative to RL',
    author='Zach Smith',
    author_email='ezke.smith@gmail.com',
    packages=['evopolicy'],
    install_requires=[
        'gym',
        'matplotlib',
        'numpy',
        'tqdm',
    ],
)
