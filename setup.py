from setuptools import setup, find_packages
from rugliderqc import __version__

setup(
    name='rugliderqc',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/rucool/rugliderqc',
    author='Laura Nazzaro, Lori Garzio',
    author_email='nazzaro@marine.rutgers.edu, lgarzio@marine.rutgers.edu',
    description='Python tools for quality control of real-time and delayed-mode glider data.'
)
