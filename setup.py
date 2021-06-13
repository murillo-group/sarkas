import glob
import os
import sys
from configparser import ConfigParser
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


# The following are needed to copy the MSU plot styles in Matplotlib folder
# From https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        import matplotlib as mpl
        import shutil
        # ~ # ref  ->  matplotlib/style/core
        BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
        STYLE_PATH = os.path.join(os.getcwd(), os.path.join('sarkas', 'mplstyles'))
        STYLE_EXTENSION = 'mplstyle'
        style_files = glob.glob(os.path.join(STYLE_PATH, "*.%s" % (STYLE_EXTENSION)))

        # Copy the plotting style in the matplotlib directory
        for _path_file in style_files:
            _, fname = os.path.split(_path_file)
            dest = os.path.join(BASE_LIBRARY_PATH, fname)
            shutil.copy(_path_file, dest)
            print("%s style installed" % (fname))


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        import matplotlib as mpl
        import shutil
        # ~ # ref  ->  matplotlib/style/core
        BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
        STYLE_PATH = os.path.join(os.getcwd(), os.path.join('sarkas', 'mplstyles'))
        STYLE_EXTENSION = 'mplstyle'
        style_files = glob.glob(os.path.join(STYLE_PATH, "*.%s" % (STYLE_EXTENSION)))

        # Copy the plotting style in the matplotlib directory
        for _path_file in style_files:
            _, fname = os.path.split(_path_file)
            dest = os.path.join(BASE_LIBRARY_PATH, fname)
            shutil.copy(_path_file, dest)
            print("%s style installed" % (fname))


# Package Requirements
BASE_DEPENDENCIES = [
    'numpy',
    'scipy',
    'pandas',
    'numba>=0.50',
    'pyfftw',
    'fdint',
    'pyyaml',
    'tqdm',
    'pyfiglet==0.8.post1',
    'pickle5',
    'jupyter',
    'jupyterlab',
    'notebook',
    'matplotlib',
    'seaborn',
]

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name')
DESCRIPTION = metadata.get('description')
DESCRIPTION_FILE = metadata.get('description-file')
PACKAGEDIR = metadata.get('package_dir')
VERSION = metadata.get('version')
AUTHOR = metadata.get('author')
AUTHOR_EMAIL = metadata.get('author_email')
LICENSE = metadata.get('license')
URL = metadata.get('url')
__minimum_python_version__ = metadata.get("minimum_python_version")

# Enforce Python version check - this is the same check as in __init__.py but
# this one has to happen before importing ah_bootstrap.
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    sys.stderr.write("ERROR: packagename requires Python {} or later\n".format(__minimum_python_version__))
    sys.exit(1)

# Read the README file into a string
with open(DESCRIPTION_FILE, "r") as fh:
    long_description = fh.read()

# Treat everything in scripts as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))]

setuptools.setup(
    name=PACKAGENAME,  # Replace with your own username
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    scripts=scripts,
    packages=setuptools.find_packages(),
    install_requires=BASE_DEPENDENCIES,
    # dependency_links = ['https://pypi.org/'],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    # Call the classes above and run the post installation scripts
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
