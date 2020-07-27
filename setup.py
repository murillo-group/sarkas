import glob
import os
import sys
from configparser import ConfigParser
import setuptools

import matplotlib as mpl
import shutil

#~ # ref  ->  matplotlib/style/core
BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
STYLE_PATH = os.path.join(os.getcwd(),os.path.join('sarkas','mplstyles'))
STYLE_EXTENSION = 'mplstyle'
style_files = glob.glob(os.path.join(STYLE_PATH,"*.%s"%(STYLE_EXTENSION)))

for _path_file in style_files:
    _, fname = os.path.split(_path_file)
    dest = os.path.join(BASE_LIBRARY_PATH, fname)
    shutil.copy(_path_file, dest)
    print("%s style installed"%(fname))

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'sarkas')
DESCRIPTION = metadata.get('description', 'Sarkas')
DESCRIPTION_FILE = metadata.get('description-file', 'README.md')
VERSION = metadata.get('version','')
AUTHOR = metadata.get('author', 'author')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'https://murillo-group.github.io/sarkas')
__minimum_python_version__ = metadata.get("minimum_python_version", "3.6")

# Enforce Python version check - this is the same check as in __init__.py but
# this one has to happen before importing ah_bootstrap.
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    sys.stderr.write("ERROR: packagename requires Python {} or later\n".format(__minimum_python_version__))
    sys.exit(1)


with open(DESCRIPTION_FILE, "r") as fh:
    long_description = fh.read()

# Treat everything in scripts except README.rst as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))]


setuptools.setup(
    name=PACKAGENAME + "-murillogroup", # Replace with your own username
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    scripts=scripts,
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'numba>=0.50.1',
        'fdint',
        'tqdm',
        'pyfiglet',
        'pyyaml'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
