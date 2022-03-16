"""Welcome to Sarkas: a fast pure Python molecular dynamics software for plasmas physics."""

__all__ = [
    "Potential",
    "Integrator",
    "Thermostat",
    "Process",
    "Simulation",
    "PreProcess",
    "PostProcess",
    "Particles",
    "Parameters",
    "Species",
    "InputOutput",
    "correlationfunction",
    "fd_integral",
    "inverse_fd_half",
    "__version__",
]

__all__.sort()
# This __init__.py follows the one from PlasmaPy

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

if sys.version_info < (3, 7):
    raise Exception("Sarkas does not support Python < 3.7")

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
import pkg_resources

from .core import Parameters, Particles, Species
from .potentials.core import Potential
from .processes import PostProcess, PreProcess, Process, Simulation
from .time_evolution.integrators import Integrator
from .time_evolution.thermostats import Thermostat
from .utilities.io import InputOutput
from .utilities.maths import correlationfunction, fd_integral, inverse_fd_half

# define version
try:
    # this places a runtime dependency on setuptools
    #
    # note: if there's any distribution metadata in your source files, then this
    #       will find a version based on those files.  Keep distribution metadata
    #       out of your repository unless you've intentionally installed the package
    #       as editable (e.g. `pip install -e {sarkas_root_directory}`),
    #       but then __version__ will not be updated with each commit, it is
    #       frozen to the version at time of install.
    #
    #: Sarkas version string
    __version__ = pkg_resources.get_distribution("sarkas").version
except pkg_resources.DistributionNotFound:
    # package is not installed
    fallback_version = "unknown"
    try:
        # code most likely being used from source
        # if setuptools_scm is installed then generate a version
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__, fallback_version=fallback_version)
        del get_version
        warn_add = "setuptools_scm failed to detect the version"
    except ModuleNotFoundError:
        # setuptools_scm is not installed
        __version__ = fallback_version
        warn_add = "setuptools_scm is not installed"

    if __version__ == fallback_version:
        from warnings import warn

        warn(
            f"sarkas.__version__ not generated (set to 'unknown'), Sarkas is "
            f"not an installed package and {warn_add}.",
            RuntimeWarning,
        )

        del warn
    del fallback_version, warn_add

del pkg_resources, sys
