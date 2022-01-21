"""Module of Exceptions and warnings specific to Sarkas."""

# __all__ = [
#     "SarkasError",
#     "ParticlesError",
#     "SarkasWarning",
#     "PhysicsWarning",
#     "AlgorithmWarning",
# ]

# ------------------------------------------------------------------------------
#   Exceptions
# ------------------------------------------------------------------------------


class SarkasError(Exception):
    """
    Base class of Sarkas custom errors.
    All custom exceptions raised by Sarkas should inherit from this
    class and be defined in this module.
    """


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


# ^^^^^^^^^^^^ Base Exceptions should be defined above this comment ^^^^^^^^^^^^


class AlgorithmError(SarkasError):
    """The base error for errors related to the chosen algorithm."""


class ParticlesError(SarkasError):
    """The base error for errors related to the Particles class."""


# ------------------------------------------------------------------------------
#   Warnings
# ------------------------------------------------------------------------------


class SarkasWarning(Warning):
    """
    Base class of Sarkas custom warnings.
    All Sarkas custom warnings should inherit from this class and be
    defined in this module.
    Warnings should be issued using `warnings.warn`, which will not break
    execution if unhandled.
    """

    pass


class PhysicsWarning(Warning):
    """The base warning for warnings related to non-physical situations."""


# ^^^^^^^^^^^^^ Base Warnings should be defined above this comment ^^^^^^^^^^^^^


class AlgorithmWarning(SarkasWarning):
    """The base warning for warnings related to the used algorithm."""
