# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import warnings
import pytz
import pint

from contextlib import contextmanager

from spectrochempy.utils.misc import DEFAULT_DIM_NAME

__all__ = [
    "SpectroChemPyWarning",
    "SpectroChemPyException",
    "CastingError",
    "NoDataError",
    "UnitsCompatibilityError",
    "IncompatibleShapeError",
    "InvalidDimensionNameError",
    "InvalidCoordinatesTypeError",
    "InvalidCoordinatesSizeError",
    "DimensionsCompatibilityError",
    "DimensionalityError",
    "CoordinateMismatchError",
    "ProtocolError",
    "UnknownTimeZoneError",
    "deprecated",
    "ignored",
]


# ==============================================================================
# Exception and Warning Subclass
# ==============================================================================

# ------------------------------------------------------------------------------
class SpectroChemPyWarning(Warning):
    """
    The base warning class for SpectroChemPy warnings.
    """


# ------------------------------------------------------------------------------
class SpectroChemPyException(Exception):
    """
    The base exception class for SpectroChemPy.
    """

    def __init__(self, message):

        self.message = message

        super().__init__(message)


class CastingError(SpectroChemPyException):
    """
    Exception raised when an array cannot be cast to the required data type
    """


class NoDataError(SpectroChemPyException):
    """
    Exception raised when no data is present in ah object.
    """


class NotHyperComplexArrayError(SpectroChemPyException):
    """Returned when a hypercomplex related method is applied to a not hypercomplex
    array"""


class UnknownTimeZoneError(pytz.UnknownTimeZoneError):
    """
    Exception raised when Timezone code is not recognized.
    """


# ------------------------------------------------------------------------------
class UnitsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when units are not compatible,
    preventing some mathematical operations.
    """


class IncompatibleShapeError(SpectroChemPyException):
    """
    Exception raised when shapes of the elements are incompatibles for math operations.
    """

    def __init__(self, obj1, obj2, extra_msg=""):

        self.message = f"Shapes of [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


# ------------------------------------------------------------------------------
class InvalidDimensionNameError(SpectroChemPyException):
    """
    Exception raised when dimension name are invalid.
    """

    def __init__(self, name, available_names=DEFAULT_DIM_NAME):

        self.message = (
            f"dim name must be one of {tuple(available_names)} "
            f"with an optional subdir indication (e.g., 'x_2'. dim=`{name}` was given!"
        )
        super().__init__(self.message)


class InvalidCoordinatesTypeError(SpectroChemPyException):
    """
    Exception raised when coordinates type is invalid.
    """


class InvalidCoordinatesSizeError(SpectroChemPyException):
    """
    Exception raised when size of coordinates does not match what is expected.
    """


class DimensionalityError(pint.DimensionalityError):
    """
    Exception raised when units have a dimensionality problem.
    """


# ------------------------------------------------------------------------------
class DimensionsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when dimensions are not compatible
    for concatenation for instance.
    """


# ------------------------------------------------------------------------------
class CoordinateMismatchError(SpectroChemPyException):
    """
    Exception raised when object coordinates differ.
    """

    def __init__(self, obj1, obj2, extra_msg=""):

        self.message = f"Coordinates [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class ProtocolError(SpectroChemPyException):
    """
    This exception is issued when a wrong protocol is secified to the
    spectrochempy importer.

    Parameters
    ----------
    protocol : str
        The protocol string that was at the origin of the exception.
    available_protocols : list of str
        The available (implemented) protocols.
    """

    def __init__(self, protocol, available_protocols):

        self.message = (
            f"IO - The `{protocol}` protocol is unknown or not yet implemented.\n"
        )
        f"It is expected to be one of {tuple(available_protocols)}"

        super().__init__(self.message)


# ------------------------------------------------------------------------------
def deprecated(type="method", replace="", extra_msg=""):
    """
    Deprecation decorator.

    Parameters
    ----------
    type : str
        By default, it is method.
    replace : str
        Name of the method taht replace the deprecated one.
    extra_msg : str
        Additional message.
    """

    def deprecation_decorator(func):
        def wrapper(*args, **kwargs):

            warnings.warn(
                f" `{func.__name__}` {type} is now deprecated and could be "
                f"completely "
                f"removed in version 0.5.*." + f" Use `{replace}`."
                if replace
                else "" + f" {extra_msg}."
                if extra_msg
                else "",
                DeprecationWarning,
            )

            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


# ..............................................................................
try:
    from contextlib import ignored
except ImportError:

    @contextmanager
    def ignored(*exc):
        """
        A context manager for ignoring exceptions.

        This is equivalent to::

            try :
                <body>
            except exc :
                pass

        Parameters
        ----------
        *exc : Exception
            One or several exceptions to ignore.

        Examples
        --------

        >>> import os
        >>> from spectrochempy.utils import ignored
        >>>
        >>> with ignored(OSError):
        ...     os.remove('file-that-does-not-exist')
        """

        try:
            yield
        except exc:
            pass


# ==============================================================================
# EOF
