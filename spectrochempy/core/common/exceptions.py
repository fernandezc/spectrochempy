# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
SpectroChemPy specific exceptions
"""
import warnings
import sys, logging
from pathlib import Path

from contextlib import contextmanager

import pint
import pytz


# ======================================================================================
# Warning subclasses
# ======================================================================================
class SpectroChemPyWarning(UserWarning):
    """
    The base warning class for SpectroChemPy warnings.
    """


class UnitWarning(SpectroChemPyWarning):
    """
    Warning raised when an issues arise regarding units
    """


class LabelWarning(SpectroChemPyWarning):
    """
    Warning raised when an issues arise regarding labels
    """


class ValueWarning(SpectroChemPyWarning):
    """
    Warning raised when an issues arise arguments or attributes
    """


# ======================================================================================
# Exception Subclasses
# ======================================================================================
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

    def __init__(self, dtype, message):
        message = f"The assigned value has type {dtype} but {message}"
        super().__init__(message)


class InvalidNameError(SpectroChemPyException):
    """
    Exception when a object name is not valid
    """


class ShapeError(SpectroChemPyException):
    """
    Exception raised when an array cannot be set due to a wrong shape.
    """

    def __init__(self, shape, message):
        message = f"Assigned value has shape {shape} but {message}"
        super().__init__(message)


class MissingDataError(SpectroChemPyException):
    """
    Exception raised when no data is present in an object.
    """


class MissingCoordinatesError(SpectroChemPyException):
    """
    Exception raised when no coordinates in present in an object.
    """


class LabelsError(SpectroChemPyException):
    """
    Exception raised when an array cannot be labeled.

    For instance, if the array is multidimensional.
    """


class NotHyperComplexArrayError(SpectroChemPyException):
    """Returned when a hypercomplex related method is applied to a not hypercomplex
    array"""


class UnknownTimeZoneError(pytz.UnknownTimeZoneError):
    """
    Exception raised when Timezone code is not recognized.
    """


class UnitsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when units are not compatible,
    preventing some mathematical operations.
    """


class InvalidUnitsError(SpectroChemPyException):
    """
    Exception raised when units is not valid.
    """


class DimensionalityError(pint.DimensionalityError):
    """
    Exception raised when units have a dimensionality problem.
    """


class CoordinateMismatchError(SpectroChemPyException):
    """
    Exception raised when object coordinates differ.
    """

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Coordinates [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class DimensionsCompatibilityError(SpectroChemPyException):
    """
    Exception raised when dimensions are not compatible
    for concatenation for instance.
    """


class IncompatibleShapeError(SpectroChemPyException):
    """
    Exception raised when shapes of the elements are incompatibles for math operations.
    """

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Shapes of [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class InvalidDimensionNameError(SpectroChemPyException):
    """
    Exception raised when dimension name are invalid.
    """

    from spectrochempy.core.common.constants import DEFAULT_DIM_NAME

    def __init__(self, name, available_names=DEFAULT_DIM_NAME):
        self.message = (
            f"dim name must be one of {tuple(available_names)} "
            f"with an optional subdir indication (e.g., 'x_2'. dim=`"
            f"{name}` was given!"
        )
        super().__init__(self.message)


class InvalidCoordinatesSizeError(SpectroChemPyException):
    """
    Exception raised when size of coordinates does not match what is expected.
    """


class InvalidCoordinatesTypeError(SpectroChemPyException):
    """
    Exception raised when coordinates type is invalid.
    """


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
            f"It is expected to be one of {tuple(available_protocols)}"
        )

        super().__init__(self.message)


class WrongFileFormatError(SpectroChemPyException):
    """ """


# noinspection PyDeprecation
def deprecated(kind="method", replace="", extra_msg=""):
    """
    Deprecation decorator.

    Parameters
    ----------
    kind : str
        By default, it is method.
    replace : str
        Name of the method that replace the deprecated one.
    extra_msg : str
        Additional message.
    """
    from spectrochempy.core import warning_

    def deprecation_decorator(func):
        def wrapper(*args, **kwargs):
            name = func.__qualname__
            if name.endswith("__init__"):
                name = name.split(".", maxsplit=1)[0]
            sreplace = f" Use `{replace}` instead" if replace else ""
            warning_(
                f" `{name}` {kind} is now deprecated and could be completely "
                f"removed in version 0.5.*. {sreplace} {extra_msg}",
                category=DeprecationWarning,
            )
            return func(*args, **kwargs)

        return wrapper

    return deprecation_decorator


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
    >>> from spectrochempy.core.common.exceptions import ignored
    >>>
    >>> with ignored(OSError):
    ...     os.remove('file-that-does-not-exist')
    """

    try:
        yield
    except exc:
        pass


def _get_trace_info(*args):
    import traceback

    typ, val, tb = args
    info = traceback.extract_tb(tb)[-1]
    return (
        f"{info.name}[{Path(info.filename).name}:{info.lineno}] - {typ.__name__}"
        f" : {val}"
    )


def handle_exception(*args):
    """
    Custom handling of the uncaught exceptions

    Parameters
    ----------
    *args
        Arguments received from the caught exception.
    """

    from spectrochempy.core import app

    stg = _get_trace_info(*args)
    app.logs.handlers[0].setFormatter(logging.Formatter("%(message)s"))
    app.logs.handlers[1].setFormatter(logging.Formatter(f"[%(asctime)s - %(message)s"))
    app.logs.error(stg)
    sys.exit(1)


def send_warnings_to_log(*args, **kwargs):
    import inspect
    from spectrochempy.core import warning_, _format_args, app, _get_class_function

    if len(args) > 1:
        kwargs["category"] = args[1]  # priority to arg
    category = kwargs.pop("category", SpectroChemPyWarning)
    stack = inspect.stack()
    stg = _format_args(f"{category.__name__}: ", str(args[0]), stacklevel=-3)
    app.logs.warning(stg)
