# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SpectroChemPy specific exceptions
"""
import inspect
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

import pint


# ======================================================================================
# Warning subclasses
# ======================================================================================
class KeyErrorWarning(UserWarning):
    """
    Warning raised when an issue arise regarding units
    """


class UnitErrorWarning(UserWarning):
    """
    Warning raised when an issue arise regarding units
    """


class LabelErrorWarning(UserWarning):
    """
    Warning raised when an issue arise regarding labels
    """


class ValueErrorWarning(UserWarning):
    """
    Warning raised when an issue arise arguments or attributes
    """


class NeedsUpdateWarning(UserWarning):
    """
    Warning raised when an issue arise arguments or attributes
    """


# ======================================================================================
# Exception Subclasses
# ======================================================================================
class SpectroChemPyError(Exception):
    """
    The base exception class for SpectroChemPy.
    """

    def __init__(self, message):
        self.message = message

        super().__init__(message)


# ======================================================================================
# Exceptions for configurable models
# ======================================================================================
class NotTransformedError(SpectroChemPyError):
    """
    Exception raised when a model has not yet been applied to a dataset,
    but one use one of its method.

    Parameters
    ----------
    attr : method, optional
        The method from which the error was issued. In general, it is determined
        automatically.
    """

    def __init__(self, attr=None, message=None):
        if message is not None:
            super().__init__(message)
            return
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}` ,  the method `apply` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message)


class NotFittedError(NotTransformedError):
    """
    Exception raised when an analysis estimator is not fitted
    but one use one of its method.

    Parameters
    ----------
    attr : method, optional
        The method from which the error was issued. In general, it is determined
        automatically.
    """

    def __init__(self, attr=None, message=None):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}` ,  the method `fit` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message=message)


class CastingError(SpectroChemPyError):
    """
    Exception raised when an array cannot be cast to the required data type
    """

    def __init__(self, dtype, message):
        message = f" Assigned value has type {dtype} but {message}"
        super().__init__(message)


class InvalidNameError(SpectroChemPyError):
    """
    Exception when a object name is not valid
    """


class ShapeError(SpectroChemPyError):
    """
    Exception raised when an array cannot be set due to a wrong shape.
    """

    def __init__(self, shape, message):
        message = f" Assigned value has shape {shape} but {message}"
        super().__init__(message)


class MissingDataError(SpectroChemPyError):
    """
    Exception raised when no data is present in an object.
    """


class NDDatasetAttributeError(SpectroChemPyError):
    """
    Exception raised when a dataset attribute was not found.
    """

    def __init__(self, attr):
        message = f" NDDataset attribute `{attr}` was not found."
        super().__init__(message)


class CoordinatesAttributeError(SpectroChemPyError):
    """
    Exception raised when a dataset attribute was not found.
    """

    def __init__(self, attr):
        message = f" Coord attribute `{attr}` was not found."
        super().__init__(message)


class MissingCoordinatesError(SpectroChemPyError):
    """
    Exception raised when no coordinates in present in an object.
    """


class LabelsError(SpectroChemPyError):
    """
    Exception raised when an array cannot be labeled.

    For instance, if the array is multidimensional.
    """


class NotHyperComplexArrayError(SpectroChemPyError):
    """Returned when a hypercomplex related method is applied to a not hypercomplex
    array"""


class UnitsCompatibilityError(SpectroChemPyError):
    """
    Exception raised when units are not compatible,
    preventing some mathematical operations.
    """


class InvalidUnitsError(SpectroChemPyError):
    """
    Exception raised when units is not valid.
    """


class InvalidReferenceError(SpectroChemPyError):
    """
    Exception raised when a reference to another coordinate is not valid
    """


class DimensionalityError(pint.DimensionalityError):
    """
    Exception raised when units have a dimensionality problem.
    """


class CoordinatesMismatchError(SpectroChemPyError):
    """
    Exception raised when object coordinates differ.
    """

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Coordinates [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class DimensionsCompatibilityError(SpectroChemPyError):
    """
    Exception raised when dimensions are not compatible
    for concatenation for instance.
    """


class IncompatibleShapeError(SpectroChemPyError):
    """
    Exception raised when shapes of the elements are incompatibles for math operations.
    """

    def __init__(self, obj1, obj2, extra_msg=""):
        self.message = f"Shapes of [{obj1}] and [{obj2}] mismatch. {extra_msg}"
        super().__init__(self.message)


class NonWritableCoordSetError(SpectroChemPyError):
    """
    Exception raised when the CoordSEt is readonly,
    but an attempt to write it has been done.
    """


class InvalidDimensionNameError(SpectroChemPyError):
    """
    Exception raised when dimension name are invalid.
    """

    from spectrochempy.utils.constants import DEFAULT_DIM_NAME

    def __init__(self, name, available_names=DEFAULT_DIM_NAME):
        self.message = (
            f"dim name must be one of {tuple(available_names)} "
            f"with an optional subdir indication (e.g., 'x_2') but dim=`"
            f"{name}` was given!"
        )
        super().__init__(self.message)


class InvalidCoordinatesSizeError(SpectroChemPyError):
    """
    Exception raised when size of coordinates does not match what is expected.
    """


class InvalidCoordinatesTypeError(SpectroChemPyError):
    """
    Exception raised when coordinates type is invalid.
    """


class InvalidCoordSetSizeError(SpectroChemPyError):
    """
    Exception raised when size of coordset does not match what is expected.
    """


class ProtocolError(SpectroChemPyError):
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


class WrongFileFormatError(SpectroChemPyError):
    """ """


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
    >>> from spectrochempy.utils.exceptions import ignored
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

    from spectrochempy.application import app

    stg = _get_trace_info(*args)
    app.logs.handlers[0].setFormatter(logging.Formatter("%(message)s"))
    app.logs.handlers[1].setFormatter(logging.Formatter("[%(asctime)s - %(message)s]"))
    app.logs.error(stg)
    sys.exit(1)


def send_warnings_to_log(*args, **kwargs):
    from spectrochempy.application import _format_args, app

    if len(args) > 1:
        kwargs["category"] = args[1]  # priority to arg
    category = kwargs.pop("category", UserWarning)
    # stack = inspect.stack()
    stg = _format_args(f"{category.__name__}: ", str(args[0]), stacklevel=-3)
    app.logs.warning(stg)
