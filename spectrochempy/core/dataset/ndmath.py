# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
This module implements the NDMath class.
"""
# TODO: test binary ufunc and put them in docs

__all__ = ["NDMath", "NDManipulation"]

__dataset_methods__ = []

import copy as cpy
import functools
import inspect
import operator
import re
import sys
from warnings import catch_warnings

import numpy as np
from numpy.random import rand

from quaternion import as_float_array

from spectrochempy.core import error_, warning_, debug_
from spectrochempy.core.common.compare import is_sequence
from spectrochempy.core.common.complex import (
    as_quat_array,
    as_quaternion,
    quat_as_complex_array,
)
from spectrochempy.core.common.constants import NOMASK, TYPE_COMPLEX
from spectrochempy.core.common.docstrings import DocstringProcessor
from spectrochempy.core.common.exceptions import (
    CoordinateMismatchError,
    IncompatibleShapeError,
)
from spectrochempy.core.dataset.ndarray import NDArray
from spectrochempy.core.units.units import DimensionalityError, Quantity, ur
from spectrochempy.utils.orderedset import OrderedSet
from spectrochempy.utils.testing import assert_coord_almost_equal

# docstring substitution (docrep)
# --------------------------------------------------------------------------------------
_docstring = DocstringProcessor()


# ======================================================================================
# utilities
# ======================================================================================


def _transpose_hypercomplex(data):
    # when transposing hypercomplex array
    # we interchange the imaginary component
    w, x, y, z = as_float_array(data).T
    q = as_quat_array(
        list(zip(w.T.flatten(), y.T.flatten(), x.T.flatten(), z.T.flatten()))
    )
    data = q.reshape(data.shape)
    return data


def _reduce_method(method):
    # Decorator
    # ---------
    # set the flag reduce to true for the _from numpy decorator.
    # Must be placed above this decorator.
    # e.g.,
    #    @_reduce_method
    #    @_from_numpy_method
    #    def somefunction(....):
    #
    method.reduce = True
    return method


class _from_numpy_method:
    # Decorator
    # ---------
    # This decorator assumes that the signature starts always by : (cls, ...)
    # the second positional only argument can be `dataset` -
    # in this case this mean that the function apply on a dataset

    reduce = False

    def __init__(self, method):
        self.method = method

    def __get__(self, instance, cls):
        @functools.wraps(self.method)
        def func(*args, **kwargs):

            # Delayed import to avoid circular reference
            from spectrochempy.core.dataset.coord import Coord
            from spectrochempy.core.dataset.nddataset import NDDataset

            method = self.method.__name__
            pars = inspect.signature(self.method).parameters

            args = list(args)
            klass = NDDataset  # by default
            new = klass()

            if "dataset" not in pars:
                if instance is not None:
                    klass = type(instance)
                elif issubclass(cls, (NDDataset, Coord)):
                    klass = cls
                else:
                    # Probably a call from the API !
                    # We return a NDDataset class constructor
                    klass = NDDataset
            else:
                # determine the input object
                if instance is not None:
                    # the call is made as an attributes of the instance
                    # instance.method(...)
                    new = instance.copy()
                    args.insert(0, new)
                else:
                    dataset = cpy.copy(args[0])
                    try:
                        # call as a classmethod
                        # class.method(dataset, ...)
                        new = cls(dataset)
                    except TypeError:
                        if issubclass(cls, NDMath):
                            # Probably a call from the API !
                            # scp.method(dataset, ...)
                            new = dataset
                            if not isinstance(new, NDArray):
                                # we have only an array-like dataset
                                # make a NDDataset object from it
                                new = NDDataset(dataset)

            argpos = []

            for par in pars.values():
                if par.name in [
                    "self",
                    "cls",
                    "kwargs",
                ]:  # or (par.name == 'dataset' and instance is not None):
                    continue
                argpos.append(
                    args.pop(0) if args else kwargs.pop(par.name, par.default)
                )

            # in principle args should be void at this point
            assert not args

            # -----------------------------
            # Creation from scratch methods
            # -----------------------------

            if "dataset" not in pars:
                # separate object keyword from other specific to the function
                kw = {}
                keys = dir(klass())
                keys.extend(["description", "source", "title"])
                # These one have an equivalent but for historical reason are often passed
                # (we will have to set them as deprecated)
                for k in list(kwargs.keys())[:]:
                    if k not in keys:
                        kw[k] = kwargs[k]
                        del kwargs[k]
                kwargs["kw"] = kw
                # now call the np function and make the object
                new = self.method(klass, *argpos, **kwargs)
                if new._implements("NDDataset"):
                    new.history = f"Created using method : {method}"  # (args:{argpos}, kwargs:{kwargs})'
                return new

            # -----------------------------
            # Create from an existing array
            # ------------------------------

            # Replace some of the attribute according to the kwargs
            for k, v in list(kwargs.items())[:]:
                if k != "units":
                    setattr(new, k, v)
                    del kwargs[k]
                else:
                    new.ito(v, force=True)
                    del kwargs[k]

            # Be sure that the dataset passed to the numpy function are a numpy (masked) array
            if isinstance(argpos[0], (NDDataset, Coord)):
                # argpos[0] = argpos[0].real.masked_data
                argpos[0] = argpos[0].masked_data

            # case of creation like method

            if not self.reduce:  # _like' in method:

                new = self.method(new, *argpos)
                if new._implements("NDDataset"):
                    new.history = f"Created using method : {method}"  # (args:{argpos}, kwargs:{kw})'
                return new

            # reduce methods

            # apply the numpy operator on the masked data
            new = self.method(new, *argpos)

            if not isinstance(new, (NDDataset, Coord)):
                # if a numpy array or a scalar is returned after reduction
                return new

            # # particular case of functions that returns Dataset with no coordinates
            # if dim is None and method in ['sum', 'trapz', 'prod', 'mean', 'var', 'std']:
            #     # delete all coordinates
            #     new._coordset = None

            new.history = f"Dataset resulting from application of `{method}` method"
            return new

        return func


def _reduce_dims(cls, dim, keepdims=False):

    dims = cls.dims
    if hasattr(cls, "coordset"):
        coordset = cls.coordset
        if dim is not None:
            if coordset is not None:
                idx = coordset.names.index(dim)
                if not keepdims:
                    del coordset.coords[idx]
                    dims.remove(dim)
                else:
                    coordset.coords[idx].data = [
                        0,
                    ]
            else:
                if not keepdims:
                    dims.remove(dim)
        else:
            # dim being None we eventually remove the coordset
            cls.set_coordset(None)

    return dims


def _get_name(x):
    return str(x.name if hasattr(x, "name") else x)


def _extract_ufuncs(strg):

    ufuncs = {}
    regex = r"^([a-z,0-9,_]*)\((x.*)\[.*]\)\W*(.*\.)$"
    matches = re.finditer(regex, strg, re.MULTILINE)

    for match in matches:
        func = match.group(1)
        args = match.group(2)
        desc = match.group(3)

        ufuncs[func] = f"({args.strip()}) -> {desc.strip()}"

    return ufuncs


DIMENSIONLESS = ur("dimensionless").units
UNITLESS = None
TYPEPRIORITY = {"Coord": 2, "NDDataset": 3}

UNARY_STR = """
abs(x [, out, where, casting, order, …])         Calculate the absolute value element-wise (alias of absolute).
absolute(x [, out, where, casting, order, …])    Calculate the absolute value element-wise.
fabs(x [, out, where, casting, order, …])        Compute the absolute values element-wise.
rint(x [, out, where, casting, order, …])        Round elements of the array to the nearest integer.
floor(x [, out, where, casting, order, …])    Return the floor of the input, element-wise.
ceil(x [, out, where, casting, order, …])    Return the ceiling of the input, element-wise.
trunc(x [, out, where, casting, order, …])    Return the truncated value of the input, element-wise.
negative(x [, out, where, casting, order, …])    Numerical negative, element-wise.

around(x [, decimals, out])                Evenly round to the given number of decimals.
round(x [, decimals, out])                Round an array to the given number of decimals.
rint(x [, out, where, casting, order, …])  Round elements of the array to the nearest integer.
fix(x[, out])                              Round to nearest integer towards zero.

exp(x [, out, where, casting, order, …])     Calculate the exponential of all elements in the input array.
exp2(x [, out, where, casting, order, …])    Calculate 2**p for all p in the input array.
log(x [, out, where, casting, order, …])     Natural logarithm, element-wise.
log2(x [, out, where, casting, order, …])    Base-2 logarithm of x.
log10(x [, out, where, casting, order, …])    Return the base 10 logarithm of the input array, element-wise.
expm1(x [, out, where, casting, order, …])    Calculate exp(x - 1) for all elements in the array.
log1p(x [, out, where, casting, order, …])    Return the natural logarithm of one plus the input array, element-wise.

sqrt(x [, out, where, casting, order, …])      Return the non-negative square-root of an array, element-wise.
square(x [, out, where, casting, order, …])    Return the element-wise square of the input.
cbrt(x [, out, where, casting, order, …])      Return the cube-root of an array, element-wise.
reciprocal(x [, out, where, casting, …])       Return the reciprocal of the argument, element-wise.

sin(x [, out, where, casting, order, …])       Trigonometric sine, element-wise.
cos(x [, out, where, casting, order, …])       Cosine element-wise.
tan(x [, out, where, casting, order, …])       Compute tangent element-wise.
arcsin(x [, out, where, casting, order, …])    Inverse sine, element-wise.
arccos(x [, out, where, casting, order, …])    Trigonometric inverse cosine, element-wise.
arctan(x [, out, where, casting, order, …])    Trigonometric inverse tangent, element-wise.

sinh(x [, out, where, casting, order, …])       Hyperbolic sine, element-wise.
cosh(x [, out, where, casting, order, …])       Hyperbolic cosine, element-wise.
tanh(x [, out, where, casting, order, …])       Compute hyperbolic tangent element-wise.
arcsinh(x [, out, where, casting, order, …])    Inverse hyperbolic sine element-wise.
arccosh(x [, out, where, casting, order, …])    Inverse hyperbolic cosine, element-wise.
arctanh(x [, out, where, casting, order, …])    Inverse hyperbolic tangent element-wise.

degrees(x [, out, where, casting, order, …])     Convert angles from radians to degrees.
radians(x [, out, where, casting, order, …])     Convert angles from degrees to radians.
deg2rad(x [, out, where, casting, order, …])     Convert angles from degrees to radians.
rad2deg(x [, out, where, casting, order, …])     Convert angles from radians to degrees.

sign(x [, out, where, casting, order, …])       Returns an element-wise indication of the sign of a number.

isfinite(x [, out, where, casting, order, …])   Test element-wise for finiteness (not infinity or not Not a Number).
isinf(x [, out, where, casting, order, …])      Test element-wise for positive or negative infinity.
isnan(x [, out, where, casting, order, …])      Test element-wise for NaN and return result as a boolean array.

logical_not(x [, out, where, casting, …])       Compute the truth value of NOT x element-wise.

signbit(x, [, out, where, casting, order, …])   Returns element-wise True where signbit is set (less than zero).
"""


def _unary_ufuncs():

    return _extract_ufuncs(UNARY_STR)


BINARY_STR = """

multiply(x1, x2 [, out, where, casting, …])    Multiply arguments element-wise.
divide(x1, x2 [, out, where, casting, …])    Returns a true division of the inputs, element-wise.

maximum(x1, x2 [, out, where, casting, …])    Element-wise maximum of array elements.
minimum(x1, x2 [, out, where, casting, …])    Element-wise minimum of array elements.
fmax(x1, x2 [, out, where, casting, …])    Element-wise maximum of array elements.
fmin(x1, x2 [, out, where, casting, …])    Element-wise minimum of array elements.

add(x1, x2 [, out, where, casting, order, …])    Add arguments element-wise.
subtract(x1, x2 [, out, where, casting, …])    Subtract arguments, element-wise.

copysign(x1, x2 [, out, where, casting, …])    Change the sign of x1 to that of x2, element-wise.
"""


def _binary_ufuncs():

    return _extract_ufuncs(BINARY_STR)


COMP_STR = """
# Comparison functions

greater(x1, x2 [, out, where, casting, …])         Return the truth value of (x1 > x2) element-wise.
greater_equal(x1, x2 [, out, where, …])            Return the truth value of (x1 >= x2) element-wise.
less(x1, x2 [, out, where, casting, …])            Return the truth value of (x1 < x2) element-wise.
less_equal(x1, x2 [, out, where, casting, …])      Return the truth value of (x1 =< x2) element-wise.
not_equal(x1, x2 [, out, where, casting, …])       Return (x1 != x2) element-wise.
equal(x1, x2 [, out, where, casting, …])           Return (x1 == x2) element-wise.
"""


def _comp_ufuncs():

    return _extract_ufuncs(COMP_STR)


LOGICAL_BINARY_STR = """

logical_and(x1, x2 [, out, where, …])          Compute the truth value of x1 AND x2 element-wise.
logical_or(x1, x2 [, out, where, casting, …])  Compute the truth value of x1 OR x2 element-wise.
logical_xor(x1, x2 [, out, where, …])          Compute the truth value of x1 XOR x2, element-wise.
"""


def _logical_binary_ufuncs():

    return _extract_ufuncs(LOGICAL_BINARY_STR)


# Expected Operand Order
ORDER = {"Panel": 1, "NDDataset": 2, "Coord": 3, "LinearCoord": 4}


class NDManipulation(object):
    """
    This class provides manipulation routines for subclass of NDArray object.
    """

    def expand_dims(self, dim=None):
        """
        Expand the shape of an array.

        Insert a new axis that will appear at the `axis` position in the expanded array shape.

        Parameters
        ----------
        dim : int or str
            Position in the expanded axes where the new axis (or axes) is placed.

        Returns
        -------
        |NDDataset|
            View of `a` with the number of dimensions increased.

        See Also
        --------
        squeeze : The inverse operation, removing singleton dimensions.
        """
        # TODO

    def atleast_1d(self):
        """ """

    def atleast_2d(self):
        """ """

    @_docstring.dedent
    def squeeze(self, *dims, keepdims=(), inplace=False, **kwargs):
        """
        Remove single-dimensional entries from the shape of an array.

        To select only some dimension to squeeze, use the `dims` parameter.
        To keep some dimensions with size 1 untouched, use the keepdims parameters.

        Parameters
        ----------
        *dims : None, int, str, or tuple of ints or str, optional
            Selects a subset of the single-dimensional entries in the
            shape. If a dimension (dim) is selected with shape entry greater than
            one, an error is raised.
        keepdims : None, int, str, or tuple of ints or str, optional
            Selects a subset of the single-dimensional entries in the
            shape which remains preserved even if hey are of size 1.
            Used only if the `dims` are None. *(Added in version 0.4)*.
        %(inplace)s
        %(kwargs)s

        Returns
        -------
        %(out)s
        returned_index
            Only if return_index is True.

        Other Parameters
        ----------------
        dim or axis : None, int or str
            Equivalent of `dims` when only one dimension is concerned.
        return_index : bool, optional
            If True the previous index of the removed dimensions are returned.
            This mainly for internal use in SpectroChemPy, but probably not
            useful for the end-user.

        Raises
        ------
        ValueError
            If `dims` is not `None`, and the dimension being squeezed is not
            of length 1.
        """

        # make a copy of the original dims
        old = self.dims[:]

        new = self if inplace else self.copy()
        if dims and is_sequence(dims[0]):
            dims = dims[0]
        if dims:
            kwargs["dims"] = dims
        dims = self._get_dims_from_args(**kwargs)
        axes = self._get_dims_index(dims)
        axes = axes if axes is not None else ()
        axes = axes if is_sequence(axes) and axes is not None else axes
        keepaxes = self._get_dims_index(keepdims)
        keepaxes = keepaxes if keepaxes is not None else ()
        keepaxes = (
            keepaxes if is_sequence(keepaxes) and keepaxes is not None else [keepaxes]
        )
        if not axes and keepaxes:
            axes = np.arange(new.ndim)
            is_axis_to_remove = (np.array(new.shape) == 1) & (axes != keepaxes)
            axes = axes[is_axis_to_remove].tolist()
        elif not axes:
            arr = np.array(new.shape)
            axes = np.argwhere(arr == 1).squeeze().tolist()
            axes = [axes] if isinstance(axes, int) else axes
            is_axis_to_remove = arr == 1

        else:
            is_axis_to_remove = np.array([axis in axes for axis in np.arange(new.ndim)])
        # try to remove None from axes tuple or transform to () if axes is None
        axes = list(axes) if axes is not None else []
        axes.remove(None) if None in axes else axes
        axes = tuple(axes)
        if not axes:
            # nothing to squeeze
            if kwargs.get("return_index", False):
                return new, axes
            return new
        # recompute new dims by taking the dims not removed
        new._dims = np.array(new.dims)[~is_axis_to_remove].tolist()
        # performs all required squeezing
        new._data = new.data.squeeze(axis=axes)

        if axes is not None and new._coordset is not None:
            # if there are coordinates they have to be squeezed as well (remove
            # coordinate for the squeezed axis)

            for axis in axes:
                dim = old[axis]
                del new._coordset[dim]

        if kwargs.get("return_index", False):
            # in case we need to know which axis has been squeezed
            return new, axes
        return new

    @_docstring.dedent
    def swapdims(self, dim1, dim2, inplace=False):
        """
        Interchange two dims of a |NDDataset|.

        This method is quite similar to transpose.

        Parameters
        ----------
        dim1 : int or str
            First dimension index.
        dim2 : int
            Second dimension index.
        %(inplace)s

        Returns
        -------
        %(out)s
        """
        if self.ndim < 2:
            return self

        new = self if inplace else self.copy()
        i0_, i1_ = axis = self._get_dims_index([dim1, dim2])
        new._data = np.swapaxes(new.data, *axis)
        new._dims[i1_], new._dims[i0_] = self.dims[i0_], self.dims[i1_]

        # all other arrays have also to be swapped to reflect
        # changes of data ordering.
        new._meta = new._meta.swap(*axis, inplace=False)

        # we need also to swap the quaternion
        # WARNING: this work only for 2D
        # when swapdims is equivalent to a 2D transpose
        if self.is_hypercomplex:
            new._data = _transpose_hypercomplex(new.data)

        if self.is_masked:
            axis = self._get_dims_index([dim1, dim2])
            new._mask = np.swapaxes(new._mask, *axis)

        new.history = f"Data swapped between dims {dim1} and {dim2}"
        return new

    swapaxes = swapdims
    swapaxes.__doc__ = swapdims.__doc__

    @property
    def T(self):
        """
        Return a transposed array.

        See Also
        --------
        transpose : Permute the dimensions of an array.
        """
        return self.transpose()

    @_docstring.dedent
    def transpose(self, *dims, inplace=False):
        """
        Permute the dimensions of a NDDataset.

        If the `dims` are not specified, the order of the dimension is reversed.

        Parameters
        ----------
        *dims : list int or str
            Sequence of dimension indexes or names, optional.
            By default, reverse the dimensions, otherwise permute the dimensions
            according to the values given. If specified the list of dimension
            index or names must match the number of dimensions.
        %(inplace)s

        Returns
        -------
        %(out)s

        """
        new = self if inplace else self.copy()
        if self.ndim < 2:  # cannot transpose 1D data
            return new
        if not dims or list(set(dims)) == [None]:
            dims = self.dims[::-1]
        axis = self._get_dims_index(dims)
        new._data = np.transpose(new.data, axis)
        new._meta = new._meta.permute(*axis, inplace=False)
        new._dims = list(np.take(self.dims, axis))

        if new.is_hypercomplex:
            new._data = _transpose_hypercomplex(new.data)

        if new.is_masked:
            if self.ndim < 2:  # cannot transpose 1D data
                return new
            if not dims or list(set(dims)) == [None]:
                dims = self.dims[::-1]
            axis = self._get_dims_index(dims)
            new._mask = np.transpose(new._mask, axis)

        new.history = f"Data transposed between dims: {dims}" if dims else ""

        return new


class NDMath(object):
    """
    This class provides the math and some other array manipulation functionalities to |NDArray| or |Coord| .

    Below is a list of mathematical functions (numpy) implemented (or
    planned for implementation).

    **Ufuncs**

    These functions should work like for numpy-ndarray, except that they
    may be units-aware.

    For instance, `ds`  being a |NDDataset| , just call the np functions like
    this. Most of the time it returns a new NDDataset, while in some cases
    noted below, one get a |ndarray| .

    >>> ds = scp.NDDataset([1., 2., 3.])
    >>> np.sin(ds)
    NDDataset: [float64] unitless (size: 3)

    In this particular case (*i.e.*, `np.sin` ufuncs) , the `ds` units must be
    `unitless` , `dimensionless` or angle-units : `radians` or `degrees` ,
    or an exception will be raised.

    Examples
    --------
    >>> nd1 = scp.read('wodger.spg')
    >>> nd1
    NDDataset: [float64] a.u. (shape: (y:2, x:5549))
    >>> nd1.data
    array([[   2.005,    2.003, ...,    1.826,    1.831],
           [   1.983,    1.984, ...,    1.698,    1.704]])
    >>> nd2 = np.negative(nd1)
    >>> nd2
    NDDataset: [float64] a.u. (shape: (y:2, x:5549))
    >>> nd2.data
    array([[  -2.005,   -2.003, ...,   -1.826,   -1.831],
           [  -1.983,   -1.984, ...,   -1.698,   -1.704]])
    """

    _radian = "radian"
    _degree = "degree"
    _require_units = {
        "cumprod": DIMENSIONLESS,
        "arccos": DIMENSIONLESS,
        "arcsin": DIMENSIONLESS,
        "arctan": DIMENSIONLESS,
        "arccosh": DIMENSIONLESS,
        "arcsinh": DIMENSIONLESS,
        "arctanh": DIMENSIONLESS,
        "exp": DIMENSIONLESS,
        "expm1": DIMENSIONLESS,
        "exp2": DIMENSIONLESS,
        "log": DIMENSIONLESS,
        "log10": DIMENSIONLESS,
        "log1p": DIMENSIONLESS,
        "log2": DIMENSIONLESS,
        "sin": _radian,
        "cos": _radian,
        "tan": _radian,
        "sinh": _radian,
        "cosh": _radian,
        "tanh": _radian,
        "radians": _degree,
        "degrees": _radian,
        "deg2rad": _degree,
        "rad2deg": _radian,
        "logaddexp": DIMENSIONLESS,
        "logaddexp2": DIMENSIONLESS,
    }
    _compatible_units = [
        "add",
        "sub",
        "iadd",
        "isub",
        "maximum",
        "minimum",
        "fmin",
        "fmax",
        "lt",
        "le",
        "ge",
        "gt",
    ]
    _complex_funcs = ["real", "imag", "absolute", "abs"]
    _keep_title = [
        "negative",
        "absolute",
        "abs",
        "fabs",
        "rint",
        "floor",
        "ceil",
        "trunc",
        "add",
        "subtract",
    ]
    _remove_title = [
        "multiply",
        "divide",
        "true_divide",
        "floor_divide",
        "mod",
        "fmod",
        "remainder",
        "logaddexp",
        "logaddexp2",
    ]
    _remove_units = [
        "logical_not",
        "isfinite",
        "isinf",
        "isnan",
        "isnat",
        "isneginf",
        "isposinf",
        "iscomplex",
        "signbit",
        "sign",
    ]
    _quaternion_aware = [
        "add",
        "iadd",
        "sub",
        "isub",
        "mul",
        "imul",
        "div",
        "idiv",
        "log",
        "exp",
        "power",
        "negative",
        "conjugate",
        "copysign",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "isnan",
        "isinf",
        "isfinite",
        "absolute",
        "abs",
    ]
    _require_same_shape = list(_binary_ufuncs().keys()) + [
        "iadd",
        "isub",
        "imul",
        "idiv",
    ]

    # the following methods are to give NDArray based class
    # a behavior similar to np.ndarray regarding the ufuncs

    def __array_function__(self, *args, **kwargs):
        # should be defined in subclass
        return NotImplemented

    @property
    def __array_struct__(self):
        return self.data.__array_struct__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        fname = ufunc.__name__

        #        # case of complex or hypercomplex data
        #        if self._implements(NDComplexArray) and self.has_complex_dims:
        #
        #            if fname in self._complex_funcs:
        #                return getattr(inputs[0], fname)()
        #
        #            if fname in ["fabs", ]:
        #                # function not available for complex data
        #                raise ValueError(f"Operation `{ufunc}` does not accept complex data!")
        #
        #        # If this reached, data are not complex or hypercomplex
        #        if fname in ['absolute', 'abs']:
        #            f = np.fabs

        # set history string
        history = f"Ufunc {fname} applied."

        if fname in ["sign", "logical_not", "isnan", "isfinite", "isinf", "signbit"]:
            return (getattr(np, fname))(inputs[0].masked_data)

        # case of a dataset
        data, units, mask, returntype = self._op(ufunc, inputs, isufunc=True)
        new = self._op_result(data, units, mask, history, returntype)

        # make a new title depending on the operation
        if fname in self._remove_title:
            new.title = f"<{fname}>"
        elif fname not in self._keep_title and isinstance(new, NDArray):
            if hasattr(new, "title") and new.title is not None:
                new.title = f"{fname}({new.title})"
            else:
                new.title = f"{fname}(data)"
        return new

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    @_from_numpy_method
    def absolute(cls, dataset, dtype=None):
        """
        Calculate the absolute value of the given NDDataset element-wise.

        `abs` is a shorthand for this function. For complex input, a + ib, the absolute value is
        :math:`\\sqrt{ a^2 + b^2}`.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dtype : dtype
            The type of the output array. If dtype is not given, infer the data type from the other input arguments.

        Returns
        -------
        absolute
            An ndarray containing the absolute value of each element in dataset.
        """

        if not cls.has_complex_dims:
            data = np.ma.fabs(
                dataset, dtype=dtype
            )  # not a complex, return fabs should be faster

        elif not cls.is_quaternion:
            data = np.ma.sqrt(dataset.real ** 2 + dataset.imag ** 2)

        else:
            data = np.ma.sqrt(
                dataset.real ** 2
                + dataset.part("IR") ** 2
                + dataset.part("RI") ** 2
                + dataset.part("II") ** 2,
                dtype=dtype,
            )
            cls._is_quaternion = False

        cls._data = data.data
        cls._mask = data.mask

        return cls

    abs = absolute
    abs.__doc__ = (
        "Calculate the absolute value element-wise.\n\nEquivalent to absolute."
    )

    @_from_numpy_method
    def conjugate(cls, dataset, dim="x"):
        """
        Conjugate of the NDDataset in the specified dimension.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : int, str, optional, default=(0,)
            Dimension names or indexes along which the method should be applied.

        Returns
        -------
        conjugated
            Same object or a copy depending on the ``inplace`` flag.

        See Also
        --------
        conj, real, imag, RR, RI, IR, II, part, set_complex, is_complex
        """

        axis, dim = cls._get_axis(dim, allows_none=True)

        if cls.is_quaternion:
            # TODO:
            dataset = dataset.swapdims(axis, -1)
            dataset[..., 1::2] = -dataset[..., 1::2]
            dataset = dataset(axis, -1)
        else:
            dataset = np.ma.conjugate(dataset)

        cls._data = dataset.data
        cls._mask = dataset.mask

        return cls

    conj = conjugate
    conj.__doc__ = (
        "Conjugate of the NDDataset in the specified dimension."
        "\n\nEquivalent to conjugate."
    )

    @_from_numpy_method
    def around(cls, dataset, decimals=0):
        """
        Evenly round to the given number of decimals.

        Parameters
        ----------
        dataset : |NDDataset|
            Input dataset.
        decimals : int, optional
            Number of decimal places to round to (default: 0).  If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point.

        Returns
        -------
        rounded_array
            NDDataset containing the rounded values.
            The real and imaginary parts of complex numbers are rounded
            separately.
            The result of rounding a float is a float.
            If the dataset contains masked data, the mask remain unchanged.

        See Also
        --------
        numpy.round, around, spectrochempy.round, spectrochempy.around: Equivalent methods.
        ceil, fix, floor, rint, trunc
        """

        m = np.ma.round(dataset, decimals)
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m

        return cls

    round = around
    round.__doc__ = (
        "Evenly round to the given number of decimals.\n\nEquivalent to around."
    )
    round_ = around
    round.__doc__ = (
        "Evenly round to the given number of decimals.\n\nEquivalent to around."
    )

    @_reduce_method
    @_from_numpy_method
    def all(cls, dataset, dim=None, keepdims=False):
        """
        Test whether all array elements along a given axis evaluate to True.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or str, optional
            Axis or axes along which a logical AND reduction is performed.
            The default (``axis=None``) is to perform a logical AND over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            If the default value is passed, then `keepdims` will not be
            passed through to the `all` method of sub-classes of
            `ndarray`, however any non-default value will be.  If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised.

        Returns
        -------
        all
            A new boolean or array is returned unless `out` is specified,
            in which case a reference to `out` is returned.

        See Also
        --------
        any : Test whether any element along a given axis evaluates to True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to `True` because these are not equal to zero.
        """
        axis, dim = cls._get_axis(dim, allows_none=True)
        data = np.all(dataset, axis, keepdims=keepdims)
        return data

    @_reduce_method
    @_from_numpy_method
    def amax(cls, dataset, dim=None, keepdims=False, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or dimension name or tuple of int or dimensions, optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
            If this is a tuple, the maximum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        amax
            Maximum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension ``ndim - 1``.

        See Also
        --------
        amin : The minimum value of a dataset along a given dimension, propagating any NaNs.
        minimum : Element-wise minimum of two datasets, propagating any NaNs.
        maximum : Element-wise maximum of two datasets, propagating any NaNs.
        fmax : Element-wise maximum of two datasets, ignoring any NaNs.
        fmin : Element-wise minimum of two datasets, ignoring any NaNs.
        argmax : Return the indices or coordinates of the maximum values.
        argmin : Return the indices or coordinates of the minimum values.

        Notes
        -----
        For dataset with complex or hypercomplex type type, the default is the
        value with the maximum real part.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        quaternion = False
        if dataset.dtype in [np.quaternion]:
            # from quaternion import as_float_array

            quaternion = True
            data = dataset
            dataset = as_float_array(dataset)[..., 0]  # real part
        m = np.ma.max(dataset, axis=axis, keepdims=keepdims)
        if quaternion:
            if dim is None:
                # we return the corresponding quaternion value
                idx = np.ma.argmax(dataset)
                c = list(np.unravel_index(idx, dataset.shape))
                m = data[..., c[-2], c[-1]][()]
            else:
                m = np.ma.diag(data[np.ma.argmax(dataset, axis=axis)])

        if np.isscalar(m) or (m.size == 1 and not keepdims):
            if not np.isscalar(m):  # case of quaternion
                m = m[()]
            if cls.units is not None:
                return Quantity(m, cls.units)
            return m

        dims = cls.dims
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m

        # Here we must eventually reduce the corresponding coordinates
        if hasattr(cls, "coordset"):
            coordset = cls.coordset
            if coordset is not None:
                if dim is not None:
                    idx = coordset.names.index(dim)
                    if not keepdims:
                        del coordset.coords[idx]
                        dims.remove(dim)
                    else:
                        coordset.coords[idx].data = [
                            0,
                        ]
                else:
                    # find the coordinates
                    idx = np.ma.argmax(dataset)
                    c = list(np.unravel_index(idx, dataset.shape))

                    coord = {}
                    for i, item in enumerate(c[::-1]):
                        dim = dims[-(i + 1)]
                        id = coordset.names.index(dim)
                        coord[dim] = coordset.coords[id][item]
                    cls.set_coordset(coord)

        cls.dims = dims
        return cls

    max = amax

    @_reduce_method
    @_from_numpy_method
    def amin(cls, dataset, dim=None, keepdims=False, **kwargs):
        """
        Return the maximum of the dataset or maxima along given dimensions.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or dimension name or tuple of int or dimensions, optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
            If this is a tuple, the minimum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        amin
            Minimum of the data. If `dim` is None, the result is a scalar value.
            If `dim` is given, the result is an array of dimension ``ndim - 1``.

        See Also
        --------
        amax : The maximum value of a dataset along a given dimension, propagating any NaNs.
        minimum : Element-wise minimum of two datasets, propagating any NaNs.
        maximum : Element-wise maximum of two datasets, propagating any NaNs.
        fmax : Element-wise maximum of two datasets, ignoring any NaNs.
        fmin : Element-wise minimum of two datasets, ignoring any NaNs.
        argmax : Return the indices or coordinates of the maximum values.
        argmin : Return the indices or coordinates of the minimum values.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        quaternion = False
        if dataset.dtype in [np.quaternion]:
            # from quaternion import as_float_array

            quaternion = True
            data = dataset
            dataset = as_float_array(dataset)[..., 0]  # real part
        m = np.ma.min(dataset, axis=axis, keepdims=keepdims)
        if quaternion:
            if dim is None:
                # we return the corresponding quaternion value
                idx = np.ma.argmin(dataset)
                c = list(np.unravel_index(idx, dataset.shape))
                m = data[..., c[-2], c[-1]][()]
            else:
                m = np.ma.diag(data[np.ma.argmin(dataset, axis=axis)])

        if np.isscalar(m) or (m.size == 1 and not keepdims):
            if not np.isscalar(m):  # case of quaternion
                m = m[()]
            if cls.units is not None:
                return Quantity(m, cls.units)
            return m

        dims = cls.dims
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m

        # Here we must eventually reduce the corresponding coordinates
        if hasattr(cls, "coordset"):
            coordset = cls.coordset
            if coordset is not None:
                if dim is not None:
                    idx = coordset.names.index(dim)
                    if not keepdims:
                        del coordset.coords[idx]
                        dims.remove(dim)
                    else:
                        coordset.coords[idx].data = [
                            0,
                        ]
                else:
                    # find the coordinates
                    idx = np.ma.argmin(dataset)
                    c = list(np.unravel_index(idx, dataset.shape))

                    coord = {}
                    for i, item in enumerate(c[::-1]):
                        dim = dims[-(i + 1)]
                        id = coordset.names.index(dim)
                        coord[dim] = coordset.coords[id][item]
                    cls.set_coordset(coord)

        cls.dims = dims
        return cls

    min = amin

    @_reduce_method
    @_from_numpy_method
    def any(cls, dataset, dim=None, keepdims=False):
        """
        Test whether any array element along a given axis evaluates to True.

        Returns single boolean unless `dim` is not ``None``

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        dim : None or int or tuple of ints, optional
            Axis or axes along which a logical OR reduction is performed.
            The default (``axis=None``) is to perform a logical OR over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
            If the default value is passed, then `keepdims` will not be
            passed through to the `any` method of sub-classes of
            `ndarray`, however any non-default value will be.  If the
            sub-class' method does not implement `keepdims` any
            exceptions will be raised.

        Returns
        -------
        any
            A new boolean or `ndarray` is returned.

        See Also
        --------
        all : Test whether all array elements along a given axis evaluate to True.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        data = np.any(dataset, axis, keepdims=keepdims)
        return data

    @_from_numpy_method
    def arange(cls, start=0, stop=None, step=None, dtype=None, **kwargs):
        """
        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval [start, stop).

        Parameters
        ----------
        start : number, optional
            Start of interval. The interval includes this value. The default start value is 0.
        stop : number
            End of interval. The interval does not include this value, except in some cases
            where step is not an integer and floating point round-off affects the length of out.
            It might be prefereble to use inspace in such case.
        step : number, optional
            Spacing between values. For any output out, this is the distance between two adjacent values,
            out[i+1] - out[i]. The default step size is 1. If step is specified as a position argument,
            start must also be given.
        dtype : dtype
            The type of the output array. If dtype is not given, infer the data type from the other input arguments.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        arange
            Array of evenly spaced values.

        See Also
        --------
        linspace : Evenly spaced numbers with careful handling of endpoints.

        Examples
        --------

        >>> scp.arange(1, 20.0001, 1, units='s', name='mycoord')
        NDDataset: [float64] s (size: 20)
        """

        return cls(np.arange(start, stop, step, dtype), **kwargs)

    @_reduce_method
    @_from_numpy_method
    def argmax(cls, dataset, dim=None):
        """
        Indexes of maximum of data along axis.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        idx = np.ma.argmax(dataset, axis)
        if cls.ndim > 1 and axis is None:
            idx = np.unravel_index(idx, cls.shape)
        return idx

    @_reduce_method
    @_from_numpy_method
    def argmin(cls, dataset, dim=None):
        """
        Indexes of minimum of data along axis.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        idx = np.ma.argmin(dataset, axis)
        if cls.ndim > 1 and axis is None:
            idx = np.unravel_index(idx, cls.shape)
        return idx

    @_reduce_method
    @_from_numpy_method
    def average(cls, dataset, dim=None, weights=None, keepdims=False):
        """
        Compute the weighted average along the specified axis.

        Parameters
        ----------
        dataset : array_like
            Array containing data to be averaged.
        dim : None or int or dimension name or tuple of int or dimensions, optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
            If this is a tuple, the minimum is selected over multiple dimensions,
            instead of a single dimension or all the dimensions as before.
        weights : array_like, optional
            An array of weights associated with the values in `dataset`. Each value in
            `a` contributes to the average according to its associated weight.
            The weights array can either be 1-D (in which case its length must be
            the size of `dataset` along the given axis) or of the same shape as `dataset`.
            If `weights=None`, then all data in `dataset` are assumed to have a
            weight equal to one.  The 1-D calculation is::

                avg = sum(a * weights) / sum(weights)

            The only constraint on `weights` is that `sum(weights)` must not be 0.

        Returns
        -------
        average,
            Return the average along the specified axis.

        Raises
        ------
        ZeroDivisionError
            When all weights along axis are zero. See `numpy.ma.average` for a
            version robust to this type of error.
        TypeError
            When the length of 1D `weights` is not the same as the shape of `a`
            along axis.

        See Also
        --------
        mean : Compute the arithmetic mean along the specified axis.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.average(nd)
        <Quantity(1.25085858, 'absorbance')>
        >>> m = scp.average(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.x
        LinearCoord: [float64] cm⁻¹ (size: 5549)
        >>> m = scp.average(nd, dim='y', weights=np.arange(55))
        >>> m.data
        array([   1.789,    1.789, ...,    1.222,     1.22])
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m, sumweight = np.ma.average(dataset, axis=axis, weights=weights, returned=True)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m
        cls.dims = dims

        return cls

    @_from_numpy_method
    def clip(cls, dataset, a_min=None, a_max=None, **kwargs):
        """
        Clip (limit) the values in a dataset.

        Given an interval, values outside the interval are clipped to
        the interval edges.  For example, if an interval of ``[0, 1]``
        is specified, values smaller than 0 become 0, and values larger
        than 1 become 1.

        No check is performed to ensure ``a_min < a_max``.

        Parameters
        ----------
        dataset : array_like
            Input array or object that can be converted to an array.
        a_min : scalar or array_like or None
            Minimum value. If None, clipping is not performed on lower
            interval edge. Not more than one of `a_min` and `a_max` may be
            None.
        a_max : scalar or array_like or None
            Maximum value. If None, clipping is not performed on upper
            interval edge. Not more than one of `a_min` and `a_max` may be
            None. If `a_min` or `a_max` are array_like, then the three
            arrays will be broadcasted to match their shapes.

        Returns
        -------
        clip
            An array with the elements of `a`, but where values
            < `a_min` are replaced with `a_min`, and those > `a_max`
            with `a_max`.
        """
        # if len(args) > 2 or len(args) == 0:
        #     raise ValueError('Clip requires at least one argument or at most two arguments')
        # amax = kwargs.pop('a_max', args[0] if len(args) == 1 else args[1])
        # amin = kwargs.pop('a_min', self.min() if len(args) == 1 else args[0])
        # amin, amax = np.minimum(amin, amax), max(amin, amax)
        # if self.has_units:
        #     if not isinstance(amin, Quantity):
        #         amin = amin * self.units
        #     if not isinstance(amax, Quantity):
        #         amax = amax * self.units
        # res = self._method('clip', a_min=amin, a_max=amax, **kwargs)
        # res.history = f'Clipped with limits between {amin} and {amax}'
        # return res

        m = np.ma.clip(dataset, a_min, a_max)
        if hasattr(m, "mask"):
            cls._data = m.data
            cls._mask = m.mask
        else:
            cls._data = m
        return cls

    @_reduce_method
    @_from_numpy_method
    def coordmax(cls, dataset, dim=None):
        """
        Find coordinates of the maximum of data along axis.
        """

        if not cls.implements("NDDataset") or cls.coordset is None:
            raise Exception(
                "Method `coordmax` apply only on NDDataset and if it has defined coordinates"
            )

        axis, dim = cls._get_axis(dim, allows_none=True)

        idx = np.ma.argmax(dataset, fill_value=-1e30)
        cmax = list(np.unravel_index(idx, dataset.shape))

        dims = cls.dims
        coordset = cls.coordset.copy()

        coord = {}
        for i, item in enumerate(cmax[::-1]):
            _dim = dims[-(i + 1)]
            coord[_dim] = coordset[_dim][item].values

        if cls._squeeze_ndim == 1:
            dim = dims[-1]

        if dim is not None:
            return coord[dim]

        return coord

    @_reduce_method
    @_from_numpy_method
    def coordmin(cls, dataset, dim=None):
        """
        Find oordinates of the mainimum of data along axis.
        """

        if not cls.implements("NDDataset") or cls.coordset is None:
            raise Exception(
                "Method `coordmin` apply only on NDDataset and if it has defined coordinates"
            )

        axis, dim = cls._get_axis(dim, allows_none=True)

        idx = np.ma.argmin(dataset, fill_value=-1e30)
        cmax = list(np.unravel_index(idx, cls.shape))

        dims = cls.dims
        coordset = cls.coordset

        coord = {}
        for i, item in enumerate(cmax[::-1]):
            _dim = dims[-(i + 1)]
            coord[_dim] = coordset[_dim][item].values

        if cls._squeeze_ndim == 1:
            dim = dims[-1]

        if dim is not None:
            return coord[dim]

        return coord

    @_from_numpy_method
    def cumsum(cls, dataset, dim=None, dtype=None):
        """
        Return the cumulative sum of the elements along a given axis.

        Parameters
        ----------
        dataset : array_like
            Calculate the cumulative sum of these values.
        dim : None or int or dimension name , optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.

        Returns
        -------
        sum
            A new array containing the cumulative sum.

        See Also
        --------
        sum : Sum array elements.
        trapezoid : Integration of array values using the composite trapezoidal rule.
        diff :  Calculate the n-th discrete difference along given axis.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.sum(nd)
        <Quantity(381755.783, 'absorbance')>
        >>> scp.sum(nd, keepdims=True)
        NDDataset: [float64] a.u. (shape: (y:1, x:1))
        >>> m = scp.sum(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.data
        array([   100.7,    100.7, ...,       74,    73.98])
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        data = np.ma.cumsum(dataset, axis=axis, dtype=dtype)
        cls._data = data
        return cls

    @_from_numpy_method
    def diag(cls, dataset, offset=0, **kwargs):
        """
        Extract a diagonal or construct a diagonal array.

        See the more detailed documentation for ``numpy.diagonal`` if you use this
        function to extract a diagonal and wish to write to the resulting array;
        whether it returns a copy or a view depends on what version of numpy you
        are using.

        Parameters
        ----------
        dataset : array_like
            If `dataset` is a 2-D array, return a copy of its `k`-th diagonal.
            If `dataset` is a 1-D array, return a 2-D array with `v` on the `k`-th.
            diagonal.
        offset : int, optional
            Diagonal in question. The default is 0. Use offset>0 for diagonals above the main diagonal, and offset<0 for
            diagonals below the main diagonal.

        Returns
        -------
        diag
            The extracted diagonal or constructed diagonal array.
        """

        new = cls

        if new.ndim == 1:

            # construct a diagonal array
            # --------------------------

            data = np.diag(new.data)

            mask = NOMASK
            if new.is_masked:
                size = new.size
                m = np.repeat(new.mask, size).reshape(size, size)
                mask = m | m.T

            coordset = None
            if new.coordset is not None:
                coordset = (new.coordset[0], new.coordset[0])

            dims = ["y"] + new.dims

            new.data = data
            new.mask = mask
            new._dims = dims
            if coordset is not None:
                new.set_coordset(coordset)
            return new

        if new.ndim == 2:

            # extract a diagonal
            # ------------------
            return new.diagonal(offset=offset, **kwargs)

        raise ValueError("Input must be 1- or 2-d.")

    @_reduce_method
    @_from_numpy_method
    def diagonal(cls, dataset, offset=0, dim="x", dtype=None, **kwargs):
        """
        Return the diagonal of a 2D array.

        As we reduce a 2D to a 1D we must specified which is the dimension for the coordinates to keep!.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to extract the diagonal.
        offset : int, optional
            Offset of the diagonal from the main diagonal.  Can be positive or
            negative.  Defaults to main diagonal (0).
        dim : str, optional
            Dimension to keep for coordinates. By default it is the last (-1, `x` or another name if the default
            dimension name has been modified).
        dtype : dtype, optional
            The type of the returned array.
        **kwargs
            Additional keyword parameters to be passed to the NDDataset constructor.

        Returns
        -------
        diagonal
            The diagonal of the input array.

        See Also
        --------
        diag : Extract a diagonal or construct a diagonal array.

        Examples
        --------

        >>> nd = scp.full((2, 2), 0.5, units='s', title='initial')
        >>> nd
        NDDataset: [float64] s (shape: (y:2, x:2))
        >>> nd.diagonal(title='diag')
        NDDataset: [float64] s (size: 2)
        """

        axis, dim = cls._get_axis(dim)
        if hasattr(dataset, "mask"):
            data = np.ma.diagonal(dataset, offset=offset)
            cls._data = data
            cls._mask = data.mask
        else:
            cls._data = np.diagonal(dataset, offset=offset)

        if dtype is not None:
            cls.data = cls.data.astype(dtype)
        cls._history = []

        # set the new coordinates
        if hasattr(cls, "coordset") and cls.coordset is not None:
            idx = cls._coordset.names.index(dim)
            cls.set_coordset({dim: cls._coordset.coords[idx][: cls.size]})
            cls.dims = [dim]

        return cls

    @_from_numpy_method
    def empty(cls, shape, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, without initializing entries.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the empty array.
        dtype : data-type, optional
            Desired output data-type.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        empty
            Array of uninitialized (arbitrary) data of the given shape, dtype, and
            order.  Object arrays will be initialized to None.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to 1.
        full : Fill a new array.

        Notes
        -----
        `empty`, unlike `zeros`, does not set the array values to zero,
        and may therefore be marginally faster.  On the other hand, it requires
        the user to manually set all the values in the array, and should be
        used with caution.

        Examples
        --------

        >>> scp.empty([2, 2], dtype=int, units='s')
        NDDataset: [int64] s (shape: (y:2, x:2))
        """

        return cls(np.empty(shape, dtype), dtype=dtype, **kwargs)

    @_from_numpy_method
    def empty_like(cls, dataset, dtype=None, **kwargs):
        """
        Return a new uninitialized |NDDataset|.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        emptylike
            Array of `fill_value` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of the input.
        ones_like : Return an array of ones with shape and type of input.
        zeros_like : Return an array of zeros with shape and type of input.
        empty : Return a new uninitialized array.
        ones : Return a new array setting values to one.
        zeros : Return a new array setting values to zero.
        full : Fill a new array.

        Notes
        -----
        This function does *not* initialize the returned array; to do that use
        for instance `zeros_like`, `ones_like` or `full_like` instead.  It may be
        marginally faster than the functions that do set the array values.
        """

        cls._data = np.empty_like(dataset, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    @_from_numpy_method
    def eye(cls, N, M=None, k=0, dtype=float, **kwargs):
        """
        Return a 2-D array with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        M : int, optional
            Number of columns in the output. If None, defaults to `N`.
        k : int, optional
            Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype : data-type, optional
            Data-type of the returned array.
        **kwargs
            Other parameters to be passed to the object constructor (units, coordset, mask ...).

        Returns
        -------
        eye
            NDDataset of shape (N,M)
            An array where all elements are equal to zero, except for the `k`-th
            diagonal, whose values are equal to one.

        See Also
        --------
        identity : Equivalent function with k=0.
        diag : Diagonal 2-D NDDataset from a 1-D array specified by the user.

        Examples
        --------

        >>> scp.eye(2, dtype=int)
        NDDataset: [float64] unitless (shape: (y:2, x:2))
        >>> scp.eye(3, k=1, units='km').values
        <Quantity([[       0        1        0]
         [       0        0        1]
         [       0        0        0]], 'kilometer')>
        """

        return cls(np.eye(N, M, k, dtype), **kwargs)

    @_from_numpy_method
    def fromfunction(
        cls, function, shape=None, dtype=float, units=None, coordset=None, **kwargs
    ):
        """
        Construct a nddataset by executing a function over each coordinate.

        The resulting array therefore has a value ``fn(x, y, z)`` at coordinate ``(x, y, z)`` .

        Parameters
        ----------
        function : callable
            The function is called with N parameters, where N is the rank of
            `shape` or from the provided `coordset`.
        shape : (N,) tuple of ints, optional
            Shape of the output array, which also determines the shape of
            the coordinate arrays passed to `function`. It is optional only if
            `coordset` is None.
        dtype : data-type, optional
            Data-type of the coordinate arrays passed to `function`.
            By default, `dtype` is float.
        units : str, optional
            Dataset units.
            When None, units will be determined from the function results.
        coordset : |Coordset| instance, optional
            If provided, this determine the shape and coordinates of each dimension of
            the returned |NDDataset|. If shape is also passed it will be ignored.
        **kwargs
            Other kwargs are passed to the final object constructor.

        Returns
        -------
        fromfunction
            The result of the call to `function` is passed back directly.
            Therefore the shape of `fromfunction` is completely determined by
            `function`.

        See Also
        --------
        fromiter : Make a dataset from an iterable.

        Examples
        --------
        Create a 1D NDDataset from a function

        >>> func1 = lambda t, v: v * t
        >>> time = scp.LinearCoord.arange(0, 60, 10, units='min')
        >>> d = scp.fromfunction(func1, v=scp.Quantity(134, 'km/hour'), coordset=scp.CoordSet(t=time))
        >>> d.dims
        ['t']
        >>> d
        NDDataset: [float64] km (size: 6)
        """

        from spectrochempy.core.dataset.coord import CoordSet

        if coordset is not None:
            if not isinstance(coordset, CoordSet):
                coordset = CoordSet(*coordset)
            shape = coordset.sizes

        idx = np.indices(shape)

        args = [0] * len(shape)
        if coordset is not None:
            for i, co in enumerate(coordset):
                args[i] = co.data[idx[i]]
                if units is None and co.has_units:
                    args[i] = Quantity(args[i], co.units)

        kw = kwargs.pop("kw", {})
        data = function(*args, **kw)

        data = data.T
        dims = coordset.names[::-1]
        new = cls(data, coordset=coordset, dims=dims, units=units, **kwargs)
        new.ito_reduced_units()
        return new

    @_from_numpy_method
    def fromiter(cls, iterable, dtype=np.float64, count=-1, **kwargs):
        """
        Create a new 1-dimensional array from an iterable object.

        Parameters
        ----------
        iterable : iterable object
            An iterable object providing data for the array.
        dtype : data-type
            The data-type of the returned array.
        count : int, optional
            The number of items to read from iterable. The default is -1, which means all data is read.
        **kwargs
            Other kwargs are passed to the final object constructor.

        Returns
        -------
        fromiter
            The output nddataset.

        See Also
        --------
        fromfunction : Construct a nddataset by executing a function over each coordinate.

        Notes
        -----
            Specify count to improve performance. It allows fromiter to pre-allocate the output array,
            instead of resizing it on demand.

        Examples
        --------

        >>> iterable = (x * x for x in range(5))
        >>> d = scp.fromiter(iterable, float, units='km')
        >>> d
        NDDataset: [float64] km (size: 5)
        >>> d.data
        array([       0,        1,        4,        9,       16])
        """

        return cls(np.fromiter(iterable, dtype=dtype, count=count), **kwargs)

    @_from_numpy_method
    def full(cls, shape, fill_value=0.0, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with `fill_value`.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `np.int8`.  Default is fill_value.dtype.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        full
            Array of `fill_value`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.

        Examples
        --------

        >>> scp.full((2, ), np.inf)
        NDDataset: [float64] unitless (size: 2)
        >>> scp.NDDataset.full((2, 2), 10, dtype=np.int)
        NDDataset: [int64] unitless (shape: (y:2, x:2))
        """

        return cls(np.full(shape, fill_value, dtype), dtype=dtype, **kwargs)

    @_from_numpy_method
    def full_like(cls, dataset, fill_value=0.0, dtype=None, **kwargs):
        """
        Return a |NDDataset| of fill_value.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        fulllike
            Array of `fill_value` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------
        3 possible ways to call this method

        1) from the API

        >>> x = np.arange(6, dtype=int)
        >>> scp.full_like(x, 1)
        NDDataset: [float64] unitless (size: 6)

        2) as a classmethod

        >>> x = np.arange(6, dtype=int)
        >>> scp.NDDataset.full_like(x, 1)
        NDDataset: [float64] unitless (size: 6)

        3) as an instance method

        >>> scp.NDDataset(x).full_like(1, units='km')
        NDDataset: [float64] km (size: 6)
        """

        cls._data = np.full_like(dataset, fill_value, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    @_from_numpy_method
    def geomspace(cls, start, stop, num=50, endpoint=True, dtype=None, **kwargs):
        """
        Return numbers spaced evenly on a log scale (a geometric progression).

        This is similar to `logspace`, but with endpoints specified directly.
        Each output sample is a constant multiple of the previous.

        Parameters
        ----------
        start : number
            The starting value of the sequence.
        stop : number
            The final value of the sequence, unless `endpoint` is False.
            In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        num : int, optional
            Number of samples to generate.  Default is 50.
        endpoint : bool, optional
            If true, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, infer the data
            type from the other input arguments.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        geomspace
            `num` samples, equally spaced on a log scale.

        See Also
        --------
        logspace : Similar to geomspace, but with endpoints specified using log
                   and base.
        linspace : Similar to geomspace, but with arithmetic instead of geometric
                   progression.
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples.
        """

        return cls(np.geomspace(start, stop, num, endpoint, dtype), **kwargs)

    @_from_numpy_method
    def identity(cls, n, dtype=None, **kwargs):
        """
        Return the identity |NDDataset| of a given shape.

        The identity array is a square array with ones on
        the main diagonal.

        Parameters
        ----------
        n : int
            Number of rows (and columns) in `n` x `n` output.
        dtype : data-type, optional
            Data-type of the output.  Defaults to ``float``.
        **kwargs
            Other parameters to be passed to the object constructor (units, coordset, mask ...).

        Returns
        -------
        identity
            `n` x `n` array with its main diagonal set to one,
            and all other elements 0.

        See Also
        --------
        eye : Almost equivalent function.
        diag : Diagonal 2-D array from a 1-D array specified by the user.

        Examples
        --------

        >>> scp.identity(3).data
        array([[       1,        0,        0],
               [       0,        1,        0],
               [       0,        0,        1]])
        """

        return cls(np.identity(n, dtype), **kwargs)

    @_from_numpy_method
    def linspace(
        cls, start, stop, num=50, endpoint=True, retstep=False, dtype=None, **kwargs
    ):
        """
        Return evenly spaced numbers over a specified interval.

        Returns num evenly spaced samples, calculated over the interval [start, stop]. The endpoint of the interval
        can optionally be excluded.

        Parameters
        ----------
        start : array_like
            The starting value of the sequence.
        stop : array_like
            The end value of the sequence, unless endpoint is set to False.
            In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is
            excluded. Note that the step size changes when endpoint is False.
        num : int, optional
            Number of samples to generate. Default is 50. Must be non-negative.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included. Default is True.
        retstep : bool, optional
            If True, return (samples, step), where step is the spacing between samples.
        dtype : dtype, optional
            The type of the array. If dtype is not given, infer the data type from the other input arguments.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        linspace : ndarray
            There are num equally spaced samples in the closed interval [start, stop] or the half-open interval
            [start, stop) (depending on whether endpoint is True or False).
        step : float, optional
            Only returned if retstep is True
            Size of spacing between samples.
        """

        return cls(np.linspace(start, stop, num, endpoint, retstep, dtype), **kwargs)

    @_from_numpy_method
    def logspace(
        cls, start, stop, num=50, endpoint=True, base=10.0, dtype=None, **kwargs
    ):
        """
        Return numbers spaced evenly on a log scale.

        In linear space, the sequence starts at ``base ** start``
        (`base` to the power of `start`) and ends with ``base ** stop``
        (see `endpoint` below).

        Parameters
        ----------
        start : array_like
            ``base ** start`` is the starting value of the sequence.
        stop : array_like
            ``base ** stop`` is the final value of the sequence, unless `endpoint`
            is False.  In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        num : int, optional
            Number of samples to generate.  Default is 50.
        endpoint : bool, optional
            If true, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        base : float, optional
            The base of the log space. The step size between the elements in
            ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
            Default is 10.0.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, infer the data
            type from the other input arguments.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        logspace
            `num` samples, equally spaced on a log scale.

        See Also
        --------
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples. Note that, when used with a float endpoint, the
                 endpoint may or may not be included.
        linspace : Similar to logspace, but with the samples uniformly distributed
                   in linear space, instead of log space.
        geomspace : Similar to logspace, but with endpoints specified directly.
        """

        return cls(np.logspace(start, stop, num, endpoint, base, dtype), **kwargs)

    @_reduce_method
    @_from_numpy_method
    def mean(cls, dataset, dim=None, dtype=None, keepdims=False):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array elements.  The average is taken over
        the flattened array by default, otherwise over the specified axis.

        Parameters
        ----------
        dataset : array_like
            Array containing numbers whose mean is desired.
        dim : None or int or dimension name, optional
            Dimension or dimensions along which to operate.
        dtype : data-type, optional
            Type to use in computing the mean.  For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        mean
            A new array containing the mean values.

        See Also
        --------
        average : Weighted average.
        std : Standard deviation values along axis.
        var : Variance values along axis.

        Notes
        -----
        The arithmetic mean is the sum of the elements along the axis divided
        by the number of elements.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.mean(nd)
        <Quantity(1.25085858, 'absorbance')>
        >>> scp.mean(nd, keepdims=True)
        NDDataset: [float64] a.u. (shape: (y:1, x:1))
        >>> m = scp.mean(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.x
        LinearCoord: [float64] cm⁻¹ (size: 5549)
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m = np.ma.mean(dataset, axis=axis, dtype=dtype, keepdims=keepdims)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        cls._data = m.data
        cls._mask = m.mask
        cls.dims = dims

        return cls

    @_from_numpy_method
    def ones(cls, shape, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        ones
            Array of `ones`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> nd = scp.ones(5, units='km')
        >>> nd
        NDDataset: [float64] km (size: 5)
        >>> nd.values
        <Quantity([       1        1        1        1        1], 'kilometer')>
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True])
        >>> nd
        NDDataset: [int64] unitless (size: 5)
        >>> nd.values
        masked_array(data=[  --,        1,        1,        1,   --],
                     mask=[  True,   False,   False,   False,   True],
               fill_value=999999)
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True], units='joule')
        >>> nd
        NDDataset: [int64] J (size: 5)
        >>> nd.values
        <Quantity([  --        1        1        1   --], 'joule')>
        >>> scp.ones((2, 2)).values
        array([[       1,        1],
               [       1,        1]])
        """

        return cls(np.ones(shape), dtype=dtype, **kwargs)

    @_from_numpy_method
    def ones_like(cls, dataset, dtype=None, **kwargs):
        """
        Return |NDDataset| of ones.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        oneslike
            Array of `1` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of the input.
        zeros_like : Return an array of zeros with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> x = scp.NDDataset(x, units='s')
        >>> x
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> scp.ones_like(x, dtype=float, units='J')
        NDDataset: [float64] J (shape: (y:2, x:3))
        """

        cls._data = np.ones_like(dataset, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    def pipe(self, func, *args, **kwargs):
        """
        Apply func(self, *args, **kwargs).

        Parameters
        ----------
        func : function
            Function to apply to the |NDDataset|.
            `*args`, and `**kwargs` are passed into `func`.
            Alternatively a `(callable, data_keyword)` tuple where
            `data_keyword` is a string indicating the keyword of
            `callable` that expects the array object.
        *args
            Positional arguments passed into `func`.
        **kwargs
            Keyword arguments passed into `func`.

        Returns
        -------
        pipe
           The return type of `func`.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        a |NDDataset|.
        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                error_(
                    f"{target} is both the pipe target and a keyword argument. Operation not applied!"
                )
                return self
            kwargs[target] = self
            return func(*args, **kwargs)

        return func(self, *args, **kwargs)

    @_reduce_method
    @_from_numpy_method
    def ptp(cls, dataset, dim=None, keepdims=False):
        """
        Range of values (maximum - minimum) along a dimension.

        The name of the function comes from the acronym for 'peak to peak'.

        Parameters
        ----------
        dim : None or int or dimension name, optional
            Dimension along which to find the peaks.
            If None, the operation is made on the first dimension.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input dataset.

        Returns
        -------
        ptp
            A new dataset holding the result.
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m = np.ma.ptp(dataset, axis=axis, keepdims=keepdims)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        cls._data = m.data
        cls._mask = m.mask
        cls.dims = dims

        return cls

    @_from_numpy_method
    def random(cls, size=None, dtype=None, **kwargs):
        """
        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the “continuous uniform” distribution over the stated interval.
        To sample :math:`\\mathrm{Uniform}[a, b), b > a` multiply the output of random by (b-a) and add a:

            (b - a) * random() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
            Default is None, in which case a single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only float64 and float32 are supported.
            The default value is np.float64.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        random
            Array of random floats of shape size (unless size=None, in which case a single float is returned).
        """
        from numpy.random import default_rng

        rng = default_rng()

        return cls(rng.random(size, dtype), **kwargs)

    @_reduce_method
    @_from_numpy_method
    def std(cls, dataset, dim=None, dtype=None, ddof=0, keepdims=False):
        """
        Compute the standard deviation along the specified axis.

        Returns the standard deviation, a measure of the spread of a distribution,
        of the array elements. The standard deviation is computed for the
        flattened array by default, otherwise over the specified axis.

        Parameters
        ----------
        dataset : array_like
            Calculate the standard deviation of these values.
        dim : None or int or dimension name , optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.
        ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        std
            A new array containing the standard deviation.

        See Also
        --------
        var : Variance values along axis.
        mean : Compute the arithmetic mean along the specified axis.

        Notes
        -----
        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.

        The average squared deviation is normally calculated as
        ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
        the divisor ``N - ddof`` is used instead. In standard statistical
        practice, ``ddof=1`` provides an unbiased estimator of the variance
        of the infinite population. ``ddof=0`` provides a maximum likelihood
        estimate of the variance for normally distributed variables. The
        standard deviation computed in this function is the square root of
        the estimated variance, so even with ``ddof=1``, it will not be an
        unbiased estimate of the standard deviation per se.

        Note that, for complex numbers, `std` takes the absolute
        value before squaring, so that the result is always real and nonnegative.
        For floating-point input, the *std* is computed using the same
        precision the input has. Depending on the input data, this can cause
        the results to be inaccurate, especially for float32 (see example below).
        Specifying a higher-accuracy accumulator using the `dtype` keyword can
        alleviate this issue.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.std(nd)
        <Quantity(0.807972021, 'absorbance')>
        >>> scp.std(nd, keepdims=True)
        NDDataset: [float64] a.u. (shape: (y:1, x:1))
        >>> m = scp.std(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.data
        array([ 0.08521,  0.08543, ...,    0.251,   0.2537])
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m = np.ma.std(dataset, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        cls._data = m.data
        cls._mask = m.mask
        cls.dims = dims

        return cls

    @_reduce_method
    @_from_numpy_method
    def sum(cls, dataset, dim=None, dtype=None, keepdims=False):
        """
        Sum of array elements over a given axis.

        Parameters
        ----------
        dataset : array_like
            Calculate the sum of these values.
        dim : None or int or dimension name , optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        sum
            A new array containing the sum.

        See Also
        --------
        mean : Compute the arithmetic mean along the specified axis.
        trapz : Integration of array values using the composite trapezoidal rule.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.sum(nd)
        <Quantity(381755.783, 'absorbance')>
        >>> scp.sum(nd, keepdims=True)
        NDDataset: [float64] a.u. (shape: (y:1, x:1))
        >>> m = scp.sum(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.data
        array([   100.7,    100.7, ...,       74,    73.98])
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m = np.ma.sum(dataset, axis=axis, dtype=dtype, keepdims=keepdims)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        cls._data = m.data
        cls._mask = m.mask
        cls.dims = dims

        return cls

    @_reduce_method
    @_from_numpy_method
    def var(cls, dataset, dim=None, dtype=None, ddof=0, keepdims=False):
        """
        Compute the variance along the specified axis.

        Returns the variance of the array elements, a measure of the spread of a
        distribution.  The variance is computed for the flattened array by
        default, otherwise over the specified axis.

        Parameters
        ----------
        dataset : array_like
            Array containing numbers whose variance is desired.
        dim : None or int or dimension name , optional
            Dimension or dimensions along which to operate.  By default, flattened input is used.
        dtype : dtype, optional
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.
        ddof : int, optional
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.
        keepdims : bool, optional
            If this is set to True, the dimensions which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.

        Returns
        -------
        var
            A new array containing the standard deviation.

        See Also
        --------
        std : Standard deviation values along axis.
        mean : Compute the arithmetic mean along the specified axis.

        Notes
        -----
        The variance is the average of the squared deviations from the mean,
        i.e.,  ``var = mean(abs(x - x.mean())**2)``.

        The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
        If, however, `ddof` is specified, the divisor ``N - ddof`` is used
        instead.  In standard statistical practice, ``ddof=1`` provides an
        unbiased estimator of the variance of a hypothetical infinite population.
        ``ddof=0`` provides a maximum likelihood estimate of the variance for
        normally distributed variables.

        Note that for complex numbers, the absolute value is taken before
        squaring, so that the result is always real and nonnegative.

        For floating-point input, the variance is computed using the same
        precision the input has.  Depending on the input data, this can cause
        the results to be inaccurate, especially for `float32` (see example
        below).  Specifying a higher-accuracy accumulator using the ``dtype``
        keyword can alleviate this issue.

        Examples
        --------

        >>> nd = scp.read('irdata/nh4y-activation.spg')
        >>> nd
        NDDataset: [float64] a.u. (shape: (y:55, x:5549))
        >>> scp.var(nd)
        <Quantity(0.652818786, 'absorbance')>
        >>> scp.var(nd, keepdims=True)
        NDDataset: [float64] a.u. (shape: (y:1, x:1))
        >>> m = scp.var(nd, dim='y')
        >>> m
        NDDataset: [float64] a.u. (size: 5549)
        >>> m.data
        array([0.007262, 0.007299, ...,  0.06298,  0.06438])
        """

        axis, dim = cls._get_axis(dim, allows_none=True)
        m = np.ma.var(dataset, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

        if np.isscalar(m):
            return Quantity(m, cls.units) if cls.units is not None else m

        dims = _reduce_dims(cls, dim, keepdims)
        cls._data = m.data
        cls._mask = m.mask
        cls.dims = dims

        return cls

    @_from_numpy_method
    def zeros(cls, shape, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
            `numpy.float64`.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        zeros
            Array of zeros.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        ones : Return a new array setting values to 1.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> nd = scp.NDDataset.zeros(6)
        >>> nd
        NDDataset: [float64] unitless (size: 6)
        >>> nd = scp.zeros((5, ))
        >>> nd
        NDDataset: [float64] unitless (size: 5)
        >>> nd.values
        array([       0,        0,        0,        0,        0])
        >>> nd = scp.zeros((5, 10), dtype=np.int, units='absorbance')
        >>> nd
        NDDataset: [int64] a.u. (shape: (y:5, x:10))
        """

        return cls(np.zeros(shape), dtype=dtype, **kwargs)

    @_from_numpy_method
    def zeros_like(cls, dataset, dtype=None, **kwargs):
        """
        Return a |NDDataset| of zeros.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).


        Returns
        -------
        zeorslike
            Array of `fill_value` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of the input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> nd = scp.NDDataset(x, units='s')
        >>> nd
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> nd.values
         <Quantity([[       0        1        2]
         [       3        4        5]], 'second')>
        >>> nd = scp.zeros_like(nd)
        >>> nd
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> nd.values
            <Quantity([[       0        0        0]
         [       0        0        0]], 'second')>
        """

        cls._data = np.zeros_like(dataset, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _check_require_units(self, fname, units):

        if fname in self._require_units.keys():
            requnits = self._require_units[fname]
            if (
                requnits in (DIMENSIONLESS, "radian", "degree")
                or units is None
                or units.dimensionless
            ):
                # this is compatible:
                units = DIMENSIONLESS
            else:
                if requnits == DIMENSIONLESS:
                    s = "DIMENSIONLESS input"
                else:
                    s = f"`{requnits}` units"
                raise DimensionalityError(
                    units, requnits, extra_msg=f"\nFunction `{fname}` requires {s}"
                )
        return units

    @staticmethod
    def _get_operand_and_return_types(fname, objs):
        debug_("Determine Return type from operand types ... ")
        objtypes = []
        for obj in objs:
            type_ = type(obj).__name__
            objtype = type_ if type_ in ORDER.keys() else None
            objtypes.append(objtype)
        if fname not in ["iadd", "isub", "imul", "idiv"]:
            returntype = sorted(objtypes, key=lambda x: ORDER.get(x, 5))[0]
        else:
            returntype = objtypes[0]

        debug_(f"... Return type is {returntype}")

        return objtypes, returntype

    def _check_compatible_operand_dimensionalities(self, fname, objs):
        debug_("Check if operand have compatible units dimensionality ...")
        compatible_units = fname in self._compatible_units
        objdimensionality = OrderedSet()
        for obj in objs:
            # Dimensionalities
            if hasattr(obj, "units"):
                objdimensionality.add(ur.get_dimensionality(obj.units))
                if len(objdimensionality) > 1 and compatible_units:
                    objdimensionality = list(objdimensionality)
                    error = DimensionalityError(
                        *objdimensionality[::-1],
                        extra_msg=f", Units must be compatible for the `{fname}` operator",
                    )
                    raise error  # raise and log error
        debug_("... Dimensionality of the units are compatible")

    def _check_compatible_operand_shapes(self, fname, objs):

        debug_("Checking compatibility of shape of operands ...")

        require_same_shape = fname in self._require_same_shape
        objshapes = OrderedSet()
        # If an object has no shape or size 1 - broadcasting will be applied.
        for obj in objs:
            if hasattr(obj, "shape") and len(np.squeeze(obj).shape) > 0:
                objshapes.add(np.squeeze(obj).shape)
            if len(objshapes) > 1 and require_same_shape:
                # because of the OrderetSet, the case where the two shapes are identical cannot happen here
                # so we need to compare only the last operand with the first dimension. To be broadcastable the
                # second operand must have a size of 1 or a size corresponding to the last dimension of the
                # first operand
                objshapes = list(objshapes)
                last_dim_size = objshapes[0][-1]
                checksize = objshapes[1][0]
                if (
                    len(objshapes[1]) > 1
                ):  # Cannot be broadcasted to the shape of the first operand
                    raise IncompatibleShapeError(*objshapes[::-1])
                elif checksize != last_dim_size and checksize > 1:  # Same problem
                    raise IncompatibleShapeError(*objshapes[::-1])

        debug_("... Shapes are compatibles")

    @staticmethod
    def _check_compatible_operand_coordinates(objs):

        debug_("If needed check that coordinates are compatible ...")

        coordsets = []
        dimss = []
        shapes = OrderedSet()

        # Three solutions for broadcasting:
        # 1. The two array have the same shape and then the coordinates must be compatible in all dimensions.
        # 2. One of the array can be reduced to a scalar.
        # 3. The second array have a shape (1,X) where X is the size of the x coordiantes of the first dataset.
        #    For multidimensional nddatset all the coordinate but the last must be of size 1.

        for obj in objs:

            # We need coord for the last dimension. The last ones in dims.
            # Bt default yhe first in coorset except if data have been transposed.
            coordset = (
                obj._coordset
                if (
                    hasattr(obj, "coordset")
                    and obj.implements() in ["NDDataset"]
                    and obj.size > 1
                )
                else None
            )
            if coordset is not None:
                coordsets.append(coordset)
                shapes.add(obj.shape)
                dimss.append(obj.dims)

            if len(coordsets) > 1 and coordset is None:
                # Probably second operand is a scalar -> Solution 3
                break
            elif len(coordsets) > 1:
                shapeslist = list(shapes)
                if len(shapeslist) == 1:
                    # Solution 1. Same shape
                    try:
                        zipcoordsets = zip(coordsets[0].coords, coordsets[1].coords)
                        for coord0, coord1 in zipcoordsets:
                            assert_coord_almost_equal(
                                coord0, coord1, quantity_only=True, decimals=4
                            )
                    except AssertionError as e:
                        raise CoordinateMismatchError(
                            coord0, coord1, extra_msg=e.args[0]
                        )
                elif shapes[1][-1] == shapes[0][-1]:
                    # Solution 2. coord1 is one dimensional shape (size,) or (1, size)
                    # and size match the last dimension size.
                    if len(shapes[1]) > 1 and shapes[1][0] > 1:
                        # but the second obj is not unidimensional
                        raise IncompatibleShapeError(
                            *objs,
                            extra_msg=" If arrays shapes are differents, the second must be 1D",
                        )

                    coord1 = coordsets[1][dimss[1][-1]]
                    coord0 = coordsets[0][dimss[0][-1]]
                    try:
                        assert_coord_almost_equal(
                            coord0, coord1, quantity_only=True, decimals=4
                        )
                    except AssertionError as e:
                        raise CoordinateMismatchError(
                            coord0, coord1, extra_msg=e.args[0]
                        )
                else:
                    print()

        debug_(
            "... Coords are compatibles"
            if coordsets != {None, None}
            else "... No coordinates to check"
        )

    @staticmethod
    def _is_quaternion_operands(*objs):

        for obj in objs:
            if hasattr(obj, "is_quaternion") and obj.is_quaternion:
                return True
        return False

    def _check_units_and_transform_data(self, fname, inputs):

        debug_("Checking units ...")
        objmagnitudes = []
        objunits = []
        remove_units = fname in self._remove_units
        compatible_units = fname in self._compatible_units

        for obj in inputs:

            if hasattr(obj, "units"):
                required_units = self._check_require_units(fname, obj.units)

                # rescale object to have common units
                if objunits and objunits[0] is not None:
                    if compatible_units:
                        debug_(
                            "Second operand data rescaling to have the same units of the first operand "
                        )
                        obj = obj.to(objunits[0])

                # check validity of units
                units = required_units
                obj = obj.to(required_units) if units != required_units else obj

                # some functions return object without units
                units = None if remove_units else obj.units

                objunits.append(units)
                objmagnitudes.append(obj.magnitude)

            else:
                objunits.append(None)
                if hasattr(obj, "data"):
                    objmagnitudes.append(obj.data)
                else:
                    objmagnitudes.append(obj)

        if len(set(objunits)) == 1:
            # then all operands have the same units or there is only one operand
            # all check have already been done. So nothing to do here
            pass
        elif None in objunits and compatible_units:
            # as this object is either of size 1 or have the same shape as the other one
            # We affect it with the same units if the function require two operand with same units
            i = objunits.index(None)
            j = 0 if i == 1 else 1
            objunits[i] == objunits[j]

        return objmagnitudes, objunits

    @staticmethod
    def _check_masks_and_transform_data(inputs, objtypes, magnitudes):

        debug_("Checking masks ...")
        objtypes = list(objtypes)
        for i, obj in enumerate(inputs):
            mask = obj.mask if hasattr(obj, "mask") and np.any(obj.mask) else NOMASK
            is_masked = np.any(mask != NOMASK)
            try:
                if is_masked and objtypes[i] == "NDDataset":
                    # Apply mask
                    magnitudes[i] = obj._umasked(magnitudes[i], mask)
                elif is_masked:
                    magnitudes[i] = np.ma.masked_array(magnitudes[i], mask=mask)

            except (ValueError, IndexError) as e:
                raise e

        if is_masked:
            debug_(
                "... Some of the data are masked. So magnitudes are transformed accordingly for the op "
                "calculations."
            )
        else:
            debug_("... No mask found")

        return magnitudes, is_masked

    def _perform_magnitude_op(self, f, this, other=None, isufunc=False):

        # perform operation on magnitudes
        debug_("Perform operation on magnitude only ...")

        fname = f.__name__

        # If one of the input is hypercomplex, this will demand a special treatment
        is_quaternion = self._is_quaternion_operands(this, other)
        quaternion_aware = fname in self._quaternion_aware

        if isufunc:

            with catch_warnings(record=True) as ws:

                # try to apply the ufunc
                if fname == "log1p":
                    fname = "log"
                    this = this + 1.0
                if fname in ["arccos", "arcsin", "arctanh"]:
                    if np.any(np.abs(this) > 1):
                        this = this.astype(np.complex128)
                elif fname in ["sqrt"]:
                    if np.any(this < 0):
                        this = this.astype(np.complex128)

                if fname == "sqrt":  # do not work with masked array
                    data = this ** (1.0 / 2.0)
                elif fname == "cbrt":
                    data = np.sign(this) * np.abs(this) ** (1.0 / 3.0)
                else:
                    data = getattr(np, fname)(this, other)

                # if a warning occurs, let handle it with complex numbers or return an exception:
                if ws and "invalid value encountered in " in ws[-1].message.args[0]:
                    ws = []  # clear
                    # this can happen with some function that do not work on some real values such as log(-1)
                    # then try to use complex
                    data = getattr(np, fname)(
                        this.astype(np.complex128), other
                    )  # data = getattr(np.emath, fname)(d, *args)
                    if ws:
                        raise ValueError(ws[-1].message.args[0])
                elif ws and "overflow encountered" in ws[-1].message.args[0]:
                    warning_(ws[-1].message.args[0])
                elif ws:
                    raise ValueError(ws[-1].message.args[0])

            # TODO: check the complex nature of the result to return it

        else:
            # make a simple operation
            try:
                if not is_quaternion:
                    data = f(this, other) if other is not None else f(this)
                elif quaternion_aware and all(
                    (m.dtype not in TYPE_COMPLEX for m in [this, other])
                ):
                    data = f(this, other) if other is not None else f(this)
                else:
                    # in this case we will work on both complex separately
                    dr, di = quat_as_complex_array(this)
                    datar = f(dr, other) if other is not None else f(dr)
                    datai = f(di, other) if other is not None else f(di)
                    data = as_quaternion(datar, datai)

            except TypeError as e:
                if (
                    hasattr(this, "dtype")
                    and str(this.dtype).startswith("datetime64")
                    and f.__name__ == "isub"
                ):
                    data = (
                        operator.sub(this, other)
                        if other is not None
                        else operator.sub(this)
                    )
                else:
                    raise ArithmeticError(e.args[0])

        return data

    def _perform_units_op(self, f, objunits, objtypes):

        debug_("Performs calculations on the units ...")

        compatible_units = f.__name__ in self._compatible_units

        if set(objunits) == {None}:
            return None

        unit0 = objunits[0]
        unit1 = objunits[1] if len(objunits) > 1 else None

        if (
            unit1 is None
            and f.__name__ in ["add", "iadd", "isub", "sub", "subtract"]
            and len(objtypes) > 1
            and objtypes[1] is None
        ):  # probably other is a scalar - for add and sub we admit it is the same units!
            unit1 = unit0

        # Create two random quantities which will be used for calculation on the units. We do calculation on
        # Quantities in order to avoid calculation with the whole data arrays.
        try:
            q0 = (rand() + 0.1) * unit0 if unit0 is not None else ur("")
            if len(objunits) > 1:
                q1 = (rand() + 0.1) * unit1 if unit1 is not None else ur("")
            else:
                q1 = None
        except Exception as e:
            raise e

        # Some functions are not handled by pint regarding units, try to solve this here
        f_u = f

        if compatible_units:
            f_u = operator.sub  # take a similar binary function handled by pint

        try:
            res = f_u(q0, q1) if q1 is not None else f_u(q0)

        except Exception as e:
            raise e

        units = res.units if hasattr(res, "units") else None

        debug_(f"Returned units is {str(units)}")

        return units

    def _op(self, f, inputs, isufunc=False):

        fname = f.__name__
        debug_(f"Apply a function {fname} of operands: {inputs}")

        # Achieve an operation f on the objs
        inputs = list(inputs)  # Work with a list of objs not tuples

        # By default the type of the returned result is set regarding the first obj
        # in inputs.except for some ufuncs that can return numpy arrays or masked
        # numpy arrays. But sometimes we have something such as 2 * nd where nd is a
        # NDDataset: In this case we expect a dataset.
        # For binary function, we must also determine if the function needs object with
        # compatible units. If the object are not compatible then we raise an error.
        # The following  methods Take the objects out of the input list and get their
        # types, dimensionality, magnitude and units. Additionally determine if we need
        # to use operation on masked arrays and/or on quaternion.

        # is_dt64 = lambda o: o.is_dt64 if hasattr(o, "is_dt64") else False

        # Dimensionality needs often to be the same
        self._check_compatible_operand_dimensionalities(fname, inputs)

        # Shapes must be compatible for most of the operations
        self._check_compatible_operand_shapes(fname, inputs)

        # Dimension coordinates must be compatibles
        self._check_compatible_operand_coordinates(inputs)

        # Get the input and return types
        objtypes, returntype = self._get_operand_and_return_types(fname, inputs)

        # Checks units compatibility and eventually rescale data to have same units
        # then returns thes magnitudes and the units suitable for further operation
        magnitudes, units = self._check_units_and_transform_data(fname, inputs)

        # Mask
        magnitudes, is_masked = self._check_masks_and_transform_data(
            inputs, objtypes, magnitudes
        )

        # Final calculations
        data = self._perform_magnitude_op(f, *magnitudes, isufunc=isufunc)

        units = self._perform_units_op(f, units, objtypes)

        data, mask = (
            (data._data, data._mask)
            if isinstance(data, np.ma.MaskedArray)
            else (data, NOMASK)
        )

        # if self._check_if_is_td64(data):
        #     data, units = self._data_and_units_from_td64(data)
        #     dtype = np.dtype("float")
        # else:
        #     dtype = None

        # if returntype in order.keys() and objtypes[0] != returntype:
        #     # TODO: TEST, I AM NOT SURE THIS IS A POSSIBLE CASE IN SPECTROCHEMPY
        #     datas.reverse()
        #
        #     if fname in ["truediv", "divide", "true_divide"]:
        #         fname = "multiply"
        #         datas[0][0] = np.reciprocal(datas[0][0])
        #     elif fname in ["sub", "subtract"]:
        #         fname = "add"
        #         datas[0][0] = np.negative(datas[0][0])
        #     else:  # fname in ["mul", "multiply", "add"]
        #         pass  # Other cases ?  # raise NotImplementedError()  # or let it unchanged

        # --- returns -----

        # return calculated data, units and mask
        return data, units, mask, returntype

    @staticmethod
    def _unary_op(f):
        @functools.wraps(f)
        def func(self):
            fname = f.__name__
            if hasattr(self, "history"):
                history = f"Unary operation {fname} applied"
            else:
                history = None

            data, units, mask, returntype = self._op(f, [self])
            return self._op_result(data, units, mask, history, returntype)

        return func

    @staticmethod
    def _check_order(fname, inputs):
        objtypes = []
        returntype = None
        for i, obj in enumerate(inputs):
            # type
            objtype = type(obj).__name__
            objtypes.append(objtype)
            if objtype == "NDDataset":
                returntype = "NDDataset"
            elif objtype == "Coord" and returntype != "NDDataset":
                returntype = "Coord"
            elif objtype == "LinearCoord" and returntype != "NDDataset":
                returntype = "LinearCoord"
            else:
                # only the three above type have math capabilities in spectrochempy.
                pass

        # it may be necessary to change the object order regarding the types
        if (
            returntype in ["NDDataset", "Coord", "LinearCoord"]
            and objtypes[0] != returntype
        ):

            inputs.reverse()
            objtypes.reverse()

            if fname in ["mul", "add"]:
                pass
            elif fname in ["truediv", "divide", "true_divide"]:
                fname = "mul"
                inputs[0] = np.reciprocal(inputs[0])
            elif fname in ["sub", "subtract"]:
                fname = "add"
                inputs[0] = np.negative(inputs[0])
            elif fname in ["pow"]:
                fname = "exp"
                inputs[0] *= np.log(inputs[1])
                inputs = inputs[:1]
            else:
                raise NotImplementedError()

        if fname in ["exp"]:
            f = getattr(np, fname)
        else:
            f = getattr(operator, fname)
        return f, inputs

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            fname = f.__name__
            if not reflexive:
                objs = [self, other]
            else:
                objs = [other, self]
            fm, objs = self._check_order(
                fname, objs
            )  # TODO: seems that this done in _ops ???

            if hasattr(self, "history"):
                history = f"Binary operation {fm.__name__} with `{_get_name(objs[-1])}` has been performed"
            else:
                history = None

            data, units, mask, returntype = self._op(fm, objs)
            new = self._op_result(data, units, mask, history, returntype)
            return new

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            fname = f.__name__
            if hasattr(self, "history"):
                history = f"Inplace binary op: {fname}  with `{_get_name(other)}` "
            else:
                history = None
            objs = [self, other]
            fm, objs = self._check_order(fname, objs)

            dt64 = self.dtype.kind == "M"
            if dt64 == "M":
                # inplace binary does not work yet for datetime64 object type.
                # take the regular binary op instead
                fm = _get_op(fname[1:])  # remove the i in the operator name
            else:
                fm = f
            data, units, mask, returntype = self._op(fm, objs)
            self = self._op_result(
                data, units, mask, history, returntype, inplace=not dt64
            )
            return self

        return func

    def _op_result(
        self, data, units=None, mask=None, history=None, returntype=None, inplace=False
    ):
        # make a new NDArray resulting of some operation

        new = self.copy() if not inplace else self

        if returntype == "NDDataset":  # and not new.implements("NDDataset"):
            from spectrochempy.core.dataset.nddataset import NDDataset

            new = NDDataset(new)

        if returntype in ["LinearCoord", "Coord"]:
            from spectrochempy.core.dataset.coord import Coord

            new = Coord(new)

        # set the new units
        new._units = units

        # set the data
        new.data = cpy.deepcopy(data)
        if returntype == "LinearCoord":
            new.linear = True

        # update the other attributes
        if mask is not None and np.any(mask != NOMASK):
            new._mask = cpy.copy(mask)
        if history is not None and hasattr(new, "history"):
            new.history = history.strip()

        # case when we want to return a simple masked ndarray
        if returntype == "masked_array":
            return new.masked_data

        return new


class _ufunc:
    def __init__(self, name):
        self.name = name
        self.ufunc = getattr(np, name)

    def __call__(self, *args, **kwargs):
        return self.ufunc(*args, **kwargs)

    @property
    def __doc__(self):
        doc = f"""
            {_unary_ufuncs()[self.name].split('->')[-1].strip()}

            Wrapper of the numpy.ufunc function ``np.{self.name}(dataset)``.

            Parameters
            ----------
            dataset : array-like
                Object to pass to the numpy function.

            Returns
            -------
            out
                |NDDataset|

            See Also
            --------
            numpy.{self.name} : Corresponding numpy Ufunc.

            Notes
            -----
            Numpy Ufuncs referenced in our documentation can be directly applied to |NDDataset| or |Coord| type
            of SpectrochemPy objects.
            Most of these Ufuncs, however, instead of returning a numpy array, will return the same type of object.

            Examples
            --------

            >>> ds = scp.read('wodger.spg')
            >>> ds_transformed = scp.{self.name}(ds)

            Numpy Ufuncs can also be transparently applied directly to SpectroChemPy object

            >>> ds_transformed = np.{self.name}(ds)
            """
        return doc


thismodule = sys.modules[__name__]


def _set_ufuncs(cls):
    for func in _unary_ufuncs():
        # setattr(cls, func, _ufunc(func))
        setattr(thismodule, func, _ufunc(func))
        thismodule.__all__ += [func]


# ------------------------------------------------------------------
# module functions
# ------------------------------------------------------------------
# make some NDMath operation accessible from the spectrochempy API


# make some API functions
api_funcs = [  # creation functions
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty",
    "zeros",
    "ones",
    "full",
    "eye",
    "identity",
    "random",
    "linspace",
    "arange",
    "logspace",
    "geomspace",
    "fromfunction",
    "fromiter",  #
    "diagonal",
    "diag",
    "sum",
    "average",
    "mean",
    "std",
    "var",
    "amax",
    "amin",
    "min",
    "max",
    "argmin",
    "argmax",
    "cumsum",
    "coordmin",
    "coordmax",
    "clip",
    "ptp",
    "pipe",
    "abs",
    "conjugate",
    "absolute",
    "conj",
    "all",
    "any",
]

for funcname in api_funcs:
    setattr(thismodule, funcname, getattr(NDMath, funcname))
    thismodule.__all__.append(funcname)

api_manipulation_funcs = [  # manipulation routines
    "squeeze",
    "expand_dims",
    "swapdims",
    "swapaxes",
    "transpose",
    "atleast_1d",
    "atleast_2d",
]

for funcname in api_manipulation_funcs:
    setattr(thismodule, funcname, getattr(NDManipulation, funcname))
    thismodule.__all__.append(funcname)

# ------------------------------------------------------------------
# ARITHMETIC ON NDArray
# ------------------------------------------------------------------

# unary operators
UNARY_OPS = ["neg", "pos", "abs"]

# binary operators
CMP_BINARY_OPS = ["lt", "le", "ge", "gt"]
NUM_BINARY_OPS = ["add", "sub", "and", "xor", "or", "mul", "truediv", "floordiv", "pow"]


def _op_str(name):
    return f"__{name}__"


def _get_op(name):
    return getattr(operator, _op_str(name))


def _set_operators(cls, priority=50):
    cls.__array_priority__ = priority

    # unary ops
    for name in UNARY_OPS:
        setattr(cls, _op_str(name), cls._unary_op(_get_op(name)))

    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, _op_str(name), cls._binary_op(_get_op(name)))

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, _op_str("r" + name), cls._binary_op(_get_op(name), reflexive=True))

        setattr(cls, _op_str("i" + name), cls._inplace_binary_op(_get_op("i" + name)))


if __name__ == "__main__":
    pass
