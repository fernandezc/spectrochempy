# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Common utilities for mixin classes
"""

import functools
import inspect
from copy import copy as copy_

from spectrochempy.core.dataset.basearrays.ndarray import NDArray

_REGISTERED_FUNCTIONS = {}
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


def _op_result(self, fname, value, history=None, out=None):
    # make a new NDArray resulting of some operations fname

    new = self.copy() if out is None else out

    # set the new units
    if hasattr(value, "units"):
        new._units = value.units
        value = value.magnitude

    if hasattr(value, "mask"):
        self._mask = value.mask
        value = value.data

    # set the data
    new._data = value

    # eventually add other attributes
    if isinstance(new, NDArray):

        if hasattr(new, "history") and history is not None:
            # set history string
            new.history = history.strip()

        # Eventually make a new title depending on the operation
        if fname in _remove_title:
            new.title = f"<{fname}>"
        elif fname not in _keep_title and isinstance(new, NDArray):
            if hasattr(new, "title") and new.title is not None:
                new.title = f"{fname}({new.title})"
            else:
                new.title = f"{fname}(value)"

    return new


def _register_implementation(numpy_function):
    """
    Register an __array_function__ implementation for NDArrayFunctionMixin
    subclassed objects.
    """

    def decorator(func):
        _REGISTERED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


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
                elif cls.__name__ in ["NDDataset", "Coord"]:
                    # issubclass(cls, (NDDataset, Coord)):
                    klass = cls
                else:
                    # nor an instance or a class
                    # Probably a call from the API !
                    # scp.method(...)
                    # We return a NDDataset class constructor
                    from spectrochempy.core.dataset.nddataset import NDDataset

                    klass = NDDataset
            else:
                # determine the input object
                if instance is not None:
                    # the call is made as an attributes of the instance
                    # instance.method(...)
                    new = instance.copy()
                    args.insert(0, new)
                else:
                    dataset = copy_(args[0])
                    try:
                        # call as a classmethod
                        # class.method(dataset, ...)
                        new = cls(dataset)
                    except TypeError:
                        if issubclass(cls, NDArray):
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
