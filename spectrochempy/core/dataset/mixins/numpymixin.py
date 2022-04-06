from numbers import Number

import numpy as np

from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.mixins.mixinutils import (
    _REGISTERED_FUNCTIONS,
    _op_result,
)

# see https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins
# .NDArrayUfuncMixin.html#numpy.lib.mixins.NDArrayUfuncMixin


class NDArrayUfuncMixin(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Provide numpy UFunc handling to Coord and NDDataset objects.
    """

    # handlers for types, ufuns and functions
    _HANDLED_TYPES = ()
    _HANDLED_UFUNCS = ()

    # Prioritize our operations over those of numpy.ndarray (priority=0)
    __array_priority__ = 50

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        fname = ufunc.__name__
        if fname not in self._HANDLED_UFUNCS:
            raise NotImplementedError(
                f"ufunc `{fname}` is not implemented on "
                f"`{type(self).__name__}` objects"
            )

        if method != "__call__":
            raise NotImplementedError(
                f"`{method}` method for ufunc `{fname}` is not implemented on "
                f"`{type(self).__name__}` objects"
            )

        out = kwargs.pop("out", ())  # We use pop to remove out
        # because if dealing with quantity,
        # pint seems not working
        # with out being a quantity!:
        # operations such as np.add(x,y,x) fail
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES
            if not isinstance(x, self._HANDLED_TYPES + (NDArrayUfuncMixin, NDArray)):
                raise NotImplementedError(
                    f"ufunc `{fname}` is not implemented on "
                    f"`{type(x).__name__}` objects"
                )

                # Defer to the implementation of the ufunc on unwrapped values : ndarray or
        # quantity. For the latter, we use the computation capabilities of pint.
        inputs = list(
            x.value if isinstance(x, (NDArrayUfuncMixin, NDArray)) else x
            for x in inputs
        )

        # Allow adding or subtracting a unitless scalar to a Quantity array.
        # This is quite strict: ufunc must be 'add' or 'subtract' only.
        if fname in ["add", "subtract"]:
            if hasattr(inputs[0], "units") and isinstance(inputs[1], Number):
                inputs[1] *= inputs[0].units
            elif hasattr(inputs[1], "units") and isinstance(inputs[0], Number):
                inputs[0] *= inputs[1].units

        # Do calculations
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if not out:
            out = (None,) * len(result)

        history = f"Ufunc `{fname}` applied."
        if type(result) is tuple:
            # multiple return value
            if len(out) != len(result):
                raise ValueError(
                    "`out` must have the same length as the number of "
                    "ufunc's  output"
                )
            return tuple(
                _op_result(self, fname, x, history, out=o) for x, o in zip(result, out)
            )
        else:
            return _op_result(self, fname, result, history, out=out[0])


class NDArrayFunctionMixin:

    _HANDLED_FUNCTIONS = ()
    #     ['alen', 'all', 'alltrue', 'amax', 'amin', 'any', 'argmax', 'argmin',
    #     'argpartition', 'argsort', 'around', 'choose', 'clip', 'compress', 'cumprod',
    #     'cumproduct', 'cumsum', 'diagonal', 'mean', 'ndim', 'nonzero', 'partition',
    #     'prod', 'product', 'ptp', 'put', 'ravel', 'repeat', 'reshape', 'resize',
    #     'round_', 'searchsorted', 'shape', 'size', 'sometrue', 'sort', 'squeeze', 'std',
    #     'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var', ]

    __array_priority__ = 45

    def __array_function__(self, func, types, args, kwargs):

        fname = func.__name__

        if func not in _REGISTERED_FUNCTIONS or fname not in self._HANDLED_FUNCTIONS:
            raise NotImplementedError(
                f"function `{fname}` is not implemented on "
                f"`{type(self).__name__}` objects"
            )

        if not all(issubclass(t, NDArrayFunctionMixin) for t in types):
            return NotImplemented

        return _REGISTERED_FUNCTIONS[func](*args, **kwargs)
