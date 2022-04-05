# -*- coding: utf-8 -*-

# ======================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
# ======================================================================================

"""
This module implements the class |Coord|.
"""

__all__ = ["Coord"]

from numbers import Number

import numpy as np
import traitlets as tr

from spectrochempy.core import warning_
from spectrochempy.core.common.compare import is_number, is_sequence
from spectrochempy.core.common.exceptions import (
    deprecated,
)
from spectrochempy.core.dataset.basearrays.ndlabeledarray import (
    NDLabeledArray,
    _docstring,
)
from spectrochempy.core.dataset.mixins.numpymixins import NDArrayUfuncMixin
from spectrochempy.core.dataset.mixins.numpymixins import NDArrayFunctionMixin
from spectrochempy.core.dataset.mixins.functionbasemixin import NDArrayFunctionBaseMixin
from spectrochempy.core.units import (
    Quantity,
    encode_quantity,
    ur,
    set_nmr_context,
    get_units,
)
from spectrochempy.utils.misc import spacings
from spectrochempy.utils.optional import import_optional_dependency


# ======================================================================================
# Coord
# ======================================================================================
class Coord(
    NDLabeledArray, NDArrayUfuncMixin, NDArrayFunctionMixin, NDArrayFunctionBaseMixin
):
    __doc__ = _docstring.dedent(
        """
    Explicit coordinates for a dataset along a given axis.

    The coordinates of a |NDDataset| can be created using the |Coord|
    object.
    This is a single dimension array with either numerical (float)
    values or labels (str, `Datetime` objects, or any other kind of objects) to
    represent the coordinates. Only a one numerical axis can be defined,
    but labels can be multiple.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    decimals : int, optional, default: 3
        The number of rounding decimals for coordinates values.
    linear : bool, optional
        If set to True, the coordinate is linearized (equal spacings).
    larmor : float
        For NMR context, specification of the larmor frequency allows conversion of
        ppm units to frequency.
    %(NDLabeledArray.other_parameters)s

    See Also
    --------
    NDDataset : Main SpectroChemPy object: an array with masks, units and coordinates.
    """
    )
    # """
    #     Examples
    #     --------
    #
    #     We create a numpy |ndarray| and use it as the numerical `data`
    #     axis of our new |Coord| object.
    #     >>> c0 = scp.Coord.arange(1., 12., 2., title='frequency', units='Hz')
    #     >>> c0
    #     Coord: [float64] Hz (size: 6)
    #
    #     We can take a series of str to create a non-numerical but labelled
    #     axis :
    #     >>> tarr = list('abcdef')
    #     >>> tarr
    #     ['a', 'b', 'c', 'd', 'e', 'f']
    #
    #     >>> c1 = scp.Coord(labels=tarr, title='mylabels')
    #     >>> c1
    #     Coord: [labels] [  a   b   c   d   e   f] (size: 6)
    #     """
    _linear = tr.Bool(False)
    _decimals = tr.Int(3)

    # specific to NMR
    _larmor = tr.Float(allow_none=True)

    _parent_dim = tr.Unicode(allow_none=True)

    # ----------------------------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------------------------

    # Handlers for arithmetic operations on Coord objects
    _HANDLED_TYPES = (Quantity, np.ndarray, Number, list)

    # Ufunc - Functions that operates element by element on whole arrays.
    _HANDLED_UFUNCS = (
        "add",
        "subtract",
        "multiply",
        "true_divide",
        "reciprocal",
        "log",
        "log2",
        "log10",
        "exp",
        "exp2",
        "sqrt",
    )  # TODO: tune this list by adding useful ufuncs for coordinates

    _HANDLED_FUNCTIONS = ()

    def __init__(self, data=None, **kwargs):

        # specific case of NMR (initialize unit context NMR)
        if "larmor" in kwargs:
            self.larmor = kwargs.pop("larmor")

        super().__init__(data=data, **kwargs)

        # Linearization with rounding to the number of given decimals
        # If linear=True is passed in parameters
        decimals = kwargs.pop("decimals", 3)

        if kwargs.get("linear", False):
            self.linearize(decimals)

    def __getattr__(self, item):
        if item == "default":
            # this is in case default is called while it is not a cordset.
            return self
        raise AttributeError(f"`Coord` object has no attribute `{item}`")

    # ----------------------------------------------------------------------------------
    # Private properties and methods
    # ----------------------------------------------------------------------------------
    def _cstr(self, **kwargs):
        out = super()._cstr(header="  coordinates: ... \n", **kwargs)
        return out

    def __larmor_default(self):
        return None

    def _to_xarray(self):
        # to be used during conversion of NDarray-like to Xarray object

        xr = import_optional_dependency("xarray")

        var = xr.Variable(dims=self.name, data=np.array(self.data))
        # dtype=np.float64))

        var.attrs["name"] = self.name
        var.attrs["pint_units"] = str(self.units)
        # We change name of attr `units`
        # To not override a builtin xarray units attribute.
        var.attrs["title"] = self.title
        var.attrs["roi"] = self.roi
        for k, v in self.meta.items():
            var.attrs[f"meta_{k}"] = v

        var.attrs = encode_quantity(var.attrs)

        coordinates = {self.name: var}

        # auxiliary coordinates
        if self.is_labeled:
            for level in range(self.labels.shape[-1]):
                label = self.labels[:, level]
                label = list(map(str, label.tolist()))
                label = xr.Variable(dims=self.name, data=label)
                coordinates[f"{self.name}_labels_{level}"] = label

        return coordinates  # TODO: add multiple coordinates

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @property
    @_docstring.dedent
    def data(self):
        """%(data)s"""
        if self.has_data and self.dtype.kind not in "M":
            return np.around(self._data, self.decimals)
        return super().data

    @data.setter
    def data(self, data):
        self._set_data(data)

    @property
    def coordinates(self):
        """
        Alias of data.
        """
        return self.data

    @property
    def decimals(self):
        """
        Return the number of decimals set for rounding coordinate values.
        """
        return self._decimals

    @property
    def larmor(self):
        """
        Return larmor frequency in NMR spectroscopy context.
        """
        return self._larmor

    @larmor.setter
    def larmor(self, val):
        self._larmor = val

    @property
    def is_descendant(self):
        """
        Return whether the coordinate has a descendant order.
        """
        return (self.data[-1] - self.data[0]) < 0

    def linearize(self, decimals=None):
        """
        Linearize the coordinates data.

        Make coordinates with a equally distributed spacing, when possible, i.e.,
        when the spacings are not two different when rounded to the number of
        decimals passed in parameters.

        Parameters
        ----------
        decimals :  Int, optional, default=3
            The number of rounding decimals for coordinates values.
        """
        # TODO: write doc examples

        if not self.has_data or self.data.size < 3:
            return

        data = self.data.squeeze()

        if decimals is None:
            decimals = 3
        self._decimals = decimals

        inc = np.diff(data)
        variation = (inc.max() - inc.min()) / 2.0
        if variation < 10 ** -decimals:
            # we set the number with their full precision
            # rounding will be made if necessary when reading the data property
            self._data = np.linspace(data[0], data[-1], data.size)
            self._linear = True
        else:
            self._linear = False

    def loc2index(self, loc, *, units=None):
        """
        Return the index corresponding to a given location.

        Parameters
        ----------
        loc : int, float, label or str
            Value corresponding to a given location on the coordinate's axis.
        units : Units
            Units of the location.

        Returns
        -------
        int
            The corresponding index.
        """
        # TODO: write doc examples

        # check units compatibility
        if (
            units is not None
            and (is_number(loc) or is_sequence(loc))
            and units != self.units
        ):
            raise ValueError(
                f"Units of the location {loc} {units} "
                f"are not compatible with those of this array:"
                f"{self.units}"
            )

        if self.is_empty and not self.is_labeled:

            raise IndexError(f"Could not find this location {loc} on an empty array")

        data = self.data

        if is_number(loc):
            # get the index of a given values
            error = None
            if np.all(loc > data.max()) or np.all(loc < data.min()):
                warning_(
                    f"This coordinate ({loc}) is outside the axis limits "
                    f"({data.min()}-{data.max()}).\nThe closest limit index is "
                    f"returned"
                )
                error = "out_of_limits"
            index = (np.abs(data - loc)).argmin()
            # TODO: add some precision to this result
            if not error:
                return index
            return index, error

        if is_sequence(loc):
            # TODO: is there a simpler way to do this with numpy functions
            index = []
            for lo_ in loc:
                index.append(
                    (np.abs(data - lo_)).argmin()
                )  # TODO: add some precision to this result
            return index

        if self.is_labeled:

            # TODO: look in all row of labels
            labels = self._labels
            indexes = np.argwhere(labels == loc).flatten()
            if indexes.size > 0:
                return indexes[0]
            raise IndexError(f"Could not find this label: {loc}")

        if isinstance(loc, np.datetime64):
            # not implemented yet
            raise NotImplementedError(
                "datetime as location is not yet implemented"
            )  # TODO: date!

        raise IndexError(f"Could not find this location: {loc}")

        idx = self._loc2index(loc)
        if isinstance(idx, tuple):
            idx, _ = idx
            # warnings.warn(warning)
        return idx

    @property
    def reversed(self):
        """
        Whether the axis is reversed.
        """
        if self.units == "ppm":
            return True
        if self.units == "1 / centimeter" and "raman" not in self.title.lower():
            return True
        return False

        # Return a correct result only if the data are sorted
        # return  # bool(
        # self.data[0] > self.data[-1])

    @property
    def spacing(self):
        """
        Return coordinates spacing.

        It will be a scalar if the coordinates are uniformly spaced,
        else an array of the different spacings.
        """
        if self.has_data:
            return spacings(self.data) * self.units
        return None

    @_docstring.dedent
    def to(self, other, inplace=False, force=False, title=None):
        """%(to)s"""

        units = get_units(other)
        if self.larmor is None:
            # no change
            return super().to(units, inplace, force)
        else:
            # set context
            set_nmr_context(self.larmor)
            with ur.context("nmr"):
                return super().to(units, inplace, force)


# ======================================================================================
# LinearCoord (Deprecated)
# ======================================================================================
class LinearCoord(Coord):
    @deprecated(
        kind="object",
        replace="Coord",
        extra_msg="with the `linear` " "keyword argument.",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ======================================================================================
if __name__ == "__main__":
    """ """
