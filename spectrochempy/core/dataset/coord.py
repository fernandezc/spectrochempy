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

import copy as cpy
import textwrap
import uuid
import warnings

import numpy as np
import traitlets as tr

from spectrochempy.core import debug_, error_, print_, warning_
from spectrochempy.core.common.compare import is_number, is_sequence
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME, INPLACE, NOMASK
from spectrochempy.core.common.exceptions import (
    InvalidCoordinatesSizeError,
    InvalidCoordinatesTypeError,
    InvalidDimensionNameError,
    ShapeError,
    SpectroChemPyWarning,
    deprecated,
)
from spectrochempy.core.common.print import colored_output, convert_to_html
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndlabeledarray import NDLabeledArray
from spectrochempy.core.dataset.ndmath import NDMath
from spectrochempy.core.units import Quantity, encode_quantity, ur
from spectrochempy.utils.misc import spacings
from spectrochempy.utils.optional import import_optional_dependency
from spectrochempy.utils.traits import Range


# ======================================================================================
# Coord
# ======================================================================================
class Coord(NDMath, NDLabeledArray):
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
    data : ndarray, tuple or list
        The actual data array contained in the |Coord| object.
        The given array (with a single dimension) can be a list,
        a tuple, a |ndarray|, or a |ndarray|-like object.
        If an object is passed that contains labels, or units,
        these elements will be used to accordingly set those of the
        created object.
        If possible, the provided data will not be copied for `data` input,
        but will be passed by reference, so you should make a copy the
        `data` before passing it in the object constructor if that's the
        desired behavior or set the `copy` argument to True.
    **kwargs
        Optional keywords parameters. See other parameters.

    Other Parameters
    ----------------
    copy : bool, optional
        Perform a copy of the passed object. Default is False.
    dtype : str or dtype, optional, default=np.float64
        If specified, the data will be cast to this dtype, else the
        type of the data will be used.
    name : str, optional
        A unique and user-friendly name for this object. If not given, it will be
        automaticlly
        attributed.
    labels : array of objects, optional
        Labels for the `data`. labels can be used only for 1D-datasets.
        The labels array may have an additional dimension, meaning
        several series of labels for the same data.
        The given array can be a list, a tuple, a |ndarray|,
        a ndarray-like, a |NDArray| or any subclass of
        |NDArray|.
    linear : bool, optional
        If set to True, the coordinate is linearized (equal spacings)
    title : str, optional
        The title of the dimension. It will later be used for instance
        for labelling plots of the data.
        It is optional but recommanded to give a title to each ndarray.
    units : |Unit| instance or str, optional
        Units of the data. If data is a |Quantity| then `units` is set
        to the unit of the `data`; if a unit is also
        explicitly provided an error is raised. Handling of units use
        the `pint <https://pint.readthedocs.org/>`_
        package.

    See Also
    --------
    NDDataset : Main SpectroChemPy object: an array with masks, units and
                coordinates.

    Examples
    --------

    We first import the object from the api :
    >>> from spectrochempy import Coord

    We then create a numpy |ndarray| and use it as the numerical `data`
    axis of our new |Coord| object.
    >>> c0 = Coord.arange(1., 12., 2., title='frequency', units='Hz')
    >>> c0
    Coord: [float64] Hz (size: 6)

    We can take a series of str to create a non-numerical but labelled
    axis :
    >>> tarr = list('abcdef')
    >>> tarr
    ['a', 'b', 'c', 'd', 'e', 'f']

    >>> c1 = Coord(labels=tarr, title='mylabels')
    >>> c1
    Coord: [labels] [  a   b   c   d   e   f] (size: 6)
    """

    _parent_dim = tr.Unicode(allow_none=True)

    # ----------------------------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):

        super().__init__(data=data, **kwargs)

        if len(self.shape) > 1:
            raise ShapeError(
                self.shape,
                message="Only one 1D arrays can be used to " "define coordinates",
            )

        # Linearization
        # If linear=True is passed in parameters

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __getitem__(self, items, **kwargs):

        if isinstance(items, list):
            # Special case of fancy indexing
            items = (items,)

        # choose, if we keep the same or create new object
        inplace = False
        if isinstance(items, tuple) and items[-1] == INPLACE:
            items = items[:-1]
            inplace = True

        # Eventually get a better representation of the indexes
        try:
            keys = self._make_index(items)
        except IndexError:
            # maybe it is a metadata
            if items in self.meta.keys():
                return self.meta[items]
            raise KeyError

        # init returned object
        new = self if inplace else self.copy()

        # slicing by index of all internal array
        if new.data is not None:
            udata = new.data[keys]
            new._data = np.asarray(udata)

        if self.is_labeled:
            # case only of 1D dataset such as Coord
            new._labels = np.array(self._labels[keys])

        if new.is_empty:
            error_(
                f"Empty array of shape {new._data.shape} resulted from slicing.\n"
                f"Check the indexes and make sure to use floats for location slicing"
            )
            new = None

        # we need to keep the names when copying coordinates to avoid later
        # problems
        new.name = self.name
        return new

    # ----------------------------------------------------------------------------------
    # Private properties and methods
    # ----------------------------------------------------------------------------------
    def _cstr2(self, header="  coordinates: ... \n", print_size=True, **kwargs):

        indent = kwargs.get("indent", 0)

        out = ""
        if not self.is_empty and print_size:
            out += f"{self._str_shape().rstrip()}\n"
        out += f"        title: {self.title}\n" if self.title else ""
        if self.has_data:
            out += f"{self._str_value(header=header)}\n"
        elif self.is_empty and not self.is_labeled:
            out += header.replace("...", "\0Undefined\0")

        if self.is_labeled:
            header = "       labels: ... \n"
            text = str(self.labels.T).strip()
            if "\n" not in text:  # single line!
                out += header.replace("...", f"\0\0{text}\0\0")
            else:
                out += header
                out += f"\0\0{textwrap.indent(text.strip(),' ' * 9)}\0\0"

        if out[-1] == "\n":
            out = out[:-1]

        if indent:
            out = f"{textwrap.indent(out, ' ' * indent)}"

        first_indent = kwargs.get("first_indent", 0)
        if first_indent < indent:
            out = out[indent - first_indent :]

        if not self._html_output:
            return out  # colored_output(out)
        return out

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
    def default(self):
        # this is in case default is called on a coord, while it is a CoordSet property
        return self

    @property
    def is_descendant(self):
        return (self.data[-1] - self.data[0]) < 0

    def linearized(self, rounding=6):
        # TODO: Continue
        if not self.has_data():
            return None

        data = self._data.squeeze()

        # try to find an increment
        if data.size > 1:
            inc = np.diff(data)
            variation = (inc.max() - inc.min()) / data.ptp()
            if variation < 1.0e-5:
                self._increment = (
                    data.ptp() / (data.size - 1) * np.sign(inc[0])
                )  # np.mean(inc)  # np.round(np.mean(inc), 5)
                self._offset = data[0]
                self._size = data.size
                self._data = None
                self._linear = True
            else:
                self._linear = False
        else:
            self._linear = False

    def loc2index(self, loc, dim=None, *, units=None):
        """
        Return the index corresponding to a given location .

        Parameters
        ----------
        loc : float.
            Value corresponding to a given location on the coordinate's axis.

        Returns
        -------
        index : int.
            The corresponding index.

        Examples
        --------

        >>> dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")
        >>> dataset.x.loc2index(1644.0)
        4517
        """

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
                print_(
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
        """bool - Whether the axis is reversed (readonly
        property).
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
        else an array of the different spacings
        """
        if self.has_data:
            return spacings(self.data) * self.units
        return None


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
# Set the operators
# ======================================================================================
# set_operators(Coord, priority=50)

# ======================================================================================
if __name__ == "__main__":
    """ """
