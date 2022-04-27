# -*- coding: utf-8 -*-

# ======================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
# ======================================================================================

"""
This module implements the class |Coord|, a subclasses of |NDLabeledArray|.

This class is intended to be used for coordinates axis definitions in [NDDataset|.


What are the basic attributes of the |Coord| class?
---------------------------------------------------

The most used ones:
~~~~~~~~~~~~~~~~~~~
* `data`: A 1D array of data contained in the object (default: None)
* `name`: A friendly name for the object (default: id)
* `title`: The title of the data array, often a quantity name(e.g, `frequency`)
* `units`: The units of the data array (for example, `Hz`).
* `labels`: An array of labels to describe or completely replace the data array.

Read-only attributes:
~~~~~~~~~~~~~~~~~~~~~
* `dtype`: data type (see numpy definition of dtypes).

"""


__all__ = ["Coord", "CoordSet"]

import copy as cpy
import uuid
import warnings
from numbers import Number

import numpy as np
import traitlets as tr

from spectrochempy.core.common.compare import is_number, is_sequence
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME
from spectrochempy.core.common.exceptions import (
    InvalidCoordinatesSizeError,
    InvalidCoordinatesTypeError,
    InvalidDimensionNameError,
    SpectroChemPyWarning,
    deprecated,
)
from spectrochempy.core.common.print import colored_output, convert_to_html
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndlabeledarray import (
    NDLabeledArray,
    _docstring,
)
from spectrochempy.core.dataset.mixins.functioncreationmixin import (
    NDArraySeriesCreationMixin,
)
from spectrochempy.core.dataset.mixins.numpymixin import (
    NDArrayFunctionMixin,
    NDArrayUfuncMixin,
)
from spectrochempy.core.units import (
    Quantity,
    encode_quantity,
    get_units,
    set_nmr_context,
    ur,
)
from spectrochempy.utils.misc import spacings
from spectrochempy.utils.optional import import_optional_dependency


# ======================================================================================
# Coord
# ======================================================================================
class Coord(
    NDLabeledArray, NDArrayUfuncMixin, NDArrayFunctionMixin, NDArraySeriesCreationMixin
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
    _larmor = tr.Instance(Quantity, allow_none=True)

    _parent_dim = tr.Unicode(allow_none=True)
    _parent = tr.ForwardDeclaredInstance("CoordSet", allow_none=True)
    # forward declared instance must be in the same module

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
        "power",
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
        if item in ("default", "coords"):
            # this is in case these attributes are called while it is not a coordset.
            return self
        if self._parent is not None and hasattr(self._parent, item):
            return getattr(self._parent, item)

        raise AttributeError(f"`Coord` object has no attribute `{item}`")

    # ----------------------------------------------------------------------------------
    # Private properties and methods
    # ----------------------------------------------------------------------------------
    def _attributes(self, removed=None, added=None):
        if added is None:
            added = []
        added.extend(["linear", "decimals", "larmor"])
        return super()._attributes(removed=removed, added=added)

    def _cstr(self, **kwargs):
        out = super()._cstr(header="  coordinates: ... \n", **kwargs)
        return out

    @tr.default("_larmor")
    def __larmor_default(self):
        return None

    @tr.default("_parent")
    def __parent_default(self):
        return None

    def _to_xarray(self):
        # to be used during conversion of NDarray-like to Xarray object

        xr = import_optional_dependency("xarray")

        var = xr.Variable(dims=self.name, data=np.array(self.data.squeeze()))
        # dtype=np.float64))

        var.attrs["name"] = self.name
        var.attrs["pint_units"] = str(self.units)
        # We change name of attr `units`
        # To not override a builtin xarray units attribute.
        var.attrs["title"] = self.title
        var.attrs["roi"] = self.roi
        var.attrs = encode_quantity(var.attrs)

        coordinates = {self.name: var}

        # auxiliary coordinates
        def fromlabel(coordinates, level, label):
            label = list(map(str, label.tolist()))
            label = xr.Variable(dims=self.name, data=label)
            level = f"_{level}" if level is not None else ""
            coordinates[f"{self.name}_labels{level}"] = label
            return coordinates

        if self.is_labeled:
            self.labels = self.labels.squeeze()
            if self.labels.ndim > 1:
                for level in range(self.labels.shape[0]):
                    label = self.labels[level]
                    coordinates = fromlabel(coordinates, level, label)
            else:
                coordinates = fromlabel(coordinates, None, self.labels)

        return coordinates

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
        if variation < 10**-decimals:
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
            index, error = self._value_to_index(loc)
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
    def parent(self):
        """
        Return the parent CoordSet in case of coordinates.
        """
        return self._parent

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
# CoordSet
# ======================================================================================
class CoordSet(tr.HasTraits):
    __doc__ = _docstring.dedent(
        """
    A collection of Coord objects for a NDArray object with validation.

    This object is an iterable containing a collection of Coord objects.

    Parameters
    ----------
    *coords : |NDarray|, |NDArray| subclass or |CoordSet| sequence of objects.
        If an instance of CoordSet is found, instead of an array, this means
        that all coordinates in this coords describe the same axis.
        It is assumed that the coordinates are passed in the order of the
        dimensions of a nD numpy array (
        `row-major
        <https://docs.scipy.org/doc/numpy-1.14.1/glossary.html#term-row-major>`_
        order), i.e., for a 3d object : 'z', 'y', 'x'.
    %(kwargs)s

    Other Parameters
    ----------------
    x : |NDarray|, |NDArray| subclass or |CoordSet|
        A single coordinate associated to the 'x'-dimension.
        If a coord was already passed in the argument, this will overwrite
        the previous. It is thus not recommended to simultaneously use
        both way to initialize the coordinates to avoid such conflicts.
    y, z, u, ... : |NDarray|, |NDArray| subclass or |CoordSet|
        Same as `x` for the others dimensions.
    dims : list of str, optional
        Names of the dims to use corresponding to the coordinates. If not
        given, standard names are used: x, y, ...
    copy : bool, optional
        Perform a copy of the passed object. Default is True.

    See Also
    --------
    Coord : Explicit coordinates object.
    NDDataset: The main object of SpectroChempy which makes use of CoordSet.

    Examples
    --------
    >>> from spectrochempy import Coord, CoordSet

    Define 4 coordinates, with two for the same dimension

    >>> coord0 = Coord.linspace(10., 100., 5, units='m', title='distance')
    >>> coord1a = Coord.linspace(20., 25., 4, units='K', title='temperature')
    >>> coord1b = Coord.linspace(1., 10., 4, units='millitesla', title='magnetic field')
    >>> coord2 = Coord.linspace(0., 1000., 6, units='hour', title='elapsed time')

    Now create a coordset

    >>> cs = CoordSet(t=coord0, u=coord2, v=[coord1a, coord1b])

    Display coordinates

    >>> cs.u
    Coord u(elapsed time): [float64] hr (size: 6)

    For multiple coordinates, the default is returned
    >>> cs.v
    Coord _0(temperature): [float64] K (size: 4)

    It can also be accessed this way
    >>> cs.v_0
    Coord _0(temperature): [float64] K (size: 4)

    To change the default
    >>> cs.v.select(1)
    >>> cs.v
    Coord _1(magnetic field): [float64] mT (size: 4)
    """
    )

    # Hidden attributes containing the collection of objects
    _coords = tr.List(allow_none=True)
    _references = tr.Dict()
    _updated = tr.Bool(False)

    # Hidden id and name of the object
    _id = tr.Unicode()
    _name = tr.Unicode()

    # Hidden attribute to specify if the collection is for a single dimension
    _is_same_dim = tr.Bool(False)

    # default coord index
    _selected = tr.Int(0)

    # other settings
    _copy = tr.Bool(False)
    _sorted = tr.Bool(True)
    _html_output = tr.Bool(False)

    # ------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------

    def __init__(self, *coords, **kwargs):

        self._copy = kwargs.pop("copy", True)
        self._sorted = kwargs.pop("sorted", True)

        keepnames = kwargs.pop("keepnames", False)
        # if keepnames is false and the names of the dimensions are not passed
        # in kwargs, then use dims if not none
        dims = kwargs.pop("dims", None)

        self.name = kwargs.pop("name", None)

        # initialise the coordinate list
        self._coords = []

        # First evaluate passed args
        # --------------------------

        # some cleaning
        if coords:

            if all(
                (
                    (
                        isinstance(coords[i], (np.ndarray, NDArray, list, CoordSet))
                        or coords[i] is None
                    )
                    for i in range(len(coords))
                )
            ):
                # Any instance of a NDArray can be accepted as coordinates
                # for a dimension. If an instance of CoordSet is found,
                # this means that all coordinates in this set describe the same axis
                coords = tuple(coords)

            elif is_sequence(coords) and len(coords) == 1:
                # if isinstance(coords[0], list):
                #     coords = (CoordSet(*coords[0], sorted=False),)
                # else:
                coords = coords[0]

                if isinstance(coords, dict):
                    # we have passed a dict, postpone to the kwargs evaluation process
                    kwargs.update(coords)
                    coords = None

            else:
                raise InvalidCoordinatesTypeError(
                    "Probably invalid coordinate have been passed."
                )

        # now store the args coordinates in self._coords (validation is fired
        # when this attribute is set)
        if coords:
            for coord in coords[::-1]:  # we fill from the end of the list
                # (in reverse order) because by convention when the
                # names are not specified, the order of the
                # coords follow the order of dims.
                if not isinstance(coord, CoordSet):
                    if isinstance(coord, list):
                        # when an argument is passed as a list,
                        # it is transformed to a coordset
                        coord = CoordSet(*coord[::-1])
                    else:
                        coord = Coord(coord, copy=True)
                else:
                    coord = cpy.deepcopy(coord)

                if not keepnames:
                    if dims is None:
                        # take the last available name of available names list
                        coord.name = self.available_names.pop(-1)
                    else:
                        # use the provided list of dims
                        dim = list(dims).pop(-1)
                        if dim in self.available_names:
                            coord.name = dims.pop(-1)
                        else:
                            raise InvalidDimensionNameError(dim, self.available_names)

                self._append(coord)  # append the coord
                # (but instead of append, use assignation -in _append -
                # to fire the validation process )

        # now evaluate keywords argument
        # ------------------------------

        for key, coord in list(kwargs.items())[:]:
            # remove the already used kwargs (Fix: deprecation warning in Traitlets -
            # all args, kwargs must be used)
            del kwargs[key]

            # prepare values to be either Coord or CoordSet
            if isinstance(coord, (list, tuple)):
                # make sure in this case it becomes a CoordSet instance
                coord = CoordSet(*coord[::-1])

            elif isinstance(coord, np.ndarray) or coord is None:
                coord = Coord(
                    coord, copy=True
                )  # make sure it's a Coord  # (even if it is None -> Coord(None)

            elif isinstance(coord, str) and coord in DEFAULT_DIM_NAME:
                # may be a reference to another coordinates (e.g. same coordinates
                # for various dimensions)
                self._references[key] = coord  # store this reference
                continue

            # Populate the coords with coord and coord's name.
            if isinstance(coord, (NDArray, Coord, CoordSet)):  # NDArray,
                if key in self.available_names or (
                    len(key) == 2
                    and key.startswith("_")
                    and key[1] in list("123456789")
                ):
                    # ok we can find it as a canonical name:
                    # this will overwrite any already defined coord value
                    # which means also that kwargs have priority over args
                    coord.name = key
                    self._append(coord)

                elif not self.is_empty and key in self.names:
                    # append when a coordinate with this name is already set
                    # in passed arg. replace it
                    idx = self.names.index(key)
                    coord.name = key
                    self._coords[idx] = coord

                else:
                    raise InvalidDimensionNameError(key, DEFAULT_DIM_NAME)

            else:
                raise InvalidCoordinatesTypeError(
                    f"Probably an invalid type of coordinates has been passed: "
                    f"{key}:{coord} "
                )

        # store the item (validation will be performed)
        # self._coords = _coords

        # inform the parent about the update
        self._updated = True

        # set a notifier on the name traits name of each coordinates
        if self._coords is not None:
            for coord in self._coords:
                if coord is not None:
                    tr.HasTraits.observe(coord, self._coords_update, "_name")

        # initialize the base class with the eventual remaining arguments
        super().__init__(**kwargs)

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        # allow the following syntax: coords(), coords(0,2) or
        coords = []
        axis = kwargs.get("axis", None)
        if args:
            for idx in args:
                coords.append(self[idx])
        elif axis is not None:
            if not is_sequence(axis):
                axis = [axis]
            for i in axis:
                coords.append(self[i])
        else:
            coords = self._coords
        if len(coords) == 1:
            return coords[0]
        return CoordSet(*coords)

    def __copy__(self):
        coords = self.__class__(tuple(cpy.copy(ax) for ax in self), keepnames=True)
        # name must be changed
        coords.name = self.name
        # and is_same_dim and default for coordset
        coords._is_same_dim = self._is_same_dim
        coords._selected = self._selected
        return coords

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __delattr__(self, item):
        if "notify_change" in item:
            pass

        else:
            try:
                self.__delitem__(item)
            except (IndexError, KeyError):
                raise AttributeError

    def __delitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                del self._coords[idx]
                return

            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                index = self.titles.index(index)
                self._coords.__delitem__(index)
                return

            # maybe it is a title in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    item.__delitem__(index)
                    return

            # let try with the canonical dimension names
            if index[0] in self.names:
                # ok we can find it a canonical name:
                c = self._coords.__getitem__(self.names.index(index[0]))
                if len(index) > 1 and index[1] == "_":
                    if isinstance(c, CoordSet):
                        c.__delitem__(index[1:])
                        return

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

    def __eq__(self, other, attrs=None):
        if attrs is not None:
            # attrs.remove("coordset")
            if "transposed" in attrs:
                attrs.remove("transposed")
            if "mask" in attrs:
                attrs.remove("mask")
            if "dims" in attrs:
                attrs.remove("dims")
            if "author" in attrs:
                attrs.remove("author")
        if other is None:
            return False
        try:
            eq = True
            for c, oc in zip(self._coords, other._coords):
                eq &= c.__eq__(oc, attrs)
            return eq
        except Exception:
            return False

    def __getattr__(self, item):
        # when the attribute was not found
        if "_validate" in item or "_changed" in item:
            raise AttributeError

        # case of loc2index
        if item == "loc2index" and hasattr(self, "is_same_dim") and self.is_same_dim:
            return self._loc2index

        # case of data
        if item == "data" and hasattr(self, "is_same_dim") and self.is_same_dim:
            return self._data

        try:
            return self.__getitem__(item)
        except (IndexError, KeyError):
            raise AttributeError

    def __getitem__(self, index):

        if isinstance(index, str):

            # find by name
            if index in self.names:
                idx = self.names.index(index)
                # here we differentiate the case of single and mutiple coordinates
                c = self._coords.__getitem__(idx)
                return c.default

            # ok we did not find it!
            # let's try in references
            if index in self._references.keys():
                return self._references[index]

            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its title. "
                        f"However `{index}` occurs several time. Only"
                        f" the first occurrence is returned!"
                    )
                return self._coords.__getitem__(self.titles.index(index))

            # maybe it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet):
                    if index in item.titles:
                        # selection by subcoord title
                        return item.__getitem__(item.titles.index(index))
                    if index in item.names:
                        # selection by subcoord name
                        return item.__getitem__(item.names.index(index))

            try:
                # let try with the canonical dimension names
                # TOD0: accept arbitrary name of coordinates
                if index[0] in self.names:
                    # ok we can find it a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        if isinstance(c, CoordSet):
                            c = c.__getitem__(index[1:])
                        else:
                            c = c.__getitem__(index[2:])  # try on labels
                    return c
            except KeyError as exc:
                raise InvalidDimensionNameError(index) from exc

            except IndexError:
                pass

            raise AttributeError(
                f"Could not find `{index}` key/attribute in coordinates names or titles"
            )

        try:
            res = self._coords.__getitem__(index)
        except TypeError as exc:
            if self.is_same_dim:
                # in this case we try to slice a coord!
                # let's take the default
                return self.default.__getitem__(index)
            raise exc

        if isinstance(index, slice):
            if isinstance(res, CoordSet):
                res = (res,)
            return CoordSet(*res, keepnames=True)
        return res

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return hash(tuple(self._coords))

    def __len__(self):
        return len(self._coords)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        out = "CoordSet: [" + ", ".join(["{}"] * len(self._coords)) + "]"
        s = []
        for item in self._coords:
            if isinstance(item, CoordSet):
                s.append(f"{item.name}:" + repr(item).replace("CoordSet: ", ""))
            else:
                s.append(f"{item.name}:{item.title}")
        out = out.format(*s)
        return out

    def __setattr__(self, key, value):
        keyb = key[1:] if key.startswith("_") else key
        if keyb in [
            "parent",
            "copy",
            "sorted",
            "coords",
            "updated",
            "name",
            "html_output",
            "is_same_dim",
            "parent_dim",
            "trait_values",
            "trait_notifiers",
            "trait_validators",
            "cross_validation_lock",
            "notify_change",
        ]:
            super().__setattr__(key, value)
            return

        try:
            self.__setitem__(key, value)
        except Exception:
            super().__setattr__(key, value)

    def __setitem__(self, index, coord):
        try:
            coord = coord.copy(keepname=True)  # to avoid modifying the original
        except TypeError as e:
            if isinstance(coord, list):
                coord = [c.copy(keepname=True) for c in coord[:]]
            else:
                raise e

        if isinstance(index, str):
            # find by name
            if index in self.names:
                idx = self.names.index(index)
                coord.name = index
                self._coords.__setitem__(idx, coord)
                return

            # ok we did not find it!
            # let's try in the title
            if index in self.titles:
                # selection by coord titles
                if self.titles.count(index) > 1:
                    warnings.warn(
                        f"Getting a coordinate from its title. "
                        f"However `{index}` occurs several time. Only"
                        f" the first occurrence is returned!",
                        SpectroChemPyWarning,
                    )
                index = self.titles.index(index)
                coord.name = self.names[index]
                self._coords.__setitem__(index, coord)
                return

            # maybe it is a title or a name in a sub-coords
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.titles:
                    # selection by subcoord title
                    index = item.titles.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return
            for item in self._coords:
                if isinstance(item, CoordSet) and index in item.names:
                    # selection by subcoord title
                    index = item.names.index(index)
                    coord.name = item.names[index]
                    item.__setitem__(index, coord)
                    return

            try:
                # let try with the canonical dimension names
                if index[0] in self.names and (
                    len(index) == 1 or (len(index) == 3 and "_" in index)
                ):
                    # ok we can find it a canonical name:
                    c = self._coords.__getitem__(self.names.index(index[0]))
                    if len(index) > 1 and index[1] == "_":
                        c.__setitem__(index[1:], coord)
                    return

            except KeyError:
                pass

            # add the new coordinates
            if index in self.available_names or (
                len(index) == 2
                and index.startswith("_")
                and index[1] in list("0123456789")
            ):
                coord.name = index
                self._coords.append(coord)
                return

            raise KeyError(f"Could not find `{index}` in coordinates names or titles")

        self._coords[index] = coord

    def __str__(self):
        return repr(self)

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # ex: change {
        #   'owner': object, # The tr.HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change.name == "_updated" and change.new:
            self._updated = False  # reset

    def _append(self, coord):
        # utility function to append coordinate with full validation
        if not isinstance(coord, tuple):
            coord = (coord,)
        if self._coords:
            # some coordinates already present, prepend the new one
            self._coords = (*coord,) + tuple(
                self._coords
            )  # instead of append, fire the validation process
        else:
            # no coordinates yet, start a new tuple of coordinate
            self._coords = (*coord,)

    def _attributes(self, removed=[]):
        return ["coords", "references", "is_same_dim", "name", "selected"]

    def _coords_update(self, change=None):
        # when notified that a coord name have been updated
        self._updated = True

    @tr.validate("_coords")
    def _coords_validate(self, proposal):
        coords = proposal["value"]
        if not coords:
            return []

        for idx, coord in enumerate(coords):
            if coord and not isinstance(coord, (Coord, CoordSet)):
                raise InvalidCoordinatesTypeError(
                    "At this point all passed coordinates should be of type Coord or CoordSet!"
                )  #
            if self._copy:
                coords[idx] = coord.copy()
            else:
                coords[idx] = coord

        for coord in coords:
            if isinstance(coord, CoordSet):
                # it must be a single dimension axis
                # in this case we must have same length for all coordinates
                coord._is_same_dim = True

                # check this is valid in terms of size
                try:
                    _ = coord.sizes
                except InvalidCoordinatesSizeError as exc:
                    raise exc

                # change the internal names
                n = len(coord)
                coord._set_names([f"_{i}" for i in range(n)])
                #     [f"_{i + 1}" for i in range(n)]
                # )  # we must have  _1 for the first coordinates,
                # # _2 the second, etc...
                coord._set_parent_dim(coord.name)
                for c in coord:
                    c._parent = coord

        # last check and sorting
        names = []
        for coord in coords:
            if coord.has_defined_name:
                names.append(coord.name)
            else:
                raise ValueError(
                    "At this point all passed coordinates should have a valid name!"
                )

        if coords:
            if self._sorted:
                _sortedtuples = sorted(
                    (coord.name, coord) for coord in coords
                )  # Final sort
                coords = list(zip(*_sortedtuples))[1]
            return list(coords)  # be sure it is a list not a tuple
        return None

    def _cstr(self, header="  coordinates: ... \n", print_size=True):

        txt = ""
        for dim in self.names:
            coord = getattr(self, dim)

            if coord:

                dimension = f"     DIMENSION `{dim}`"
                for k, v in self.references.items():
                    if dim == v:
                        # reference to this dimension
                        dimension += f"=`{k}`"
                txt += dimension + "\n"

                if isinstance(coord.parent, CoordSet):
                    # txt += '        index: {}\n'.format(idx)
                    if not coord.is_empty:
                        if print_size:
                            txt += f"{coord[0]._str_shape().rstrip()}\n"

                        coord._html_output = self._html_output
                        for idx_s, dim_s in enumerate(coord.names):
                            c = getattr(coord, dim_s)
                            sel = "*" if idx_s == coord.selected else ""
                            txt += f"          ({dim_s}){sel} ...\n"
                            c._html_output = self._html_output
                            sub = c._cstr(
                                print_size=False
                            )  # , indent=4, first_indent=-6)
                            txt += f"{sub}\n"

                elif not coord.is_empty:
                    # coordinates if available
                    # txt += '        index: {}\n'.format(idx)
                    coord._html_output = self._html_output
                    txt += f"{coord._cstr(print_size=print_size)}\n"

        txt = txt.rstrip()  # remove the trailing '\n'

        if not self._html_output:
            return colored_output(txt.rstrip())
        return txt.rstrip()

    @tr.default("_id")
    def __id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-', maxsplit=1)[0]}"

    def _loc2index(self, loc):
        # Return the index of a location
        # it searches in the default coordinate for same_dim coordset.
        if not self.is_same_dim:
            raise TypeError("The current object is not a coordinate")
        return self.default.loc2index(loc)

    def _repr_html_(self):
        return convert_to_html(self)

    def _set_names(self, names):
        # utility function to change names of coordinates (in batch)
        # useful when a coordinate is a CoordSet itself
        for coord, name in zip(self._coords, names):
            coord.name = name

    def _set_parent_dim(self, name):
        # utility function to set the parent name for sub coordset
        for coord in self._coords:
            coord._parent_dim = name

    # ------------------------------------------------------------------------
    # Readonly Properties
    # ------------------------------------------------------------------------

    @property
    def available_names(self):
        """
        Chars that can be used for dimension name (list).

        It returns DEFAULT_DIM_NAMES less those already in use.
        """
        _available_names = DEFAULT_DIM_NAME.copy()
        for item in self.names:
            if item in _available_names:
                _available_names.remove(item)
        return _available_names

    @property
    def coords(self):
        """
        Coordinates in the coordset (list).
        """
        return self._coords

    @property
    def has_defined_name(self):
        """
        True if the name has been defined (bool).
        """
        return not (self.name == self.id)

    # ..........................................................................
    @property
    def id(self):
        """
        Object identifier.
        """
        return self._id

    @property
    def selected(self):
        """
        Return index of the selected coordinates for multicoordinates.
        """
        return self._selected

    @property
    def is_empty(self):
        """
        Return whether there is no coords defined.
        """
        if self._coords:
            return len(self._coords) == 0
        return True

    @property
    def is_same_dim(self):
        """
        Return whether the coords define a single dimension (multicoordinates).
        """
        return self._is_same_dim

    @property
    def references(self):
        """
        Return a dictionary containing the references to other dimensions.

        Examples
        --------

        >>> coord0 = scp.Coord.arange(10)
        >>> c = scp.CoordSet(x=coord0, y="x")
        >>> assert c.references == dict(y="x")
        """
        return self._references

    @property
    def sizes(self):
        """
        Sizes of the coord object for each dimension (int or tuple of int).

        If the coordset is for a single dimension (mulitcoordinates) return a
        single size as all coordinates must have the same.
        """
        _sizes = []
        for i, item in enumerate(self._coords):
            _sizes.append(item.size)  # recurrence if item is a CoordSet

        if self.is_same_dim:
            _sizes = list(set(_sizes))
            if len(_sizes) > 1:
                raise InvalidCoordinatesSizeError(
                    "Coordinates must be of the same size for a dimension "
                    "with multiple coordinates"
                )
            return _sizes[0]
        return _sizes

    # alias
    size = sizes

    @property
    def names(self):
        """
        Names of the coords in the current coords (list - read only property).
        """
        _names = []
        if self._coords:
            for item in self._coords:
                if item.has_defined_name:
                    _names.append(item.name)
        return _names

    @property
    def default(self):
        """
        Default coordinates (Coord).
        """
        if hasattr(self, "is_same_dim") and self.is_same_dim:
            # valid only for multi-coordinate CoordSet
            default = self[self._selected]
            return default
        # warning_('There is no multi-coordinate in this CoordSet. Return None')

    @property
    def _data(self):
        # in case data is called on a coordset for dimension with multiple coordinates
        # return the first coordinates
        if hasattr(self, "is_same_dim") and self.is_same_dim:
            return self.default.data

    # ------------------------------------------------------------------------
    # Mutable Properties
    # ------------------------------------------------------------------------

    @property
    def name(self):
        """
        Return the coordset name.
        """
        if self._name:
            return self._name
        return self._id

    @name.setter
    def name(self, value):
        if value is not None:
            self._name = value

    @property
    def titles(self):
        """
        Return titles of the coords in the current coords (list).
        """
        _titles = []
        for item in self._coords:
            if isinstance(item, NDArray):
                _titles.append(item.title if item.title else item.name)  # TODO:name
            elif isinstance(item, CoordSet):
                _titles.append(
                    [el.title if el.title else el.name for el in item]
                )  # TODO:name
            else:  # pragma: no cover
                raise ValueError("Something is wrong with the titles!")
        return _titles

    @property
    def labels(self):
        """
        Labels of the coordinates in the current coordset (list).
        """
        return [item.labels for item in self]

    @property
    def units(self):
        """
        Units of the coords in the current coords (list of string).
        """
        units = []
        for item in self:
            if isinstance(item, CoordSet):
                units.append(item.units)
            else:
                units.append(str(item.units))
        return units

    # ------------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _implements(name=None):
        """
        Utility to check if the current object implement `CoordSet`.

        Rather than isinstance(obj, CoordSet) use object.implements('CoordSet').

        This is useful to check type without importing the module.
        """
        if name is None:
            return "CoordSet"
        return name == "CoordSet"

    @_docstring.dedent
    def copy(self, keepname=False):
        """
        Make a disconnected copy of the current coords.

        Parameters
        ----------
        keepname : bool
            Whether the name ofthe coordset object should also be copied.

        Returns
        -------
        %(new)s
        """
        return self.__copy__()

    def keys(self):
        """
        Similar to the names property but includes references to other coordinates.

        Returns
        -------
        list
            List of all coordinates names (including reference to other coordinates).
        """
        keys = []
        if self.names:
            keys.extend(self.names)
        if self._references:
            keys.extend(list(self.references.keys()))
        return keys

    # ..........................................................................
    def select(self, index):
        """
        Select the default coord by index.

        Parameters
        ----------
        index : int
            Index of the default coordinate for multiple coordinates.
        """
        if index < 0:  # handle negative indexing
            index = len(self.names) + index - 1
        self._selected = min(max(0, int(index)), len(self.names) - 1)

    @_docstring.dedent
    def set(self, *coords, **kwargs):
        """
        Set one or more coordinates in the current CoordSet.

        Parameters
        ----------
        *coords : array-like, Coord or CoordSet object
            The coordinates to define the coordset.
        %(kwargs)s

        Other Parameters
        ----------------
        mode : str
            Mode is set to 'a' to append, 'w' to replace.

        Notes
        -----
        Alternatively to the coords passed in arguments, the can also be passed using
        their names: e.g.  t=Coord(...), z=CoordSet(...).
        """
        if not coords and not kwargs:
            return

        mode = kwargs.pop("mode", "a")

        if mode == "w":
            self._coords = []

        if len(coords) == 1 and (
            (is_sequence(coords[0]) and not isinstance(coords[0], Coord))
            or isinstance(coords[0], CoordSet)
        ):
            coords = coords[0]

        if isinstance(coords, CoordSet):
            kwargs.update(coords.to_dict())
            coords = ()

        for i, item in enumerate(coords[::-1]):
            item.name = self.available_names.pop()
            self._append(item)

        for k, item in kwargs.items():
            if isinstance(item, CoordSet):
                # try to keep this parameter to True!
                item._is_same_dim = True
            self[k] = item

    @_docstring.dedent
    def set_titles(self, *titles, **kwargs):
        """
        Set one or more coord title at once.

        Parameters
        ----------
        *titles : str(s)
            The list of titles to apply to the set of coordinates. They must be given
            according to the coordinate's name alphabetical order.
        %(kwargs)s

        Other Parameters
        ----------------
        **titles
            Keyword attribution of the titles. The keys must be valid names among the
            coordinate's name list. This is the recommended way to set titles as this
            will be less prone to errors.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's
        name alphabetical order :
        e.g, the first title will be for the `x` coordinates,
        the second for the `y`, etc.
        """
        if len(titles) == 1 and (
            is_sequence(titles[0]) or isinstance(titles[0], CoordSet)
        ):
            titles = titles[0]

        for i, item in enumerate(titles):
            if not isinstance(self[i], CoordSet):
                self[i].title = item
            else:
                if is_sequence(item):
                    for j, v in enumerate(self[i]):
                        v.title = item[j]

        for k, item in kwargs.items():
            self[k].title = item

    def set_units(self, *units, **kwargs):
        """
        Set one or more coord units at once.

        Parameters
        ----------
        *units : str(s)
            The list of units to apply to the set of coordinates. They must be given
            according to the coordinate's name alphabetical order.
        **kwargs
            Keyword attribution of the units. The keys must be valid names among the
            coordinate's name list. This is the recommended way to set units as this
            will be less prone to errors.
            See also Other Parameters.

        Other Parameters
        ----------------
        force : bool, optional, default=False
            Whether the new units must be compatible with the current units.
            See the `Coord`.`to` method.

        Notes
        -----
        If the args are not named, then the attributions are made in coordinate's name
        alphabetical order : e.g, the first units will be for the `x` coordinates,
        the second for the `y`, etc.
        """
        force = kwargs.pop("force", False)

        if len(units) == 1 and is_sequence(units[0]):
            units = units[0]

        for i, item in enumerate(units):
            if not isinstance(self[i], CoordSet):
                self[i].to(item, force=force, inplace=True)
            else:
                # if is_sequence(item):
                #     for j, v in enumerate(self[i]):
                #         v.to(item[j], force=force, inplace=True)
                self[i].set_units(item, force=force)
        for k, item in kwargs.items():
            self[k].to(item, force=force, inplace=True)

    def to_dict(self):
        """
        Return a dict of the coordinates from the coordset.

        Returns
        -------
        dict
            A dictionary where keys are the names of the coordinates, and the values
            the coordinates themselves.
        """
        return dict(zip(self.names, self._coords))

    @_docstring.dedent
    def update(self, **kwargs):
        """
        Update a specific coordinates in the CoordSet.

        Parameters
        ----------
        %(kwargs)s

        Other Parameters
        ----------------
        **names
            Only keywords among the `CoordSet.names` are allowed - they denote the
            name of a dimension.
        """
        dims = kwargs.keys()
        for dim in list(dims)[:]:
            if dim in self.names:
                # we can replace the given coordinates
                idx = self.names.index(dim)
                self[idx] = Coord(kwargs.pop(dim), name=dim)
            elif len(dim) > 1 and dim[1] == "_":  # Probably in a sub-coordset
                idx = self.names.index(dim[0])
                subdim = dim[1:]
                self[idx].update(**{subdim: kwargs.pop(dim)})
            else:
                raise InvalidDimensionNameError(dim)


# ======================================================================================
if __name__ == "__main__":
    """ """
