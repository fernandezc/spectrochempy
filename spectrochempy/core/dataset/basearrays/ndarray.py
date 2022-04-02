# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
This module implements the base class |NDArray| and several subclasses of |NDArray|.
with (hyper)complexes related attributes, masks or labels .

These base classes are not intended to be used directly by the end user.
They serve as abstract classes that already implement the basic methods to define
spectral arrays and related attribute methods.

More specifically, derived classes such as NDDataset or Coord implement additional
methods and attributes.

What are the basic attributes of the NDArray class?
-----------------------------------------------------

The most used ones:
~~~~~~~~~~~~~~~~~~~
* `id`: A unique identifier of the NDArray object.
* `data`: An array of data contained in the object (default: None)
* `name`: A friendly name for the object (default: id)
* `dims`: The names of the dimensions.
* `title`: The title of the data array, often a quantity name(e.g, `frequency`)
* `units`: The units of the data array (for example, `Hz`).

Useful additional attributes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `meta`: A dictionary of any additional metadata.

Read-only attributes:
~~~~~~~~~~~~~~~~~~~~~
* `dtype`: data type (see numpy definition of dtypes).

Most attributes are optional and automatically set by default.
Even data can be absent.  In this case, the NDArray is considered empty.

Derived classes
---------------
* NDComplexArray : This class has the same attributes as NDArray,
  but unlike NDArray, it accepts complex or hypercomplex data.

* NDMaskedComplexArray : This class has the same attributes as NDComplexArray,
  but has an additional attribute `mask`: NOMASK or a boolean array of the same form
  as the data array.

* NDLabeledArray : This class is a 1D NDArray, with an additional attribute `labels`.
  Labels can be defined in addition to the main data or replace it completely.
"""

import copy as cpy

import itertools
import pathlib
import re
import textwrap
import uuid
from datetime import datetime

import numpy as np
import pint

import traitlets as tr
from traittypes import Array

from spectrochempy.core import info_, print_, warning_
from spectrochempy.core.common.compare import is_datetime64, is_number, is_sequence
from spectrochempy.core.common.constants import (
    DEFAULT_DIM_NAME,
    INPLACE,
    TYPE_INTEGER,
)
from spectrochempy.core.common.datetimes import from_dt64_units
from spectrochempy.core.common.docstrings import DocstringProcessor
from spectrochempy.core.common.exceptions import (
    CastingError,
    DimensionalityError,
    InvalidDimensionNameError,
    InvalidNameError,
    InvalidUnitsError,
    # ShapeError,
    UnitWarning,
    ValueWarning,
)
from spectrochempy.core.common.print import (
    colored_output,
    convert_to_html,
    numpyprintoptions,
    pstr,
)
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.units import (
    Quantity,
    Unit,
    get_units,
    remove_units,
    set_nmr_context,
    ur,
)


# Printing settings
# --------------------------------------------------------------------------------------
numpyprintoptions()


# Docstring substitution (docrep)
# --------------------------------------------------------------------------------------
_docstring = DocstringProcessor()


# Validators
# --------------------------------------------------------------------------------------
def _check_dtype():
    def validator(trait, value):
        if value.dtype.kind == "m":
            raise CastingError(
                value.dtype,
                "if you want to pass a timedelta64 array to the "
                "data attribute, use the property  "
                "'data' and not the hidden '_data' attribute.\n"
                "e.g.,  self.data = a_timedelta64_array\n "
                "\tnot  self._data = a_timedelta64_array  ",
            )
        return value

    return validator


# ======================================================================================
# A simple NDArray class
# ======================================================================================
class NDArray(tr.HasTraits):
    """
    The basic |NDArray| object.

    The |NDArray| class is an array (numpy |ndarray|-like) container, usually not
    intended to be used directly, as its basic functionalities may be quite limited,
    but to be subclassed.

    The key distinction from raw numpy |ndarray| is the presence of optional properties
    such as dimension names, units and metadata.

    Parameters
    ----------
    data : list, tuple, or |ndarray|-like object
        Data array contained in the object. Any
        size or shape of data is accepted. If not given, an empty object will
        be inited. At the initialisation the provided data will be eventually cast to a
        numpy-ndarray. If possible, the provided data will not be copied for `data`
        input, but will be passed by reference, so you should make a copy of the
        `data` before passing them if that's the desired behavior or set the `copy`
        argument to True.
    **kwargs
        Optional keywords parameters. See Other Parameters.

    Other Parameters
    ----------------
    copy : bool, optional, Default:False
        If True, a deep copy of the passed object is performed.
    dtype : str or dtype, optional, default=np.float64
        If specified, the data will be cast to this dtype.
    dims : list of chars, optional
        If specified the list must have a length equal to the number of data
        dimensions (ndim).
        If not specified, dimension names are automatically attributed in the order
        given by `DEFAULT_DIM_NAME`.
    title : str, optional
        The title of the array.
        It is optional but recommended giving a title to each array.
    meta : dict-like object, optional
        Additional metadata for this object. Must be dict-like but no
        further restriction is placed on meta.
    name : str, optional
        A user-friendly name for this object. If not given, the automatic `id`
        given at the object creation will be used as a name.
    timezone : datetime.tzinfo, optional
        The timezone where the data were created. If not specified, the local
        timezone is assumed.
    units : |Unit| instance or str, optional
        Units of the data. If data is a |Quantity| then `units` is set to the unit of
        the `data`; if a unit is also explicitly provided an error is raised.
        Handling of units use the `pint <https://pint.readthedocs.org/>`__
        package.
    """

    _docstring.get_sections(_docstring.dedent(__doc__), base="NDArray")

    # Hidden properties

    _id = tr.Unicode()
    _name = tr.Unicode()
    _title = tr.Unicode(allow_none=True)
    _data = Array(None, allow_none=True).valid(_check_dtype())
    _dtype = tr.Instance(np.dtype, allow_none=True)
    _dims = tr.List(tr.Unicode(), allow_none=True)
    _units = tr.Instance(Unit, allow_none=True)
    _meta = tr.Instance(Meta, allow_none=True)

    # Region of interest
    _roi = Array(allow_none=True)

    # Basic NDArray setting.
    # by default, we do shallow copy of the data
    # which means that if the same numpy array is used for too different NDArray,
    # they will share it.
    _copy = tr.Bool(False)

    # Other settings
    _text_width = tr.Integer(88)
    _html_output = tr.Bool(False)
    _filename = tr.Union((tr.Instance(pathlib.Path), tr.Unicode()), allow_none=True)

    def _implements(self, name=None):
        """
        Utility to check if the current object _implements a given class.

        Rather than isinstance(obj, <class>) use object._implements('<classname>').
        This is useful to check type without importing the module.

        Parameters
        ----------
        name : str, optional
            Name of the class implemented.

        Returns
        -------
        str or bool
            Return the class name or a boolean if the type is checked
        """
        if name is None:
            return self.__class__.__name__
        return name == self.__class__.__name__

    def __init__(self, data=None, **kwargs):
        self.accepted_kind = kwargs.pop("accepted_kind", "iufM")

        # By default, we try to keep a reference to the data, so we do not copy them.
        self._copy = kwargs.pop("copy", False)
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            self._dtype = np.dtype(dtype)
            if self._dtype.kind == "M":
                # by default datatime must be set in ns (at least internally, e.g.
                # due to comparison purpose
                self._dtype = np.dtype("datetime64[ns]")
        self.data = data
        self.title = kwargs.pop("title", self.title)
        if "dims" in kwargs:
            self.dims = kwargs.pop("dims")
        self.units = kwargs.pop("units", self.units)
        self.name = kwargs.pop("name", self.name)
        self.meta = kwargs.pop("meta", self.meta)

        super().__init__()

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __copy__(self):
        # in SpectroChemPy, copy is equivalent to deepcopy
        # To do a shallow copy, just use assignment.
        return self.copy()

    def __deepcopy__(self, memo=None):
        # memo not used
        return self.copy()

    def __eq__(self, other, attrs=None):
        equ = True
        otherunits = False
        otherdata = None
        if not isinstance(other, NDArray):
            # try to make some assumption to make useful comparison.
            if isinstance(other, Quantity):
                otherdata = other.magnitude
                otherunits = other.units
            elif isinstance(other, (float, int, np.ndarray)):
                otherdata = other
                otherunits = False
            else:
                equ = False
            if equ:
                if not self.has_units and not otherunits:
                    equ = np.all(self._data == otherdata)
                elif self.has_units and otherunits:
                    equ = np.all(self._data * self.units == otherdata * otherunits)
                else:  # pragma: no cover
                    equ = False
        else:
            if attrs is None:
                attrs = self._attributes()
                attrs.remove("name")  # name are uniques, so should not be compared
            for attr in attrs:
                equ &= self._compare_attribute(other, attr)
                if not equ:
                    break
        return equ

    def __getitem__(self, items, return_index=False):
        if isinstance(items, list):
            # Special case of fancy indexing
            items = (items,)
        # choose, if we keep the same or create new object
        inplace = False
        if isinstance(items, tuple) and items[-1] == INPLACE:
            items = items[:-1]
            inplace = True
        # Eventually get a better representation of the indexes
        items = self._make_index(items)
        # init returned object
        new = self if inplace else self.copy()
        if new.has_data:
            udata = new.data[items]
            new._data = np.asarray(udata)
        if new.is_empty:
            warning_(
                f"Empty array of shape {new.data.shape} resulted from slicing.\n"
                f"Check the indexes."
            )
            new = None
        # for all other cases,
        # we do not need to take care of dims, as the shape is not reduced by
        # this operation. Only a subsequent squeeze operation will do it
        if not return_index:
            return new
        return new, items

    def __hash__(self):
        # all instance of this class have same hashes, so they can be compared
        return hash((type(self), self.shape, self.units))

    def __iter__(self):
        # iterate on the first dimension
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        # len of the last dimension
        if not self.is_empty:
            return self.shape[0]
        return 0

    def __ne__(self, other, attrs=None):
        return not self.__eq__(other, attrs)

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, items, value):
        keys = self._make_index(items)
        if isinstance(value, Quantity):
            # first convert value to the current units
            try:
                value.ito(self.units)
            except pint.DimensionalityError as exc:
                raise DimensionalityError(
                    exc.dim1, exc.dim2, exc.units1, exc.units2, extra_msg=exc.extra_msg
                )

            # self._data[keys] = np.array(value.magnitude, subok=True, copy=self._copy)
            value = np.array(value.magnitude, subok=True, copy=self._copy)
        self._data[keys] = value

    def __str__(self):
        name = f" {self.name}" if self.has_defined_name else " "

        prefix = f"{type(self).__name__}{name}({self.title}): "
        units = ""
        sizestr = ""

        def _unitless_or_dimensionless(units):
            if self.is_unitless:
                return " unitless"
            if units.scaling != 1:
                return f" {units:~P}"
            return " dimensionless"

        if not self.is_empty and self.has_data:
            sizestr = f" ({self._str_shape().strip()})"
            dtype = self.dtype
            data = ""
            units = (
                f" {self.units:~P}"
                if self.has_units and not self.is_dimensionless
                else _unitless_or_dimensionless(self.units)
            )
            body = f"[{dtype}]{data}"
        else:
            body = "empty"
        return "".join([prefix, body, units, sizestr]).rstrip()

    def _argsort(self, descend=False, **kwargs):
        # find the indices sorted by values
        args = np.argsort(self._data)
        if descend:
            args = args[::-1]
        return args

    def _attributes(self, removed=[]):
        attrs = [
            "dims",
            "data",
            "name",
            "units",
            "title",
            "meta",
            "roi",
        ]
        for item in removed:
            if item in attrs:
                attrs.remove(item)
        return attrs

    def _compare_attribute(self, other, attr):
        equ = True
        sattr = getattr(self, f"_{attr}")
        if not hasattr(other, f"_{attr}"):  # pragma: no cover
            equ = False
        else:
            oattr = getattr(other, f"_{attr}")
            if sattr is None and oattr is None:
                equ = True
            elif sattr is None and oattr is not None:  # pragma: no cover
                equ = False
            elif oattr is None and sattr is not None:  # pragma: no cover
                equ = False
            elif attr == "data" and sattr is not None and sattr.dtype.kind == "M":
                equ &= np.mean(sattr - oattr) < np.timedelta64(10, "ns")
            elif attr != "data" and hasattr(sattr, "__eq__"):
                equ &= np.all(sattr.__eq__(oattr))
            else:
                equ &= np.all(sattr == oattr)
        # print(attr, eq)
        return equ

    def _cstr(self, **kwargs):
        str_name = f"         name: {self.name}"
        str_title = f"        title: {self.title}"
        return str_name, str_title, self._str_value(**kwargs), self._str_shape()

    @staticmethod
    def _data_and_units_from_td64(data):
        if data.dtype.kind != "m":  # pragma: no cover
            raise CastingError(data.dtype, "Not a timedelta array")
        newdata = data.astype(float)
        dt64unit = re.findall(r"^<m\d\[(\w+)\]$", data.dtype.str)[0]
        newunits = from_dt64_units(dt64unit)
        return newdata, newunits

    @tr.validate("_data")
    def _data_validate(self, proposal):
        data = proposal["value"]
        # we accept only float or integer values
        if data.dtype.kind not in self.accepted_kind:
            raise CastingError(
                data.dtype,
                "The data attribute accept only numbers or " "datetime64 arrays",
            )
        # return the validated data
        if self._copy:
            return data.copy()
        return data

    @tr.default("_dims")
    def __dims_default(self):
        if self.ndim > 0:
            return DEFAULT_DIM_NAME[-self.ndim :]
        return []

    @tr.default("_title")
    def __title_default(self):
        return "value"

    def _get_dims_from_args(self, *dims, **kwargs):
        # utility function to read dims args and kwargs
        # sequence of dims or axis, or `dim`, `dims` or `axis` keyword are accepted
        # check if we have arguments
        if not dims:
            dims = None
        elif isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        # Check if keyword dims (or synonym axis) exists
        axis = kwargs.pop("axis", None)
        kdims = kwargs.pop("dims", kwargs.pop("dim", axis))  # dim or dims keyword
        if kdims is not None:
            if dims is not None:
                warning_(
                    "The unnamed arguments are interpreted as `dims`. But a named "
                    "argument `dims` or `axis` has been specified. The unnamed "
                    "arguments will thus be ignored.",
                    ValueWarning,
                )
            dims = kdims
        if dims is not None and not isinstance(dims, list):
            if isinstance(dims, tuple):
                dims = list(dims)
            else:
                dims = [dims]
        if dims is not None:
            for i, item in enumerate(dims[:]):
                if item is not None and not isinstance(item, str):
                    item = self.dims[item]
                dims[i] = item
        if dims is not None and len(dims) == 1:
            dims = dims[0]
        return dims

    def _get_dims_index(self, dims):
        # get the index(es) corresponding to the given dim(s) which can be expressed
        # as integer or string
        if dims is None:
            return None
        if is_sequence(dims):
            if np.all([d is None for d in dims]):
                return None
        else:
            dims = [dims]
        axis = []
        for dim in dims:
            if isinstance(dim, int):
                axis.append(dim)  # nothing else to do
            elif isinstance(dim, str):
                if dim not in self.dims:
                    raise ValueError(
                        f"Dimension `{dim}` is not recognized "
                        f"(not in the dimension's list: {self.dims})."
                    )
                idx = self.dims.index(dim)
                axis.append(idx)
            else:
                raise InvalidDimensionNameError(
                    f"Dimensions must be specified as string or integer index, "
                    f"but a value of type {type(dim)} has been passed (value:{dim})!"
                )
        for i, item in enumerate(axis):
            # convert to positive index
            if item < 0:
                axis[i] = self.ndim + item
        axis = tuple(axis)
        return axis

    @tr.default("_id")
    def __id_default(self):
        # Return a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-', maxsplit=1)[0]}"

    @staticmethod
    def _ellipsis_in_keys(_keys):
        # Check if ellipsis (...) is present in the keys.
        test = False
        try:
            # Ellipsis
            if isinstance(_keys[0], np.ndarray):
                return False
            test = Ellipsis in _keys
        except ValueError as err:
            if err.args[0].startswith("The truth "):
                # probably an array of index (Fancy indexing)
                # should not happen anymore with the test above
                test = False
        finally:
            return test

    def _value_to_index(self, value):
        data = self._data
        error = None
        if np.all(value > data.max()) or np.all(value < data.min()):
            warning_(
                f"This ({value}) values is outside the axis limits "
                f"({data.min()}-{data.max()}).\n"
                f"The closest limit index is returned"
            )
            error = "out_of_limits"
        index = (np.abs(data - value)).argmin()
        return index, error

    def _interpret_strkey(self, key):
        # key can be a date
        try:
            key = np.datetime64(key)
        except ValueError as exc:
            raise NotImplementedError

        index, error = self._value_to_index(key)
        return index, error

    def _interpret_key(self, key):

        # Interpreting a key which is not an integer is only possible if the array is
        # single-dimensional.
        if self.is_empty:
            raise IndexError(f"Can not interpret this key:{key} on an empty array")

        if self._squeeze_ndim != 1:
            raise IndexError("Index for slicing must be integer.")

        # Allow passing a Quantity as indices or in slices
        key, units = remove_units(key)

        # Check units compatibility
        if (
            units is not None
            and (is_number(key) or is_sequence(key))
            and units != self.units
        ):
            raise InvalidUnitsError(
                f"Units of the key {key} {units} are not compatible with "
                f"those of this array: {self.units}"
            )

        if is_number(key) or isinstance(key, np.datetime64):
            # Get the index of a given values
            index, error = self._value_to_index(key)

        elif is_sequence(key):
            index = []
            error = None
            for lo_ in key:
                idx, err = self._value_to_index(lo_)
                index.append(idx)
                if err:
                    error = err

        elif isinstance(key, str):
            index, error = self._interpret_strkey(key)
        else:
            raise IndexError(f"Could not find this location: {key}")

        if not error:
            return index
        return index, error

    def _get_slice(self, key, dim):

        error = None

        # In the case of ordered 1D arrays, we can use values as slice keys.
        # allow passing a quantity as index or in slices
        # For multidimensional arrays, only integer index can be used as for numpy
        # arrays.

        if not isinstance(key, slice):
            start = key
            if not isinstance(key, TYPE_INTEGER):
                # float or quantity
                start = self._interpret_key(key)
                if isinstance(start, tuple):
                    start, error = start
                if start is None:
                    return slice(None)
            else:
                # integer
                if key < 0:  # reverse indexing
                    axis, dim = self._get_axis(dim)
                    start = self.shape[axis] + key
            stop = start + 1  # in order to keep a non squeezed slice
            return slice(start, stop, 1)

        # Slice (which allows float and quantity in the slice)
        start, stop, step = key.start, key.stop, key.step
        if start is not None and not isinstance(start, TYPE_INTEGER):
            start = self._interpret_key(start)
            if isinstance(start, tuple):
                start, error = start
        if stop is not None and not isinstance(stop, TYPE_INTEGER):
            stop = self._interpret_key(stop)
            if isinstance(stop, tuple):
                stop, error = stop
            if start is not None and stop < start:
                start, stop = stop, start
            if stop != start:
                stop = stop + 1  # to include last loc or label index
        if step is not None and not isinstance(step, TYPE_INTEGER):
            raise IndexError("Not integer step in slicing is not possible.")
            # TODO: we have may be a special case with datetime  # step = 1
        if step is None:
            step = 1
        if start is not None and stop is not None and start == stop and error is None:
            stop = stop + 1  # to include last index

        newkey = slice(start, stop, step)
        return newkey

    def _make_index(self, key):

        # Case where we can proceed directly with the provided key:
        # - a boolean selection.
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return key
        # # - already a slice or tuple of slice.
        # if isinstance(key, slice) or (
        #     isinstance(key, (tuple, list))
        #     and all([isinstance(k, slice) for k in key])
        # ):
        #     return key (this does not work if slice elements are not integer.

        # For the other case, we need to have a list of slice for each argument
        # or a single slice acting on the axis=0, so some transformations are
        # necessary.

        # For iteration, we need to have lis, not tuple, nor scalar.
        if isinstance(key, tuple):
            keys = list(key)
        else:
            keys = [
                key,
            ]

        # Iterate ellipsis.
        while self._ellipsis_in_keys(keys):
            i = keys.index(Ellipsis)
            keys.pop(i)
            for _ in range(self.ndim - len(keys)):
                keys.insert(i, slice(None))

        # Number of keys should not be higher than the number of dims!
        if len(keys) > self.ndim:
            raise IndexError("invalid index")

        # If the length of keys is lower than the number of dimension,
        # pad the list with additional dimensions
        for i in range(len(keys), self.ndim):
            keys.append(slice(None))

        # Now make slices or list of indexes for fancy indexing with all the list
        # elements
        for axis, key_ in enumerate(keys):
            # The keys are in the order of the dimension in self.dims!
            # so we need to get the correct dim in the coordinates lists
            dim = self.dims[axis]
            if is_sequence(key_):
                # Fancy indexing: all items of the sequence must be integer index
                if self._squeeze_ndim == 1:
                    key_ = [
                        self._interpret_key(k) if not isinstance(k, TYPE_INTEGER) else k
                        for k in key_
                    ]
                keys[axis] = key_
            else:
                # We need a little more work to get the slices in a correct format
                # allowing use of values for instances...
                keys[axis] = self._get_slice(key_, dim)
        return tuple(keys)

    @tr.default("_meta")
    def __meta_default(self):
        return Meta()

    @tr.default("_name")
    def __name_default(self):
        return self.id

    @tr.validate("_name")
    def __name_validate(self, proposal):
        name = proposal["value"]
        if not name:
            return self.id
        regex = r"[a-z,A-Z,0-9,_,-]+"
        pattern = re.compile(regex)
        match = pattern.findall(name)
        if len(match) != 1 or match[0] != name:
            raise InvalidNameError(
                f"name of {self._implements()} objects can't contain any space or "
                f"special characters. Ideally it should be a single word or "
                f"multiple words linked by an underscore `_` or a dash '-'."
            )
        return name

    @tr.validate("_title")
    def __title_validate(self, proposal):
        title = proposal["value"]
        if title and "GMT" in title:
            title = title.replace("GMT", "UTC")
        return title

    def _repr_html_(self):
        return convert_to_html(self)

    @tr.default("_roi")
    def __roi_default(self):
        return None

    def _set_data(self, data):
        if data is None:
            self._data = None
            return

        if isinstance(data, NDArray):
            # init data with data from another NDArray or NDArray's subclass
            # No need to check the validity of the data
            # because the data must have been already
            # successfully initialized for the passed NDArray.data
            for attr in self._attributes():
                try:
                    val = getattr(data, f"_{attr}")
                    if self._copy:
                        val = cpy.deepcopy(val)
                    setattr(self, f"_{attr}", val)
                except AttributeError:
                    # some attribute of NDDataset are missing in NDArray
                    pass
            if self._dtype is not None:
                self._data = self._data.astype(
                    self._dtype, casting="unsafe", copy=self._copy
                )
            # we do not keep name if the data array is not of the same type as self
            if data._implements() != self._implements():
                self.name = self._id

        elif isinstance(data, Quantity):
            self._data = np.array(
                data.magnitude, dtype=self._dtype, subok=True, copy=self._copy
            )
            self._units = data.units
        elif (
            not hasattr(data, "shape")
            or not hasattr(data, "__getitem__")
            or not hasattr(data, "__array_struct__")
        ) and not isinstance(data, (list, tuple)):
            # Data doesn't look like a numpy array, try converting it to
            # one.
            self._data = np.array(data, dtype=self._dtype, subok=True, copy=False)
        else:
            try:
                data = np.array(data, dtype=self._dtype, subok=True, copy=self._copy)
            except TypeError as exc:
                # Generally a casting error
                raise CastingError(self._dtype, exc.args[0])
            except ValueError:
                # happens if data is a list of quantities
                if isinstance(data[0], Quantity):
                    self._data = np.array(
                        [d.m for d in data], dtype=self._dtype, subok=True, copy=False
                    )
                    self._units = data[0].units
                    return
            if data.dtype.kind == "O":  # likely None value
                self._data = data.astype(float)
            elif data.dtype.kind == "m":  # timedelta64
                data, units = self._data_and_units_from_td64(data)
                self._data, self._units = data, units
                self._title = "time"
            else:
                self._data = data

    def _sort(self, descend=False, inplace=False, **kwargs):
        # sort an ndarray using data values
        args = self._argsort(descend, **kwargs)
        new = self if inplace else self.copy()
        new = new[args, INPLACE]
        return new

    @property
    def _squeeze_ndim(self):
        # The number of dimensions of the squeezed`data` array (Readonly property).
        if not self.has_data:
            return 0
        return len([x for x in self._data.shape if x > 1])

    def _str_shape(self):
        if self.is_empty:
            return ""
        out = ""
        shape_ = (
            x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape)))
        )
        shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)
        size = self.size
        out += (
            f"         size: {size}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )
        return out

    def _str_formatted_array(self, data, sep="\n", prefix="", units=""):
        das = data.copy()
        arraystr = np.array2string(das, separator=" ", prefix=prefix)
        arraystr = arraystr.replace("\n", sep)
        text = "".join([arraystr, units])
        text += sep
        return text

    def _get_axis(self, *dims, **kwargs):
        # """
        # Helper function to determine an axis index.
        #
        # It is designed to work whatever the syntax used: axis index or dimension names.
        #
        # Parameters
        # ----------
        # *dims : str, int, or list of str or index
        #     The axis indexes or dimensions names - they can be specified as argument
        #     or using keyword 'axis', 'dim' or 'dims'.
        #
        # Returns
        # -------
        # axis : int
        #     The axis indexes.
        # dim : str
        #     The axis name.
        #
        # Other Parameters
        # ----------------
        # negative_axis : bool, optional, default=False
        #     If True a negative index is returned for the axis value
        #     (-1 for the last dimension, etc...).
        # allow_none : bool, optional, default=False
        #     If True, if input is none then None is returned.
        # only_first : bool, optional, default: True
        #     By default return only information on the first axis if dim is a list.
        #     Else, return a list for axis and dims information.
        # """

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(*dims, **kwargs)
        axis = self._get_dims_index(dims)
        allow_none = kwargs.get("allow_none", False)
        if axis is None and allow_none:
            return None, None
        if isinstance(axis, tuple):
            axis = list(axis)
        if not isinstance(axis, list):
            axis = [axis]
        dims = axis[:]
        for i, cax in enumerate(axis[:]):
            # axis = axis[0] if axis else self.ndim - 1  # None
            if cax is None:
                cax = self.ndim - 1
            if kwargs.get("negative_axis", False):
                cax = cax - self.ndim if cax >= 0 else cax
            axis[i] = cax
            dims[i] = self.dims[cax]
        only_first = kwargs.pop("only_first", True)
        if len(dims) == 1 and only_first:
            dims = dims[0]
            axis = axis[0]
        return axis, dims

    def _get_data_to_print(self, ufmt=" {:~K}"):
        units = ""
        data = self.values
        if isinstance(data, Quantity):
            data = data.magnitude
            units = ufmt.format(self.units) if self.has_units else ""
        return data, units

    def _str_value(
        self, sep="\n", ufmt=" {:~P}", prefix="", header="         data: ... \n"
    ):
        #
        # Empty data case
        if self.is_empty and "data: ..." not in header:
            return f'{header}{textwrap.indent("empty", " " * 9)}'
        if self.is_empty:
            return f'{textwrap.indent("empty", " " * 9)}'
        #
        # set numpy formatting options
        numpyprintoptions(precision=4, edgeitems=3, spc=1, linewidth=88)
        #
        # Create and return formatted output
        data, units = self._get_data_to_print(ufmt)
        text = self._str_formatted_array(data, sep, prefix, units).strip()
        out = ""
        if "\n" not in text:  # single line!
            out += header.replace("...", f"\0{text}\0")
        else:
            out += header
            out += f'\0{textwrap.indent(text, " " * 9)}\0'
        out = out.rstrip()  # remove the trailing '\n'
        numpyprintoptions()
        return out

    @staticmethod
    def _unittransform(new, units):
        oldunits = new.units
        udata = (new.data * oldunits).to(units)
        new._data = udata.m
        new._units = udata.units
        if new._roi is not None:
            roi = (np.array(new.roi) * oldunits).to(units)
            new._roi = roi.m
        return new

    @staticmethod
    def _uarray(data, units=None):
        # return the array or scalar with units
        # if data.size==1:
        #    uar = data.squeeze()[()]
        # else:
        uar = data
        if units:
            return Quantity(uar, units)
        return uar

    # ----------------------------------------------------------------------------------
    # Public Methods and Properties
    # ----------------------------------------------------------------------------------
    @_docstring.get_docstring(base="astype")
    @_docstring.dedent
    def astype(self, dtype=None, casting="safe", inplace=False):
        """
        Cast the data to a specified type.

        Modify the data type according to the casting rules.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
            Controls what kind of data casting may occur. Defaults to ‘safe'

            * ‘no’ means the data types should not be cast at all.
            * ‘equiv’ means only byte-order changes are allowed.
            * ‘safe’ means only casts which can preserve values are allowed.
            * ‘same_kind’ means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * ‘unsafe’ means any data conversions may be done.
        %(inplace)s

        Returns
        -------
        %(out)s
        """
        if self.has_data and dtype is not None:
            new = self.copy() if not inplace else self
            dtype = np.dtype(dtype)
            try:
                new._data = new.data.astype(dtype, casting=casting, copy=False)
                return new
            except TypeError as exc:
                raise CastingError(dtype, exc)
            except (tr.TraitError, CastingError) as exc:
                if dtype.kind == "m":
                    new.data = new.data.astype(dtype, casting=casting, copy=False)
                    return new
                raise CastingError(dtype, exc.message)
        return self

    @_docstring.dedent
    def copy(self, keepname=True):
        """
        Make a copy of the current object.

        The copy is a deepcopy disconnected from the original.

        Parameters
        ----------
        keepname : bool, optional, default=True
            If True keep the same name for the copied object.

        Returns
        -------
        %(new)s
        """
        copy = cpy.deepcopy
        new = type(self)()
        if not keepname:
            # remove name from the list of attributes to copy
            removed = ["name"]
        else:
            removed = []
        for attr in self._attributes(removed=removed):
            _attr = copy(getattr(self, f"_{attr}"))
            setattr(new, f"_{attr}", _attr)
        return new

    @property
    @_docstring.get_docstring(base="data")
    @_docstring.dedent
    def data(self):
        """
        Return or set the `data` array.

        The `data` array contains the object values if any or None.
        """
        return self._data

    @data.setter
    def data(self, data):
        # property.setter for data
        # note that a subsequent validation is done in _data_validate
        # NOTE: as property setter doesn't work with super(),
        # see
        # https://stackoverflow.com/
        #              questions/10810369/python-super-and-setting-parent-class-property
        # we use an intermediate function that can be called from a subclass
        self._set_data(data)

    @property
    def is_dimensionless(self):
        """
        Return whether the `data` array is dimensionless.

        This property is read-only.

        See Also
        --------
        is_unitless : Return whether the data have no units.
        has_units : Return whether the data have units.

        Notes
        -----
        `Dimensionless` is different of `unitless` which means no units.
        """
        if self.is_unitless:
            return False
        return self.units.dimensionless

    @property
    def dims(self):
        """
        Return the names of the dimensions.

        By default, the name of the dimensions are 'x', 'y', 'z'....
        depending on the number of dimensions.
        """
        ndim = self.ndim
        if ndim > 0:
            dims = self._dims[:ndim]
            return dims
        return []

    @dims.setter
    def dims(self, values):
        if isinstance(values, str) and len(values) == 1:
            values = [values]
        if not is_sequence(values) or len(values) != self.ndim:
            raise ValueError(
                f"a sequence of chars with a length of {self.ndim} is expected, "
                f"but `{values}` has been provided"
            )

        self._dims = tuple(values)

    @property
    def dtype(self):
        """
        Return the data type.
        """
        if self.is_empty:
            return None
        return self._data.dtype

    @property
    def has_data(self):
        """
        Return whether the `data` array is not empty and size > 0.
        """
        if self._data is None or self._data.size == 0:
            return False

        return True

    @property
    def has_defined_name(self):
        """
        True is the name has been defined (bool).
        """
        return not (self.name == self.id)

    @property
    def has_units(self):
        """
        Return whether the `data` array have units.

        See Also
        --------
        is_unitless : Return whether the data have no units.
        is_dimensionless : Return whether the units of data are dimensionless.

        Notes
        -----
        This method return false for both unitless or dimensionless cases.
        """
        if self.units:
            if not f"{self.units:P}".strip():
                return False
            return True
        return False

    @property
    def id(self):
        """
        Return a read-only object identifier.
        """
        return self._id

    @property
    def is_1d(self):
        """
        Return whether the `data` array has only one dimension.
        """
        return self._squeeze_ndim == 1

    @property
    def is_dt64(self):
        """
        Return whether the data have a np.datetime64 dtype.
        """
        return is_datetime64(self)

    @property
    @_docstring.get_docstring(base="is_empty")
    @_docstring.dedent
    def is_empty(self):
        """
        Return whether the `data` array is empty.

        `is_empty`is true if there is no data or the data array has a size=0.
        """
        if not self.has_data or self._data.size == 0:
            return True
        return False

    @property
    def is_float(self):
        """
        Return whether the `data` are real float values.
        """
        if not self.has_data:
            return False
        return self._data.dtype.kind == "f"

    @property
    def is_real(self):
        """
        Return whether the `data` are real float values.

        (alias of is_float property)
        """
        return self.is_float

    @property
    def is_integer(self):
        """
        Return whether `data` are integer values.
        """
        if not self.has_data:
            return False
        return self._data.dtype.kind == "i"

    @_docstring.dedent
    def is_units_compatible(self, other):
        """
        Check the compatibility of units with another object.

        This method compare the units of another object with the present units.

        Parameters
        ----------
        other : |ndarray|
            The ndarray object for which we want to compare units compatibility.

        Returns
        -------
        bool
            Whether the units are compatible.
        """
        try:
            other.to(self.units, inplace=False)
        except DimensionalityError:
            return False
        return True

    def ito(self, other, force=False):
        """
        Inplace scaling of the current object data to different units.

        Same as `to` with the `inplace` parameters is True.

        Parameters
        ----------
        other : |Unit|, |Quantity| or str
            Destination units.
        force : bool, optional, default=`False`
            If True the change of units is forced, even for incompatible units.

        See Also
        --------
        to : Rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Rescaling to reduced units.
        """
        self.to(other, inplace=True, force=force)

    def ito_base_units(self):
        """
        Inplace rescaling to base units.

        Same as `to_base_units` with the `inplace`parameter set to True.

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Inplace rescaling to reduced units.
        """
        self.to_base_units(inplace=True)

    def ito_reduced_units(self):
        """
        Quantity scaled in place to reduced units, inplace.

        Same as `to_reduced_units` with `inplace` set to True.

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object to different units.
        to_reduced_units : Rescaling to reduced units.
        """
        self.to_reduced_units(inplace=True)

    @property
    def limits(self):
        """
        Return the maximum range of the data.
        """
        if not self.has_data:
            return None
        return np.array([self._data.min(), self._data.max()])

    @property
    def local_timezone(self):
        """
        Return the local timezone.
        """
        return str(datetime.utcnow().astimezone().tzinfo)

    @property
    def title(self):
        """
        Return a user-friendly name for the array title.

        When the title is provided, it can be used for labeling the object,
        e.g., axe title in a matplotlib plot.
        """
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def m(self):
        """
        Alias for data.
        """
        return self._data

    @property
    def magnitude(self):
        """
        Alias for data.
        """
        return self._data

    @property
    def meta(self):
        """
        Return an additional metadata dictionary.
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is not None:
            self._meta.update(meta)

    @property
    def name(self):
        """
        Return a user-friendly name.
        """
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            self._name = name

    @property
    @_docstring.get_docstring(base="ndim")
    @_docstring.dedent
    def ndim(self):
        """
        Return the number of dimensions of the `data` array.
        """
        if not self.size:
            return 0
        return self._data.ndim

    @property
    def roi(self):
        """
        Return the region of interest (ROI) limits.
        """
        if self._roi is None:
            self._roi = self.limits
        return self._roi

    @roi.setter
    def roi(self, val):
        self._roi = np.array(val)

    @property
    def roi_values(self):
        """
        Return the values correcponding to the region of interest (ROI) limits.
        """
        if self.units is None:
            return self.roi
        return self._uarray(self.roi, self.units)

    @property
    @_docstring.get_summary(base="shape")
    @_docstring.dedent
    def shape(self):
        """
        Return a tuple with the size of each dimension.

        The number of `data` element on each dimension.
        """
        if not self.has_data:
            return ()
        return self._data.shape

    @property
    def size(self):
        """
        Return the size of the underlying `data` array.

        The total number of data elements (possibly complex or hypercomplex
        in the array).
        """
        if not self.has_data:
            return 0
        return self._data.size

    @property
    def summary(self):
        """
        Return a detailled summary of the object content.
        """
        return colored_output(pstr(self))

    @_docstring.dedent
    def to(self, other, inplace=False, force=False):
        """
        Return the object with data rescaled to different units.

        If the required units are not compatible with the present units,
        it is possible to force the change using the `force` parameter.

        Parameters
        ----------
        other : |Quantity| or str
            Destination units.
        %(inplace)s
        force : bool, optional, default=False
            If True the change of units is forced, even for incompatible units.

        Returns
        -------
        %(out)s

        See Also
        --------
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Rescaling to reduced units.
        """
        if self.is_dt64:
            warning_("`to` method cannot be used with datetime object. Ignored!")
            return self

        new = self.copy()
        if other is None:
            if self.units is None:
                return new
            if force:
                new._units = None
                if inplace:
                    self._units = None
            return new
        units = get_units(other)

        if self.has_units:
            oldunits = self._units
            try:
                if new.meta.larmor:  # _origin in ['topspin', 'nmr']
                    set_nmr_context(new.meta.larmor)
                    with ur.context("nmr"):
                        new = self._unittransform(new, units)

                # particular case of dimensionless units: absorbance and transmittance
                else:

                    if f"{oldunits:P}" in ["transmittance", "absolute_transmittance"]:
                        if f"{units:P}" == "absorbance":
                            udata = (new.data * new.units).to(units)
                            new._data = -np.log10(udata.m)
                            new._units = units
                            new._title = "absorbance"

                        elif f"{units:P}" in [
                            "transmittance",
                            "absolute_transmittance",
                        ]:
                            new._data = (new.data * new.units).to(units)
                            new._units = units
                            new._title = "transmittance"

                    elif f"{oldunits:P}" == "absorbance":
                        if f"{units:P}" in ["transmittance", "absolute_transmittance"]:
                            scale = Quantity(1.0, self._units).to(units).magnitude
                            new._data = 10.0 ** -new.data * scale
                            new._units = units
                            new._title = "transmittance"
                    else:
                        new = self._unittransform(new, units)
                        # change the title for spectroscopic units change
                        if (
                            oldunits.dimensionality
                            in [
                                "1/[length]",
                                "[length]",
                                "[length] ** 2 * [mass] / [time] ** 2",
                            ]
                            and new._units.dimensionality == "1/[time]"
                        ):
                            new._title = "frequency"
                        elif (
                            oldunits.dimensionality
                            in ["1/[time]", "[length] ** 2 * [mass] / [time] ** 2"]
                            and new._units.dimensionality == "1/[length]"
                        ):
                            new._title = "wavenumber"
                        elif (
                            oldunits.dimensionality
                            in [
                                "1/[time]",
                                "1/[length]",
                                "[length] ** 2 * [mass] / [time] ** 2",
                            ]
                            and new._units.dimensionality == "[length]"
                        ):
                            new._title = "wavelength"
                        elif (
                            oldunits.dimensionality
                            in ["1/[time]", "1/[length]", "[length]"]
                            and new._units.dimensionality == "[length] ** 2 * "
                            "[mass] / [time] "
                            "** 2"
                        ):
                            new._title = "energy"

                if force:
                    new._units = units
            except pint.DimensionalityError as exc:
                if force:
                    new._units = units
                    info_("units forced to change")
                else:
                    raise DimensionalityError(
                        exc.dim1,
                        exc.dim2,
                        exc.units1,
                        exc.units2,
                        extra_msg=exc.extra_msg,
                    )
        elif force:
            new._units = units
        else:
            warning_("There is no units for this NDArray!", UnitWarning)
        if inplace:
            self._data = new._data
            self._units = new._units
            self._title = new._title
            self._roi = new._roi

        return new

    @_docstring.dedent
    def to_base_units(self, inplace=False):
        """
        Return an array rescaled to base units.

        E.g. base units of `hour` is `second`.

        Parameters
        ----------
        %(inplace)s

        Returns
        -------
        %(out)s

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of to different units.
        to_reduced_units : Rescaling to reduced units.
        ito_reduced_units : Inplace rescaling to reduced units.
        """
        quant = Quantity(1.0, self.units)
        quant.ito_base_units()
        new = self if inplace else self.copy()
        new.ito(quant.units)
        if not inplace:
            return new
        return self

    @_docstring.dedent
    def to_reduced_units(self, inplace=False):
        """
        Return an array scaled in place to reduced units.

        Scaling to reduced units' means one unit per
        dimension. This will not reduce compound units (e.g., `J/kg` will not
        be reduced to `m**2/s**2`).

        Parameters
        ----------
        %(inplace)s

        Returns
        -------
        %(out)s

        See Also
        --------
        to : Rescaling of the current object data to different units.
        ito : Inplace rescaling of the current object data to different units.
        to_base_units : Rescaling of the current object data to different units.
        ito_base_units : Inplace rescaling of the current object to different units.
        ito_reduced_units : Inplace rescaling to reduced units.
        """
        quant = Quantity(1.0, self.units)
        quant.ito_reduced_units()
        new = self if inplace else self.copy()
        new.ito(quant.units)
        if not inplace:
            return new
        return self

    @property
    def is_unitless(self):
        """
        Return whether the `data` does not have `units`.
        """
        return self.units is None

    @property
    def units(self):
        """
        Return the units of the data.
        """
        return self._units

    @units.setter
    def units(self, units):
        if units is None:
            return
        if isinstance(units, str):
            units = ur.Unit(units)
        elif isinstance(units, Quantity):
            units = units.units
        if self.has_units and units != self.units:
            # first try to cast
            try:
                self.to(units)
            except Exception as err:
                raise InvalidUnitsError(
                    f"Provided units {units} does not match data units:"
                    f" {self.units}.\nTo force a change,"
                    f" use the to() method, with force flag set to True"
                )
        self._units = units

    @property
    @_docstring.get_docstring(base="uarray")
    @_docstring.dedent
    def uarray(self):
        """
        Return the actual quantity contained in the object.

        For unitless object, the data itself are returned.

        See Also
        --------
        values: Similar property but returning squeezed array.
        value: Alias of values.
        """
        if self.has_data:
            return self._uarray(self._data, self.units)
        return None

    @property
    @_docstring.get_docstring(base="values")
    def values(self):
        """
        Return the actual quantity (data, units) contained in this object.

        It is equivalent to uarray property, except for single-element array which are
        returned as scalar (or quantity)
        """
        if self.has_data:
            return self.uarray.squeeze()[()]
        return None

    @property
    def value(self):
        """
        Alias of `values`.
        """
        return self.values


# ======================================================================================
if __name__ == "__main__":
    """"""
