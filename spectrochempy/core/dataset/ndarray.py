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
from quaternion import as_float_array, as_quat_array
from traittypes import Array

from spectrochempy.core import error_, exception_, info_, print_, warning_
from spectrochempy.core.common.compare import is_datetime64, is_number, is_sequence
from spectrochempy.core.common.complex import as_quaternion
from spectrochempy.core.common.constants import (
    DEFAULT_DIM_NAME,
    INPLACE,
    MASKED,
    NOMASK,
    TYPE_INTEGER,
    MaskedConstant,
    typequaternion,
)
from spectrochempy.core.common.datetimes import from_dt64_units
from spectrochempy.core.common.docstrings import DocstringProcessor
from spectrochempy.core.common.exceptions import (
    CastingError,
    DimensionalityError,
    LabelsError,
    ShapeError,
    SpectroChemPyWarning,
)
from spectrochempy.core.common.print import (
    convert_to_html,
    insert_masked_print,
    numpyprintoptions,
)
from spectrochempy.core.dataset.meta import Meta
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


# docstring substitution (docrep)
# --------------------------------------------------------------------------------------
_docstring = DocstringProcessor()


# Validators
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
# The basic NDArray class
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
        super().__init__(**kwargs)
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
                    equ = np.all(self.data == otherdata)
                elif self.has_units and otherunits:
                    equ = np.all(self.data * self.units == otherdata * otherunits)
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
        if new.data is not None:
            udata = new.data[items]
            new._data = np.asarray(udata)
        if new.is_empty:
            error_(
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
        prefix = f"{type(self).__name__} ({self.title}): "
        units = ""
        sizestr = ""

        def _unitless_or_dimensionless(units):
            return " unitless" if self.is_unitless else " dimensionless"

        if not self.is_empty and self.data is not None:
            sizestr = f" ({self._str_shape().strip()})"
            dtype = self.dtype
            data = ""
            units = (
                f" {self.units:~P}"
                if self.has_units
                else _unitless_or_dimensionless(self.units)
            )
            body = f"[{dtype}]{data}"
        else:
            body = "empty"
        return "".join([prefix, body, units, sizestr]).rstrip()

    def __setitem__(self, items, value):
        keys = self._make_index(items)
        if isinstance(value, Quantity):
            # first convert value to the current units
            try:
                value.ito(self.units)
            except pint.DimensionalityError as exc:
                raise DimensionalityError(
                    exc.dim1, exc.dim2, exc.units1, exc.units2, extra_msg=exc.extra_msg
                ) from exc

            # self._data[keys] = np.array(value.magnitude, subok=True, copy=self._copy)
            value = np.array(value.magnitude, subok=True, copy=self._copy)
        self._data[keys] = value

    def __str__(self):
        strg = "\n".join(self._cstr()).replace("\0", "")
        strg = textwrap.dedent(strg).rstrip()
        return strg

    def _argsort(self, descend=False, **kwargs):
        # find the indices sorted by values
        args = np.argsort(self.data)
        if descend:
            args = args[::-1]
        return args

    @staticmethod
    def _attributes():
        return [
            "dims",
            "data",
            "name",
            "units",
            "title",
            "meta",
            "roi",
        ]

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

    def _cstr(self):
        str_name = f"         name: {self.name}"
        return str_name, self._str_value(), self._str_shape()

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
        owner = proposal["owner"]
        if owner._implements("NDArray"):
            # we accept only float or integer values
            accepted = "iufM"
            if data.dtype.kind not in accepted:
                raise CastingError(
                    data.dtype,
                    "The NDArray.data attribute accept only float, integer or "
                    "datetime64 arrays",
                )
        # return the validated data
        if self._copy:
            return data.copy()
        return data

    @tr.default("_dims")
    def _dims_default(self):
        if self.ndim > 0:
            return DEFAULT_DIM_NAME[-self.ndim :]
        return []

    @tr.default("_title")
    def _title_default(self):
        return "value"

    def _get_dims_from_args(self, *dims, **kwargs):
        # utility function to read dims args and kwargs
        # sequence of dims or axis, or `dim`, `dims` or `axis` keyword are accepted
        # check if we have arguments
        if not dims:
            dims = None
        # Check if keyword dims (or synonym axis) exists
        axis = kwargs.pop("axis", None)
        kdims = kwargs.pop("dims", kwargs.pop("dim", axis))  # dim or dims keyword
        if kdims is not None:
            if dims is not None:
                warning_(
                    "The unnamed arguments are interpreted as `dims`. But a named "
                    "argument `dims` or `axis` has been specified. The unnamed "
                    "arguments will thus be ignored.",
                    SpectroChemPyWarning,
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
                raise TypeError(
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
    def _id_default(self):
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
        data = self.data
        error = None
        if np.all(value > data.max()) or np.all(value < data.min()):
            print_(
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
            raise NotImplementedError from exc

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
            raise ValueError(
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
            if not isinstance(key, int):
                # float or quantity
                start = self._interpret_key(key)
                if isinstance(start, tuple):
                    start, error = start
                if start is None:
                    return slice(None)
            else:
                # integer
                if key < 0:  # reverse indexing
                    axis, dim = self.get_axis(dim)
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
    def _meta_default(self):
        return Meta()

    @tr.default("_name")
    def _name_default(self):
        return self.id

    @tr.validate("_name")
    def _name_validate(self, proposal):
        name = proposal["value"]
        if not name:
            return "value"
        regex = r"[a-z,A-Z,0-9,_,-]+"
        pattern = re.compile(regex)
        match = pattern.findall(name)
        if len(match) != 1 or match[0] != name:
            exception_(
                ValueError(
                    f"name of {self._implements()} objects can't contain any space or "
                    f"special characters. Ideally it should be a single word or "
                    f"multiple words linked by an underscore `_` or a dash '-'."
                )
            )
        return name

    @tr.validate("_title")
    def _title_validate(self, proposal):
        title = proposal["value"]
        if title and "GMT" in title:
            title = title.replace("GMT", "UTC")
        return title

    def _repr_html_(self):
        return convert_to_html(self)

    @tr.default("_roi")
    def _roi_default(self):
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
                self._data = self.data.astype(
                    self._dtype, casting="unsafe", copy=self._copy
                )
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
                raise CastingError(self._dtype, exc.args[0]) from exc
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
        if self.data is None:
            return 0
        return len([x for x in self.data.shape if x > 1])

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

    def _get_data_to_print(self, ufmt=" {:~K}"):
        units = ""
        data = self.values
        if isinstance(data, Quantity):
            data = data.magnitude
            units = ufmt.format(self.units) if self.has_units else ""
        return data, units

    def _str_value(
        self, sep="\n", ufmt=" {:~K}", prefix="", header="         data: ... \n"
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
                raise CastingError(dtype, exc) from exc
            except (tr.TraitError, CastingError) as exc:
                if dtype.kind == "m":
                    new.data = new.data.astype(dtype, casting=casting, copy=False)
                    return new
                raise CastingError(dtype, exc.message) from exc
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
        for attr in self._attributes():
            _attr = copy(getattr(self, f"_{attr}"))
            setattr(new, f"_{attr}", _attr)
        # name must be changed
        if not keepname:
            new._name = ""  # reinit to default
        return new

    @property
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
        return self.data.dtype

    @_docstring.dedent
    def get_axis(self, *dims, **kwargs):
        """
        Helper function to determine an axis index.

        It is designed to work whatever the syntax used: axis index or dimension names.

        Parameters
        ----------
        *dims : str, int, or list of str or index
            The axis indexes or dimensions names - they can be specified as argument
            or using keyword 'axis', 'dim' or 'dims'.
        %(kwargs)s

        Returns
        -------
        axis : int
            The axis indexes.
        dim : str
            The axis name.

        Other Parameters
        ----------------
        negative_axis : bool, optional, default=False
            If True a negative index is returned for the axis value
            (-1 for the last dimension, etc...).
        allow_none : bool, optional, default=False
            If True, if input is none then None is returned.
        only_first : bool, optional, default: True
            By default return only information on the first axis if dim is a list.
            Else, return a list for axis and dims information.
        """
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

    @property
    def has_data(self):
        """
        Return whether the `data` array is not empty and size > 0.
        """
        if (self.data is None) or (self.data.size == 0):
            return False

        return True

    # ..........................................................................
    @property
    def has_defined_name(self):
        """
        True is the name has been defined (bool).
        """
        return not (self.name == self.id)

    # ..........................................................................
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
        if self.data is None or self.data.size == 0:
            return True
        return False

    @property
    def is_float(self):
        """
        Return whether the `data` are real float values.
        """
        if self.data is None:
            return False
        return self.data.dtype.kind == "f"

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
        if self.data is None:
            return False
        return self.data.dtype.kind == "i"

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
        if self.data is None:
            return None
        return np.array([self.data.min(), self.data.max()])

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
        return self.data

    @property
    def magnitude(self):
        """
        Alias for data.
        """
        return self.data

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
        if self._name:
            return self._name
        return self._id

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
        return self.data.ndim

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
        if self.data is None:
            return ()
        return self.data.shape

    @property
    def size(self):
        """
        Return the size of the underlying `data` array.

        The total number of data elements (possibly complex or hypercomplex
        in the array).
        """
        if self.data is None:
            return 0
        return self.data.size

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
                    ) from exc
        elif force:
            new._units = units
        else:
            warning_("There is no units for this NDArray!", SpectroChemPyWarning)
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
                raise TypeError(
                    f"Provided units {units} does not match data units:"
                    f" {self.units}.\nTo force a change,"
                    f" use the to() method, with force flag set to True"
                ) from err
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
        if self.data is not None:
            return self._uarray(self.data, self.units)
        return None

    @property
    @_docstring.get_docstring(base="values")
    def values(self):
        """
        Return the actual quantity (data, units) contained in this object.

        It is equivalent to urray property, except for single-element array which are
        returned as scalar (or quantity)
        """
        if self.data is not None:
            return self.uarray.squeeze()[()]
        return None

    @property
    def value(self):
        """
        Alias of `values`.
        """
        return self.values


# ======================================================================================
# NDArray subclass : NDComplexArray
# ======================================================================================
class NDComplexArray(NDArray):
    __doc__ = _docstring.dedent(
        """
    A |NDArray| derived class with complex/quaternion related functionalities.

    The private |NDComplexArray| class is an array (numpy |ndarray|-like) container,
    usually not intended to be used directly. In addition to the |NDArray|
    functionalities, this class adds complex/quaternion related methods and properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    """
    )

    _interleaved = tr.Bool(False)

    def __init__(self, data=None, **kwargs):
        dtype = np.dtype(kwargs.get("dtype", None))
        if dtype.kind in "cV":
            kwargs["dtype"] = None  # The treatment will be done after the NDArray
            # initialisation
        super().__init__(data, **kwargs)
        if dtype.kind == "c":
            self.set_complex(inplace=True)
        if dtype.kind == "V":  # quaternion
            self.set_hypercomplex(inplace=True)

    def __getattr__(self, item):
        if item in "RRRIIIRR":
            return self.component(select=item)
        # return super().__getattr__(item)
        raise AttributeError

    def __setitem__(self, items, value):
        if self.is_hypercomplex and np.isscalar(value):
            # sometimes do not work directly : here is a work around
            keys = self._make_index(items)
            self._data[keys] = np.full_like(self.data[keys], value).astype(
                np.dtype(np.quaternion)
            )
        else:
            super().__setitem__(items, value)

    def _make_complex(self, data):
        if self.is_complex:  # pragma: no cover
            return data  # normally, this is never used as checks are done before
            # calling this function
        if data.shape[-1] % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array of real data to be transformed to complex must have "
                "an even number of columns!.",
            )
        data = data.astype(np.float64, copy=False)
        data.dtype = np.complex128
        return data

    def _make_quaternion(self, data, quat_array=False):
        if data.dtype == typequaternion:
            # nothing to do
            return data
        # as_quat_array
        if quat_array:
            if data.shape[-1] != 4:
                raise ShapeError(
                    data.shape,
                    "An array of data to be transformed to quaternion "
                    "with the option `quat_array` must have its last dimension "
                    "size equal to 4.",
                )
            data = as_quat_array(data)
            return data
        # interlaced data
        if data.ndim % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array of data to be transformed to quaternion must be 2D.",
            )
            # TODO: offer the possibility to have more dimension (interleaved)
        if not self.is_complex:
            if data.shape[1] % 2 != 0:
                raise ShapeError(
                    data.shape,
                    "an array of real data to be transformed to quaternion "
                    "must have even number of columns!.",
                )
            # convert to double precision complex
            data = self._make_complex(data)
        if data.shape[0] % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array data to be transformed to quaternion must have"
                " even number of rows!.",
            )
        rea = data[::2]
        ima = data[1::2]
        return as_quaternion(rea, ima)

    def _cplx(self):
        cplx = [False] * self.ndim
        if self.is_hypercomplex:
            cplx = [True, True]
        elif self.is_complex:
            cplx[-1] = True
        return cplx

    @staticmethod
    def _get_component(data, select="REAL"):
        """
        Take selected components of a hypercomplex array (RRR, RIR, ...).

        Parameters
        ----------
        data : ndarray
        select : str, optional, default='REAL'
            If 'REAL', only real component in all dimensions will be selected.
            Else a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        component
            A component of the complex or hypercomplex array.
        """
        if select in ["REAL", "R"]:
            select = "R" * data.ndim

        if data.dtype == typequaternion:
            w, x, y, z = as_float_array(data).T
            w, x, y, z = w.T, x.T, y.T, z.T
            if select == "I":
                as_float_array(data)[..., 0] = 0
                # see imag (we just remove the real
                # part
            elif select in ["RR", "R"]:
                data = w
            elif select == "RI":
                data = x
            elif select == "IR":
                data = y
            elif select == "II":
                data = z
            else:
                raise ValueError(
                    f"something wrong: cannot interpret `{select}` for "
                    f"hypercomplex (quaternion) data!"
                )

        elif data.dtype.kind in "c":
            w, x = data.real, data.imag
            if select in ["R", "RR", "RRR"]:
                data = w
            elif select in ["I", "RI", "RRI"]:
                data = x
            else:
                raise ValueError(
                    f"something wrong: cannot interpret `{select}` for complex "
                    f"data!"
                )
        else:
            raise ValueError(
                f"No selection was performed because datasets with no complex data "
                f"have no `{select}` component. "
            )

        return data

    def _str_shape(self):
        if self.is_empty:
            return ""
        out = ""
        cplx = self._cplx()
        shcplx = (
            x
            for x in itertools.chain.from_iterable(
                list(zip(self.dims, self.shape, cplx))
            )
        )
        shape = (
            (", ".join(["{}:{}{}"] * self.ndim))
            .format(*shcplx)
            .replace("False", "")
            .replace("True", "(complex)")
        )
        size = self.size
        sizecplx = "" if not self.has_complex_dims else "(complex)"
        out += (
            f"         size: {size}{sizecplx}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )
        return out

    @property
    def has_complex_dims(self):
        """
        Return whether at least one of the `data` array dimension is complex.
        """
        if self.data is None:
            return False
        return self.is_complex or self.is_hypercomplex

    @_docstring.dedent
    def astype(self, dtype=None, casting="safe", inplace=False):
        """%(astype)s"""
        if self.has_data and dtype is not None:
            new = self.copy() if not inplace else self
            dtype = np.dtype(dtype)
            try:
                if dtype.kind not in "cV":  # not complex nor quaternion
                    new._data = new.data.astype(dtype, casting=casting, copy=False)
                elif dtype.kind == "c":  # complex
                    new._data = self._make_complex(new.data)
                else:  # quaternion
                    new._data = self._make_quaternion(new.data)
            except TypeError as exc:
                raise CastingError(dtype, exc) from exc
            return new
        return self

    @_docstring.get_docstring(base="component")
    @_docstring.dedent
    def component(self, select="REAL"):
        """
        Take selected components of a hypercomplex array.

        By default, the real component is returned. Using the select keyword any
        components (e.g. RRR, RIR, ...) of the hypercomplex array can be selected.

        Parameters
        ----------
        select : str, optional, default='REAL'
            If 'REAL', only real component in all dimensions will be selected,
            else a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        %(new)s

        Notes
        -----
        The definition is somewhat different from e.g., Bruker TOPSPIN, as we order the
        component in the order of the dimensions in dataset:
        e.g., for dims = ['y','x'], 'IR' means that the `y` component is
        imaginary while the `x` is real.
        """
        new = self.copy()
        new._data = self._get_component(new.data, select)
        return new

    @property
    def is_complex(self):
        """
        Return whether the array is complex.
        """
        if self.data is None:
            return False
        return self.data.dtype.kind == "c"

    @property
    def is_hypercomplex(self):
        """
        Return whether the array is hypercomplex.
        """
        if self.data is None:
            return False
        return self.data.dtype.kind == "V"

    # #
    #
    #
    #
    #
    # TODO: should we keep this possibility of storing complex number?
    #       For the moment, no functions use this.
    #  ..................................................................................
    # @property
    # def is_interleaved(self):
    #     """
    #     Return whether the hypercomplex array has interleaved data.
    #     """
    #     if self.data is None:
    #         return False
    #     return self._interleaved

    @property
    def imag(self):
        """
        Return the imaginary component of the complex array.
        """
        new = self.copy()
        if not new.has_complex_dims:
            return None
        data = new.data
        if self.is_complex:
            new._data = data.imag.data
        elif self.is_hypercomplex:
            # this is a more complex situation than for real component
            # get the imaginary component (vector component)
            # q = a + bi + cj + dk  ->   qi = bi+cj+dk
            as_float_array(data)[..., 0] = 0  # keep only the imaginary component
            new._data = data  # .data
        return new

    @property
    def limits(self):
        """
        Return the range of the data.
        """
        if self.data is None:
            return None
        if self.is_complex:
            return [self.data.real.min(), self.data.real.max()]
        if self.is_hypercomplex:
            data = as_float_array(self.data)[..., 0]
            return [data.min(), data.max()]
        return [self.data.min(), self.data.max()]

    @property
    def real(self):
        """
        Return the real component of a complex array.
        """
        new = self.copy()
        if not new.has_complex_dims:
            return self
        data = new.data
        if self.is_complex:
            new._data = data.real
        elif self.is_hypercomplex:
            # get the scalar component
            # q = a + bi + cj + dk  ->   qr = a
            new._data = as_float_array(data)[..., 0]
        return new

    @_docstring.dedent
    def set_complex(self, inplace=False):
        """
        Set the object data as complex.

        When nD-dimensional array are set to complex, we assume that it is along the
        first dimension. Two succesives rows are merged to form a complex rows. This
        means that the number of row must be even. If the complexity is to be applied
        in other dimension, either transpose/swapdims your data before applying this
        function in order that the complex dimension is the first in the array.

        Parameters
        ----------
        %(inplace)s

        Returns
        -------
        %(out)s

        """
        new = self if inplace else self.copy()
        if new.has_complex_dims:
            # not necessary in this case, it is already complex
            return new
        new._data = new._make_complex(new.data)
        return new

    @_docstring.dedent
    def set_hypercomplex(self, quat_array=False, inplace=False):
        """
        Set the object data as hypercomplex.

        Four components are created : RR, RI, IR, II if they not already exit.

        Parameters
        ----------
        quat_array : bool, optional, default: False
            If True, this indicates that the four coponents of quaternion are given in
            an extra dimension: e.g., an array with shape (3,2,4) will be transfored to
            a quaternion array with shape (3,2) with each element of type quaternion.
            If False, the components are supposed to be interlaced: e.g, an array
            with shape (6,4) with float dtype or an array of shape (6,2) of complex
            dtype, will both be transformed to an array of shape (3,2) with quaternion
            dtype.
        %(inplace)s

        Returns
        -------
        %(out)s
        """
        new = self if inplace else self.copy()
        new._data = new._make_quaternion(new.data, quat_array=quat_array)
        return new

    @property
    @_docstring.dedent
    def shape(self):
        """%(shape.summary)s

        The number of `data` element on each dimension (possibly complex).
        """
        return super().shape


# ======================================================================================
# NDArray subclass : NDMaskedComplexArray
# ======================================================================================
class NDMaskedComplexArray(NDComplexArray):
    __doc__ = _docstring.dedent(
        """
    A |NDComplexArray| derived class with additional mask and related functionalities.

    The private |NDMaskedComplexArray| class is an array (numpy |ndarray|-like)
    container, usually not intended to be used directly. In addition to the
    |NDComplexArray| functionalities, this class adds a mask and related methods and
    properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    mask : array of bool or `NOMASK`, optional
        Mask for the data. The mask array must have the same shape as the
        data. The given array can be a list,
        a tuple, or a |ndarray|. Each values in the array must be `False`
        where the data are *valid* and True when
        they are not (like in numpy masked arrays). If `data` is already a
        :class:`~numpy.ma.MaskedArray`, or any object providing a `mask`, the mask
        diefined by this parameter and the mask from the mask from the data will be
        combined (`mask` OR `data.mask`).
    """
    )
    _docstring.get_sections(__doc__, base="NDMaskedComplexArray")

    # masks
    _mask = tr.Union((tr.Bool(), Array(tr.Bool()), tr.Instance(MaskedConstant)))

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        mask = kwargs.pop("mask", NOMASK)
        if np.any(mask):
            self.mask = mask

    def __getitem__(self, items, return_index=False):
        new, keys = super().__getitem__(items, return_index=True)
        if (new.data is not None) and new.is_masked:
            new._mask = new._mask[keys]
        else:
            new._mask = NOMASK
        if not return_index:
            return new
        return new, keys

    def __setitem__(self, items, value):
        if isinstance(value, (bool, np.bool_, MaskedConstant)):
            keys = self._make_index(items)
            # the mask is modified, not the data
            if value is MASKED:
                value = True
            if not np.any(self._mask):
                self._mask = np.zeros(self.data.shape).astype(np.bool_)
            self._mask[keys] = value
            return
        super().__setitem__(items, value)

    def _attributes(self):
        return super()._attributes() + ["mask"]

    @tr.default("_mask")
    def _mask_default(self):
        return NOMASK if self.data is None else np.zeros(self.data.shape).astype(bool)

    @tr.validate("_mask")
    def _mask_validate(self, proposal):
        pv = proposal["value"]
        mask = pv
        if mask is None or mask is NOMASK:
            return mask
        # no particular validation for now.
        if self._copy:
            return mask.copy()
        return mask

    def _set_data(self, data):
        mask = NOMASK
        if hasattr(data, "mask"):
            # an object with data and mask attributes
            if isinstance(data.mask, np.ndarray) and data.mask.shape == data.data.shape:
                mask = np.array(data.mask, dtype=np.bool_, copy=False)
        super()._set_data(data)
        if np.any(mask):
            self.mask = mask

    def _str_formatted_array(self, data, sep="\n", prefix="", units=""):
        ds = data.copy()
        if self.is_masked:
            dtype = self.data.dtype
            mask_string = f"--{dtype}"
            ds = insert_masked_print(ds, mask_string=mask_string)
        arraystr = np.array2string(ds, separator=" ", prefix=prefix)
        arraystr = arraystr.replace("\n", sep)
        text = "".join([arraystr, units])
        text += sep
        return text

    @staticmethod
    def _masked_data(data, mask):
        # This ensures that a masked array is returned.
        if not np.any(mask):
            mask = np.zeros(data.shape).astype(bool)
        data = np.ma.masked_where(mask, data)  # np.ma.masked_array(data, mask)
        return data

    @_docstring.dedent
    def component(self, select="REAL"):
        """%(component)s"""
        new = super().component(select)
        if new is not None and self.is_masked:
            new._mask = self.mask
        else:
            new._mask = NOMASK
        return new

    @property
    def is_masked(self):
        """
        Whether the `data` array has masked values.
        """
        try:
            if isinstance(self._mask, np.ndarray):
                return np.any(self._mask)
            if self._mask == NOMASK or self._mask is None:
                return False
            if isinstance(self._mask, (np.bool_, bool)):
                return self._mask
        except Exception:
            if self._data.dtype == typequaternion:
                return np.any(self._mask["R"])
        return False

    @property
    def mask(self):
        """
        Mask for the data (|ndarray| of bool).
        """
        if not self.is_masked:
            return NOMASK
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is NOMASK or mask is MASKED:
            pass
        elif isinstance(mask, (np.bool_, bool)):
            if not mask:
                mask = NOMASK
            else:
                mask = MASKED
        else:
            # from now, make sure mask is of type np.ndarray if it provided
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.bool_)
            if not np.any(mask):
                # all element of the mask are false
                mask = NOMASK
            elif mask.shape != self.shape:
                raise ValueError(
                    f"mask {mask.shape} and data {self.shape} shape mismatch!"
                )
        # finally, set the mask of the object
        if isinstance(mask, MaskedConstant):
            self._mask = (
                NOMASK if self.data is None else np.ones(self.shape).astype(bool)
            )
        else:
            if np.any(self._mask):
                # this should happen when a new mask is added to an existing one
                # mask to be combined to an existing one
                info_(
                    f"{type(self).__name__} is already a masked array.\n"
                    f"The new mask will be combined with the current array's mask."
                )
                self._mask |= mask  # combine (is a copy!)
            else:
                if self._copy:
                    self._mask = mask.copy()
                else:
                    self._mask = mask

    @property
    def masked_data(self):
        """
        The actual masked `data` array .
        """
        if not self.is_empty:
            return self._masked_data(self.data, self.mask)
        return self.data

    @property
    def real(self):
        """
        Return the real component of complex array (Readonly property).
        """
        new = super().real
        if new is not None and self.is_masked:
            new._mask = self.mask
        return new

    @property
    def imag(self):
        """
        Return the imaginary component of complex array (Readonly property).
        """
        new = super().imag
        if new is not None and self.is_masked:
            new._mask = self.mask
        return new

    def remove_masks(self):
        """
        Remove data masks.

        Remove all masks previously set on this array, unconditionnaly.
        """
        self._mask = NOMASK

    @property
    @_docstring.dedent
    def uarray(self):
        """%(uarray)s"""
        if self.data is not None:
            return self._uarray(self.masked_data, self.units)
        return None


# ======================================================================================
# NDArray subclass : NDLabeledArray
# ======================================================================================
class NDLabeledArray(NDArray):
    __doc__ = _docstring.dedent(
        """
    A |NDArray| derived class with additional labels and related functionalities.

    The private |NDLabeledArray| class is an array (numpy |ndarray|-like)
    container, usually not intended to be used directly. In addition to the
    |NDArray| functionalities, this class adds labels and related methods and
    properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    labels : list, tuple, or |ndarray|-like object, optional
        Labels for the `data`.
        The labels array may have an additional dimension, meaning several
        series of labels for the same data.
    """
    )
    _docstring.get_sections(__doc__, base="NDLabeledArray")

    _labels = Array(allow_none=True)

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        self.labels = kwargs.pop("labels", None)

    def __getitem__(self, items, return_index=False):
        new, keys = super().__getitem__(items, return_index=True)
        if self.is_labeled:
            if new._labels.ndim == 1:
                new._labels = np.array(self._labels[keys[0]])
            else:
                new._labels = np.array(self._labels[:, keys[0]])
        if not return_index:
            return new
        return new, keys

    def __repr__(self):
        rep = super().__repr__()
        if "empty" in rep and self.is_labeled:
            # no data but labels
            lab = self.get_labels(level=0)
            data = f" {lab}"
            size = f" (size: {len(lab)})"
            dtype = "labels"
            body = f"[{dtype}]{data}{size}"
            rep = rep.replace("empty", body)
        return rep

    def _argsort(self, descend=False, by="value", level=None):
        # found the indices sorted by values or labels
        args = self.data
        if by == "value":
            args = np.argsort(self.data)
        elif "label" in by and not self.is_labeled:
            exception_(KeyError, "no label to sort")
        elif "label" in by and self.is_labeled:
            labels = self.labels
            if len(self.labels.shape) > 1:
                # multidimensional labels
                if not level:
                    level = 0
                    # try to find a pos in the by string
                    pattern = re.compile(r"label\[(\d)]")
                    p = pattern.search(by)
                    if p is not None:
                        level = int(p[1])
                labels = self.labels[level]
            args = np.argsort(labels)
        if descend:
            args = args[::-1]
        return args

    def _attributes(self):
        return super()._attributes() + ["labels"]

    def _is_labels_allowed(self):
        return self._squeeze_ndim <= 1

    def _interpret_strkey(self, key):

        if self.is_labeled:
            labels = self._labels
            indexes = np.argwhere(labels == key).flatten()
            if indexes.size > 0:
                return indexes[-1], None

        # key can be a date
        try:
            key = np.datetime64(key)
        except ValueError as exc:
            raise NotImplementedError from exc

        index, error = self._value_to_index(key)
        return index, error

    def _get_data_to_print(self, ufmt=" {:~K}"):
        units = ""
        data = self.values
        if isinstance(data, Quantity):
            data = data.magnitude
            units = ufmt.format(self.units) if self.has_units else ""
        return data, units

    @tr.default("_labels")
    def _labels_default(self):
        return None

    # def _repr_shape(self):
    #     if self.is_empty:
    #         return "size: 0"
    #     out = ""
    #     shape_ = (
    #         x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape)))
    #     )
    #     shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)
    #     size = self.size
    #     out += f"size: {size}" if self.ndim < 2 else f"shape: ({shape})"
    #     return out
    #

    # def _repr_value(self):
    #     numpyprintoptions(precision=4, edgeitems=0, spc=1, linewidth=120)
    #     prefix = f"{type(self).__name__} ({self.name}): "
    #     units = ""
    #     if not self.is_empty:
    #         if self.data is not None:
    #             dtype = self.dtype
    #             data = ""
    #             units = " {:~K}".format(self.units) if self.has_units else " unitless"
    #         else:
    #             # no data but labels
    #             lab = self.get_labels()
    #             data = f" {lab}"
    #             dtype = "labels"
    #         body = f"[{dtype}]{data}"
    #     else:
    #         body = "empty"
    #     numpyprintoptions()
    #     return "".join([prefix, body, units])

    @property
    def _squeeze_ndim(self):
        # The number of dimensions of the squeezed`data` array (Readonly property).
        if self.data is None and self.is_labeled:
            return 1
        return super()._squeeze_ndim

    @staticmethod
    def _get_label_str(lab, idx=None, sep="\n"):
        arraystr = np.array2string(lab, separator=" ", prefix="")
        if idx is None:
            text = "       labels: "
        else:
            text = f"    labels[{idx}]: "
        text += f"\0{arraystr}\0{sep}"
        return text

    def _str_value(
        self, sep="\n", ufmt=" {:~K}", prefix="", header="         data: ... \n"
    ):
        text = super()._str_value(sep, ufmt, prefix, header)
        text += sep

        if self.is_labeled:
            lab = self.labels
            idx = 0
            if self.data is None:
                if lab.ndim > 1:
                    idx = 1
                else:
                    # already in data
                    lab = None
            if lab is not None:
                if lab.ndim == 1 and idx == 0:
                    text += self._get_label_str(lab, sep=sep)
                else:
                    for i in range(idx, len(lab)):
                        text += self._get_label_str(lab[i], i, sep=sep)
        return text.rstrip()

    def get_labels(self, level=0):
        """
        Get the labels at a given level.

        Used to replace `data` when only labels are provided, and/or for
        labeling axis in plots.

        Parameters
        ----------
        level : int, optional, default:0
            For multilabel array, specify the label row to extract.

        Returns
        -------
        |ndarray|
            The label array at the desired level or None.
        """
        if not self.is_labeled:
            return None

        if (self.labels.ndim > 1 and level > len(self.labels) - 1) or (
            self.labels.ndim == 1 and level > 0
        ):
            warning_(
                "There is no such level in the existing labels", SpectroChemPyWarning
            )
            return None

        if len(self.labels) > 1 and self.labels.ndim > 1:
            return self.labels[level]
        return self.labels

    @property
    @_docstring.dedent
    def is_empty(self):
        """%(is_empty)s"""
        if self.is_labeled:
            return False
        return super().is_empty

    @property
    def is_labeled(self):
        """
        Return whether the `data` array have labels.
        """
        # label cannot exist for now for nD dataset - only 1D dataset, such
        # as Coord can be labeled.
        if self._data is not None and self._squeeze_ndim > 1:
            return False
        return self._labels is not None and np.any(self.labels != "")

    @property
    def labels(self):
        """
        An array of labels for `data` (|ndarray| of str).

        An array of objects of any type (but most generally string), with the last
        dimension size equal to that of the dimension of data. Note that's labeling
        is possible only for 1D data. One classical application is the labeling of
        coordinates to display informative strings instead of numerical values.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):

        if labels is None:
            return
        if not self._is_labels_allowed():
            exception_(
                LabelsError("We cannot set the labels for multidimentional data.")
            )
        # make sure labels array is of type np.ndarray or Quantity arrays
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, subok=True, copy=True).astype(object, copy=False)
        if (self.data is not None) and labels.ndim > 1:
            # allow the fact that the labels
            # may have been passed in a transposed array
            if labels.shape[-1] != self.size and labels.shape[0] == self.size:
                labels = labels.T
        if self.data is not None and labels.shape[-1] != self.size:
            raise LabelsError(
                f"labels {labels.shape} and data {self.shape} " f"shape mismatch!"
            )
        if np.any(self._labels):
            info_(
                f"{type(self).__name__} is already a labeled array.\n"
                f"The explicitly provided labels will "
                f"be appended to the current labels"
            )
            labels = labels.squeeze()
            self._labels = self._labels.squeeze()
            self._labels = np.vstack((self._labels, labels))
        else:
            if self._copy:
                self._labels = labels.copy()
            else:
                self._labels = labels

    @property
    @_docstring.dedent
    def ndim(self):
        """%(ndim)s"""
        if self.data is None and self.is_labeled:
            return 1
        return super().ndim

    @property
    @_docstring.dedent
    def shape(self):
        """%(shape.summary)s

        The number of `data` element on each dimension (possibly complex or
        hypercomplex).
        For only labeled array, there is no data, so it is the 1D and the size
        is the size of the array of labels.
        """
        if self.data is None and self.is_labeled:
            # noinspection PyRedundantParentheses
            return (self.labels.shape[0],)
        if self.data is None:
            return ()
        return self.data.shape

    @property
    def size(self):
        """
        Size of the underlying `data` array - Readonly property (int).

        The total number of data elements (possibly complex or hypercomplex
        in the array).
        """

        if self.data is None and self.is_labeled:
            return self.labels.shape[-1]
        if self.data is None:
            return 0
        return self.data.size

    @property
    @_docstring.dedent
    def values(self):
        """%(values)s"""
        if self.data is not None:
            return self.uarray.squeeze()[()]
        if self.is_labeled:
            if self.labels.size == 1:
                return np.asscalar(self.labels)
            if self.labels.ndim == 1:
                return self.labels
            return self.labels[0]
        return None


# ======================================================================================
if __name__ == "__main__":
    pass
