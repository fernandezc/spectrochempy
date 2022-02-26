# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

from copy import copy, deepcopy

import numpy as np
import pytest
from traitlets import TraitError
from quaternion import as_float_array, as_quat_array, quaternion

import spectrochempy
from spectrochempy.core.dataset.ndarray import (
    NDArray,
    CastingError,
    NDComplexArray,
    NDMaskedComplexArray,
    NDLabelledArray,
)
from spectrochempy.core.units import Quantity, ur
from spectrochempy.core.exceptions import DimensionalityError, UnknownTimeZoneError
from spectrochempy.utils import (
    INPLACE,
    MASKED,
    NOMASK,
    TYPE_COMPLEX,
    TYPE_FLOAT,
    TYPE_INTEGER,
    RandomSeedContext,
    assert_approx_equal,
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_produces_warning,
)

from ... import test_docstrings as td

typequaternion = np.dtype(np.quaternion)

# ------------------------------------------------------------------
# create reference arrays
# ------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref_mask = ref_data < -4
    ref_mask[0, 0] = True


# ------------------------------------------------------------------
# Fixtures: some NDArray's
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def refarray():
    return ref_data


@pytest.fixture(scope="module")
def refmask():
    return ref_mask


@pytest.fixture(scope="module")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, comment="An array", copy=True)


@pytest.fixture(scope="module")
def ndarrayunit():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, units="m/s", copy=True)


# ##############
# TEST NDArray #
# ##############
def test_ndarray_eq(ndarrayunit, ndarray):
    nd = ndarrayunit.copy()
    nd.meta = {"some_metadata": "toto"}
    nd1 = ndarrayunit.copy()
    assert nd1 == nd
    nd1.meta.some_metadata = "titi"
    assert nd1 != nd
    nd2 = ndarrayunit.copy()
    nd2._units = ur.km
    assert nd2 != nd
    # check comparison unitless and dimensionless
    nd = ndarray.copy()
    nd1 = ndarray.copy()
    nd1._units = ur.absorbance
    assert nd != nd1
    # check some uncovered situation
    assert nd != [False]
    assert nd.__eq__(nd1, attrs=["data"])
    # case of datetime64
    nd = NDArray(["2022-01", "2023-02", "2024-03"], dtype="datetime64")
    assert nd.data.dtype.kind == "M"
    nd1 = NDArray(["2022-01", "2023-02", "2024-03-02"], dtype="datetime64")
    assert nd1 != nd


def test_ndarray_getitem(ndarray, ndarrayunit, refarray):
    ref = refarray.copy()
    nd = ndarray.copy()
    assert nd.dims == ["y", "x"]
    # slicing is different in scpy than with numpy. We always return
    # unsqueezed dimensions, except for array of size 1, which are considered as scalar
    with pytest.raises(IndexError):
        # too many keys
        nd[0, 0, 1]
    nd1 = nd[0, 0]
    assert_equal(nd1.data, nd.data[0:1, 0:1])
    assert nd1 is not nd[0, 0]
    assert nd1.ndim == 2  # array not reduced
    assert nd1.size == 1
    assert nd1.shape == (1, 1)
    assert isinstance(nd1, NDArray)
    assert isinstance(nd1.data, np.ndarray)
    assert isinstance(nd1.values, TYPE_FLOAT)
    nd1b = nd.__getitem__(
        (0, 0),
    )
    assert nd1b == nd1
    _, index = nd.__getitem__((0, 0), return_index=True)
    assert index == (slice(0, 1, 1), slice(0, 1, 1))
    nd1a = nd[0, 0:2]
    assert_equal(nd1a.data, nd.data[0:1, 0:2])
    assert nd1a is not nd[0, 0:2]
    assert nd1a.ndim == 2
    assert nd1a.size == 2
    assert nd1a.shape == (1, 2)
    assert isinstance(nd1a, NDArray)
    assert nd1a.dims == ["y", "x"]
    nd1b = nd[11:, 11:]
    # returning none if empty when slicing
    assert nd1b is None
    nd2 = nd[7:10]
    assert_equal(nd2.data, nd.data[7:10])
    nd3 = nd2[1]
    assert nd3.shape == (1, ref.shape[1])
    assert nd3.dims == ["y", "x"]
    nd4 = nd2[:, 1]
    assert nd4.shape == (3, 1)
    assert nd4.dims == ["y", "x"]
    # squezzing
    nd5 = nd4.squeeze()
    assert nd5.shape == (3,)
    assert nd5.dims == ["y"]
    # boolean indexing
    nd = ndarray.copy()
    nd1 = nd[nd.data > 0]
    assert nd1.shape == (47,)
    # fancy indexing
    df = nd.data[[-1, 1, 0]]
    ndf = nd[[-1, 1, 0]]
    assert_array_equal(ndf.data, df)
    # inplace assignment
    ndf = nd[[-1, 1, 0], INPLACE]
    assert_array_equal(ndf, nd)
    # use with selection from other numpy functions
    nd = ndarray.copy()
    am = np.argmax(nd.data, axis=1)
    assert_array_equal(am, np.array([6, 3, 2, 2, 0, 0, 5, 3, 4, 7]))
    amm = nd.data[..., am]
    assert_array_equal(nd[..., am].data, amm)


def test_ndarray_init(refarray, refmask, ndarray, ndarraymask):
    # initialisation with null array
    nd = NDArray()
    assert nd._implements("NDArray")
    assert nd._implements() == "NDArray"
    assert isinstance(nd, NDArray)
    assert nd.is_empty
    assert nd.id.startswith("NDArray_")
    assert not nd.has_data
    assert len(nd) == 0
    assert nd.shape == ()
    assert nd.id.startswith("NDArray")
    assert nd.quantity is None
    assert nd.ndim == 0
    assert nd.size == 0
    assert nd.dtype is None
    assert nd.is_unitless
    assert not nd.dims
    assert nd.meta == {}
    assert hash(nd) is not None
    assert repr(nd) == "NDArray (value): empty"
    # Passing attributes during initialisation.
    nd = NDArray(
        name="intensity", quantity="Raman intensity", title="An example of NDArray"
    )
    assert nd.name == "intensity"
    assert nd.quantity == "Raman intensity"
    # Initialisation with a scalar
    nd = NDArray(25)
    assert nd.data == np.array(25)
    assert nd.data.dtype in TYPE_INTEGER
    # Initialisation with a scalar + cast to float
    nd = NDArray(25, dtype="float64")
    assert nd.data == np.array(25.0)
    assert nd.data.dtype in TYPE_FLOAT
    # Initialisation with a quantity
    nd = NDArray(13.0 * ur.tesla)
    assert nd.data == np.array(13.0)
    assert nd.data.dtype in TYPE_FLOAT
    assert nd.shape == ()
    assert nd.ndim == 0
    assert not nd.dims
    assert nd.units == "tesla"
    assert nd.values == 13.0 * ur.tesla
    # initialisation with a 1D array  quantity
    nd = NDArray([13.0] * ur.tesla)
    assert nd.data == np.array([13.0])
    assert nd.shape == (1,)
    assert nd.ndim == 1
    assert nd.dims == ["x"]
    assert nd.units == "tesla"
    assert nd.values == 13.0 * ur.tesla
    # initialisation with a 1D vector quantity
    nd = NDArray([[13.0, 20.0]] * ur.tesla)
    assert_array_equal(nd.data, np.array([[13.0, 20.0]]))
    assert nd.shape == (1, 2)
    assert nd.ndim == 2
    assert nd.dims == ["y", "x"]
    assert nd.units == "tesla"
    # initialisation with a sequence
    nd = NDArray((2, 3, 4))
    assert nd.shape == (3,)
    assert nd.size == 3
    assert nd.dims == ["x"]
    # Initialisation with a list
    nd = NDArray([1, 2, 3], name="xxxx")
    assert nd[1].value == 2  # only a single element so we get a squeezed array
    nd.units = "absorbance"
    assert nd.units == ur.absorbance
    assert nd[2] == 3 * ur.absorbance
    assert nd.dims == ["x"]
    # initialization with an array
    nd = NDArray(refarray)
    assert nd.shape == refarray.shape
    assert nd.size == refarray.size
    # initialization with an NDArray object
    nd = NDArray(ndarray)
    assert nd.quantity is None
    assert nd.shape == refarray.shape
    assert nd.dims == ["y", "x"]
    assert nd.size == refarray.size
    assert_array_equal(nd.data, refarray)
    assert nd._data is ndarray._data
    assert nd is not ndarray  # copy is False, we should have the same object
    # initialization with an NDArray object with copy
    nd = NDArray(ndarray, copy=True)
    assert_array_equal(nd.data, refarray)
    # by default, we do not copy but here copy is True
    assert nd.data is not ndarray.data
    # changing dims name
    nd = NDArray(
        [1, 2, 3],
        quantity="intensity",
        dims=["q"],
    )
    assert nd.dims == ["q"]
    # timedelta
    nd = NDArray([1, 2, 3])
    nd1 = nd.astype("timedelta64[s]")
    assert nd1.units == ur("s")
    nd = NDArray([1, 2, 3], dtype="timedelta64[ms]")
    assert nd.units == ur("ms")


def test_ndarray_iter(ndarray):
    for i, row in enumerate(ndarray):
        assert row == ndarray[i]


def test_ndarray_len(ndarray):
    assert len(ndarray) == ndarray.shape[0]


def test_ndarray_setitem(ndarrayunit):
    nd = ndarrayunit.copy()
    # set item
    nd[1] = 2.0  # assume same units as the array
    assert nd[1, 0].value == 2 * ur("m/s")
    # set item with quantity
    nd[1] = 3.0 * ur("km/hr")
    assert nd[1, 0].value == (30 / 36) * ur("m/s")
    with pytest.raises(DimensionalityError):
        nd[1] = 3.0 * ur("km/h")  # here there is a mistake (h is the planck
        # constant not a time, so dimensionality is not correct
    # set item with fancy indexing
    nd[[0, 1, 5]] = 2
    assert np.all(nd[5].data == np.array([2] * 8))
    # set with a quantity


def test_ndarray_repr_html(ndarray):
    # test repr_html
    assert "<table style='background:transparent'>" in ndarray._repr_html_()


def test_ndarray_str(ndarrayunit):
    nd = ndarrayunit.copy()
    nd.name = "intensity"
    assert str(nd).startswith(" name: intensity")


def test_ndarray_astype(ndarray):
    nd = NDArray([1, 2, 4])
    assert nd.dtype == np.dtype(int)
    nd1 = nd.astype(float)
    assert nd1.dtype == np.dtype(float)
    assert nd1 is not nd
    assert nd1.data is not nd.data
    with pytest.raises(CastingError):
        nd1.astype("int")
    nd2 = nd.astype(float, inplace=True)
    assert nd2 is nd
    assert nd2.data is nd.data
    nd = NDArray()
    nd1 = nd.astype(float)
    assert nd1 is nd


def test_ndarray_copy(ndarray):
    nd = ndarray.copy()
    nd1 = nd.copy()
    assert nd1 is not nd
    assert nd1.data is not nd.data
    assert_array_equal(nd1, nd)
    assert nd1.name == nd.name == "value"  # default name
    d2 = copy(nd)
    assert d2 == nd
    d3 = deepcopy(nd)
    assert d3 == nd
    nd2 = nd1
    assert nd2 is nd1
    nd3 = nd1.copy()
    assert repr(nd3) == repr(nd1)
    assert nd3 is not nd1
    nd4 = nd1.copy(keepname=False)
    assert repr(nd4) == "NDArray (value): [float64] unitless (shape: (y:10, x:8))"


def test_ndarray_data(ndarray):
    nd = ndarray.copy()
    assert_array_equal(nd.data, nd._data)
    assert nd.data is nd._data
    nd.data = None
    assert nd.data is None
    assert not nd.has_data
    nd.data = [1, 2, 3]  # put some data
    assert_array_equal(nd.data, np.array([1, 2, 3]))
    assert nd.dtype in TYPE_INTEGER
    assert nd.has_data
    assert_array_equal(nd.m, nd.magnitude)
    assert_array_equal(nd.m, nd.data)
    nd.data = [1.0 * ur("Hz"), 2.0 * ur("Hz")]
    assert_array_equal(nd.data, [1, 2])
    nd.data = [None, 1]
    assert_array_equal(nd.data, [np.nan, 1])
    with pytest.raises(CastingError):
        nd.data = ["x", "y"]


def test_ndarray_dimensionless(ndarray):
    nd = ndarray.copy()
    assert nd.is_unitless  # no units
    assert not nd.is_dimensionless  # no unit so dimensionless has no sense
    with assert_produces_warning(
        raise_on_extra_warnings=False,
        match="There is no units for this NDArray!",
    ):
        # try to change to an array with units
        nd.to("m")  # should not change anything (but raise a warning)
    nd._units = ur.absorbance
    assert not nd.is_unitless  # no units
    assert nd.is_dimensionless
    nd._units = ur.km
    assert not nd.is_dimensionless


def test_ndarray_dims(ndarray):
    nd = ndarray.copy()
    assert nd.name == "value"  # default name for the dataset
    assert nd.dims == ["y", "x"]  # default name for dims
    nd.dims = ["yoyo", "xaxa"]  # user named axis
    assert nd.dims == ["yoyo", "xaxa"]
    nd1 = nd.copy()
    assert nd1.dims == nd.dims  # name should be copied


def test_ndarray_get_axis(ndarray):
    nd = ndarray.copy()
    assert nd.get_axis(None) == (1, "x")
    assert nd.get_axis(None, allow_none=True) == (None, None)
    axis, dim = nd.get_axis(1)
    assert axis == 1
    assert dim == "x"
    axis, dim = nd.get_axis("y")
    assert axis == 0
    assert dim == "y"
    axis, dim = nd.get_axis("y", negative_axis=True)
    assert axis == -2
    assert dim == "y"
    axis, dim = nd.get_axis("x", "y", negative_axis=True)
    assert axis == [-1, -2]
    assert dim == ["x", "y"]

    # user named axis
    nd.dims = ["yoyo", "xaxa"]
    axis, dim = nd.get_axis("yoyo", negative_axis=True)
    assert axis == -2
    assert dim == "yoyo"

    with pytest.raises(ValueError):
        # axis not exits
        nd.get_axis("notexists", negative_axis=True)


def test_ndarray_has_data():
    nd = NDArray()
    assert not nd.has_data
    nd = NDArray([1, 2, 3])
    assert nd.has_data


def test_ndarray_has_units(ndarray, ndarrayunit):
    assert not ndarray.has_units
    assert ndarrayunit.has_units


def test_ndarray_implements(ndarray):
    assert ndarray._implements("NDArray")
    assert ndarray._implements() == "NDArray"


def test_ndarray_is_1d(ndarray):
    assert not ndarray.is_1d
    assert ndarray[0].is_1d


def test_ndarray_is_dt64():
    nd = NDArray(["2022", "2023", "2024"], dtype="M")
    assert nd.is_dt64
    with pytest.raises(CastingError):
        # str are not accepted in NDArray data
        NDArray(["2022", "2023", "2024"])


def test_ndarray_is_float(ndarray):
    nd = ndarray.copy()
    assert nd.is_float
    with pytest.raises(CastingError):
        nd.astype("int64")  # cannot cast with safe option
    nd1 = nd.astype("int64", casting="unsafe")
    assert nd1.is_integer


def test_ndarray_is_units_compatible():
    nd1 = NDArray([1.0, 2.0], units="meters")
    nd2 = NDArray([1.0, 3.0], units="seconds")
    assert not nd1.is_units_compatible(nd2)
    nd1.ito("minutes", force=True)
    assert nd1.is_units_compatible(nd2)


def test_ndarray_sort():
    nd = NDArray(
        np.linspace(4000, 1000, 10),
        units="s",
        quantity="wavelength",
    )
    nd1 = nd._sort()
    assert nd1.data[0] == 1000
    assert nd1.data[-1] == nd.data[0]

    # check inplace
    nd2 = nd._sort(inplace=True)
    assert nd.data[0] == nd2.data[0] == 1000
    assert nd2 is nd

    # check descend
    nd._sort(descend=True, inplace=True)
    assert nd.data[0] == 4000
    nd._sort(descend=False, inplace=True)
    assert nd.data[0] == 1000


def test_ndarray_ito_and_to(ndarrayunit):
    nd = ndarrayunit.copy()
    assert nd.units == ur("m/s")
    x = nd[0, 0].data
    nd.ito("km/s")
    assert nd.units == ur("km/s")
    assert nd[0, 0].data == x / 1000.0
    with pytest.raises(DimensionalityError):
        nd.ito("kg/s")
    nd.ito("kg/min", force=True)
    assert nd.units == ur("kg/min")
    x = nd[0, 0].data
    nd.ito_base_units()
    assert nd.units == ur("kg/s")
    assert nd[0, 0].data == x / 60.0
    nd.ito("J/kg", force=True)
    nd.ito_reduced_units()
    assert nd.units == ur("m^2/s^2")
    nd.ito(None)  # no change
    assert nd.units == ur("m^2/s^2")
    nd.ito(None, force=True)
    assert nd.is_unitless
    # NMR
    nd.meta.larmor = 100 * ur.MHz
    nd.ito("ppm", force=True)
    assert nd.units == ur.ppm
    nd.ito("MHz")
    assert nd.units == ur.MHz
    # Transmittance/absorbance
    nd.meta.larmor = None
    nd.ito("absorbance", force=True)
    nd.quantity = "absorbance"
    nd1 = nd.to("transmittance")
    nd1.ito("absolute_transmittance")
    nd2 = nd1.to("absorbance")
    assert_array_almost_equal(nd2.data, nd.data)
    nd1.ito("1/s", force=True)
    nd1.ito("1/cm")
    assert nd1.quantity == "wavenumber"
    nd1.ito("cm")
    assert nd1.quantity == "wavelength"
    nd1.ito("eV")
    assert nd1.quantity == "energy"
    nd1.ito("GHz")
    assert nd1.quantity == "frequency"

    nd1.ito("km/hour", force=True)
    assert nd1.units == ur.km / ur.hour
    nd2 = nd1.to_base_units()
    assert nd2.units == ur("m/s")
    nd3 = nd2.to_reduced_units()
    assert nd3.units == ur("m/s")


def test_ndarray_limits():
    nd = NDArray([4, 5, 6, 3, 2, 1])
    assert_array_equal(nd.limits, np.array([1, 6]))
    assert_array_equal(nd.limits, nd.roi)


def test_ndarray_meta(ndarray):
    nd = ndarray.copy()
    nd.meta = {"essai": "un essai"}
    assert nd.meta.essai == "un essai"
    nd.meta = {"essai2": "un essai2"}
    assert list(nd.meta.keys()) == ["essai", "essai2"]


def test_ndarray_roi(ndarrayunit):
    nd = ndarrayunit.copy()
    l = [0.1, 0.9]
    nd.roi = l
    assert_array_equal(nd.roi, np.array(l))
    assert (nd.roi_values == np.array(l) * nd.units).all()
    nd.ito("km/s")
    assert_array_almost_equal(nd.roi_values.m, np.array(l) / 1000.0)


def test_ndarray_roi_values(ndarray, ndarrayunit):
    nd = ndarray.copy()
    assert_array_equal(nd.roi_values, nd.limits)
    nd = ndarrayunit.copy()
    assert nd.roi_values[0] == nd.limits[0] * nd.units
    nd.roi = [nd.limits * 0.1, nd.limits * 0.9]
    assert_array_equal(nd.roi, np.array([nd.limits * 0.1, nd.limits * 0.9]))


def test_ndarray_squeeze(ndarray):
    nd = ndarray.copy()
    assert nd.shape == (10, 8)
    d = nd[..., 0]
    assert d.shape == (10, 1)
    d = d.squeeze()
    assert d.shape == (10,)
    d = nd[0]
    assert d.shape == (1, 8)
    d = d.squeeze()
    assert d.shape == (8,)
    d = nd[..., 0]
    d = np.squeeze(d)
    assert d.shape == (10,)
    nd = NDArray()
    assert nd._squeeze_ndim == 0
    nd = NDArray(np.ones((1, 3, 1, 2)), name="value")
    assert (
        repr(nd) == "NDArray (value): [float64] unitless (shape: (u:1, z:3, y:1, "
        "x:2))"
    )
    assert nd._squeeze_ndim == 2
    nd1 = nd.squeeze()
    assert repr(nd1) == "NDArray (value): [float64] unitless (shape: (z:3, x:2))"
    nd2 = nd1.squeeze()
    nd2, idx = nd1.squeeze(return_index=True)
    assert not idx
    nd1, idx = nd.squeeze("u", "y", return_index=True)
    assert repr(nd1) == "NDArray (value): [float64] unitless (shape: (z:3, x:2))"
    assert idx == (0, 2)
    nd1 = nd.squeeze(("u", "y"))
    assert repr(nd1) == "NDArray (value): [float64] unitless (shape: (z:3, x:2))"
    nd1 = nd.squeeze(dims=("u", "y"))
    assert repr(nd1) == "NDArray (value): [float64] unitless (shape: (z:3, x:2))"
    nd2 = nd.squeeze(dim="y")
    assert repr(nd2) == "NDArray (value): [float64] unitless (shape: (u:1, z:3, x:2))"
    nd3 = nd.squeeze(dim=0)
    assert repr(nd3) == "NDArray (value): [float64] unitless (shape: (z:3, y:1, x:2))"
    nd4 = nd.squeeze(axis=0)
    assert repr(nd4) == "NDArray (value): [float64] unitless (shape: (z:3, y:1, x:2))"
    nd5 = nd.squeeze(keepdims=(2,))
    assert repr(nd5) == "NDArray (value): [float64] unitless (shape: (z:3, y:1, x:2))"


def test_ndarray_swapdims():
    nd = NDArray(np.ones((1, 3, 1, 2)), name="value")
    assert nd.dims == ["u", "z", "y", "x"]
    nd1 = nd.swapdims(0, 1)
    assert nd1.dims == ["z", "u", "y", "x"]
    nd2 = nd1.swapdims("y", "z")
    assert nd2.dims == ["y", "u", "z", "x"]
    assert nd2.shape == (1, 1, 3, 2)
    nds = nd[0, 0, 0].squeeze()
    assert nds.ndim == 1
    assert nds.swapdims(0, 1) == nds


def test_ndarray_transpose():
    nd = NDArray(np.ones((1, 3, 1, 2)), name="value")
    assert (
        repr(nd) == "NDArray (value): [float64] unitless (shape: (u:1, z:3, y:1, x:2))"
    )
    nd1 = nd.T
    assert (
        repr(nd1) == "NDArray (value): [float64] unitless (shape: (x:2, y:1, z:3, u:1))"
    )
    nd2 = nd.transpose()
    assert nd1 == nd2
    assert nd.shape == nd2.shape[::-1]
    # cannot transpose 1D
    nd = NDArray([1, 2, 3])
    assert_array_equal(nd.T.data, nd.data)


def test_ndarray_timezone():
    nd = NDArray(np.ones((1, 3, 1, 2)), name="value")
    assert nd.timezone is not None
    assert str(nd.timezone) == nd.local_timezone
    nd.timezone = "UTC"
    assert nd.timezone != nd.local_timezone
    with pytest.raises(UnknownTimeZoneError):
        nd.timezone = "XXX"


def test_ndarray_units():
    nd = NDArray([1, 2, 3], units="Hz")
    nd.units = 1 * ur.MHz
    with pytest.raises(TypeError):
        # units incompatible
        nd.units = "km/hours"
    nd.units = ur.cm


def test_ndarray_urray(ndarrayunit):
    nd = NDArray()
    assert nd.uarray is None
    assert (ndarrayunit.uarray == ndarrayunit.values).all()
    assert ndarrayunit[1, 1].uarray.squeeze()[()] == ndarrayunit[1, 1].values


def test_deprecated():
    nd = NDArray((1, 2, 3))
    with assert_produces_warning(DeprecationWarning, check_stacklevel=False):
        nd.title = "length"
        assert nd.title == "length"


# test docstring
def test_ndarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.ndarray"
    result = td.check_docstrings(
        module, obj=spectrochempy.core.dataset.ndarray.NDArray, exclude=["SA01", "EX01"]
    )


# bug fix
def test_ndarray_issue_23():
    nd = NDArray(np.ones((10, 10)))
    assert nd.shape == (10, 10)
    assert nd.dims == ["y", "x"]
    # slicing
    nd1 = nd[1]
    assert nd1.shape == (1, 10)
    assert nd1.dims == ["y", "x"]
    # transposition
    ndt = nd1.T
    assert ndt.shape == (10, 1)
    assert ndt.dims == ["x", "y"]
    # squeezing
    nd2 = nd1.squeeze()
    assert nd2.shape == (10,)
    assert nd2.dims == ["x"]

    nd = NDArray(np.ones((10, 10, 2)))
    assert nd.shape == (10, 10, 2)
    assert nd.dims == ["z", "y", "x"]
    # slicing
    nd1 = nd[:, 1]
    assert nd1.shape == (10, 1, 2)
    assert nd1.dims == ["z", "y", "x"]
    # transposition
    ndt = nd1.T
    assert ndt.shape == (2, 1, 10)
    assert ndt.dims == ["x", "y", "z"]
    # squeezing
    nd2 = nd1.squeeze()
    assert nd2.shape == (10, 2)
    assert nd2.dims == ["z", "x"]


def test_ndarray_issue_13(ndarrayunit):
    nd = ndarrayunit[0]

    assert isinstance(nd[0], NDArray)

    # reproduce our bug (now solved)
    nd[0] = Quantity("10 cm.s^-1")

    with pytest.raises(DimensionalityError):
        nd[0] = Quantity("10 cm")


# ################### #
# TEST NDComplexArray #
# ################### #

# ------------------------------------------------------------------
# Fixtures: Some NDComplex's array
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def ndarraycplx():
    # return a complex ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.complex128, copy=True).copy()


@pytest.fixture(scope="module")
def ndarrayquaternion():
    # return a quaternion ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.quaternion, copy=True).copy()


# test docstring
def test_ndcomplexarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.ndarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.ndarray.NDComplexArray,
        exclude=["SA01", "EX01"],
    )


def test_ndcomplexarray_init():
    # test with complex data in the last dimension
    # A list
    nd = NDComplexArray()
    assert not nd.has_complex_dims
    assert not nd.is_complex
    assert not nd.is_hypercomplex
    arr = [1.0j, 2.0j, 3.0j]
    nd = NDComplexArray(arr)
    assert nd.dtype == np.complex128
    assert nd.has_complex_dims
    assert nd.shape == (3,)
    assert nd.size == 3
    assert (
        repr(nd) == "NDComplexArray (value): [complex128]"
        " unitless (size: 3(complex))"
    )
    print(nd)

    arr = [[1 + 1.0j, 2], [3 + 2.0j, 4]]
    nd = NDComplexArray(arr)
    assert nd.dtype == np.complex128
    assert nd.has_complex_dims
    assert nd.shape == (2, 2)
    assert nd.size == 4
    assert (
        repr(nd) == "NDComplexArray (value): [complex128]"
        " unitless (shape: (y:2, x:2(complex)))"
    )

    # a numpy array
    arr = np.array([[1, 2], [3, 4]]) * np.exp(0.1j)
    nd = NDComplexArray(arr)
    assert nd.dtype == np.complex128
    assert nd.has_complex_dims
    assert nd.shape == (2, 2)
    assert nd.size == 4

    print(nd.data)
    print(nd)

    # init with another NDArray or NDComplexArray
    # we should not be able to put complex data in a NDArray
    arr = [1.0j, 2.0j, 3.0j]
    with pytest.raises(CastingError):
        nd = NDArray(arr)
    nd1 = NDComplexArray(arr)
    nd2 = NDComplexArray(nd1)
    assert nd1 is not nd2  # shallow copy only data
    assert nd1.data is nd2.data  # by default copy is false

    # ndComplexarray can also accept other type at initialisation
    nd = NDComplexArray([25])
    assert nd.data == np.array([25])
    assert nd.data.dtype in TYPE_INTEGER

    with pytest.raises(CastingError):
        # must have even number of colums
        nd = NDComplexArray([25], dtype=np.complex128)
    nd = NDComplexArray([25, 1], dtype=np.complex128)
    assert_array_equal(nd.data, np.array([25 + 1.0j]))
    assert nd.data.dtype in TYPE_COMPLEX

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    nd = NDComplexArray(
        d,
        units=ur.Hz,
        dtype=typequaternion,
    )
    assert nd.shape == (2, 3)
    assert (
        repr(nd) == "NDComplexArray (value):"
        " [quaternion] Hz (shape: (y:2(complex), x:3(complex)))"
    )

    d = np.random.random((3, 4))
    with pytest.raises(CastingError):  # not even
        NDComplexArray(d, dtype=typequaternion)

    d = np.random.random((3, 3)) * np.exp(0.1j)
    with pytest.raises(CastingError):
        NDComplexArray(d, dtype=typequaternion)


def test_ndcomplexarray_astype():
    nd = NDComplexArray()
    nd1 = nd.astype("complex")  # nothing happen, no data
    assert nd == nd1
    np.random.seed(12345)
    d = np.random.random((4, 4))
    nd = NDComplexArray(
        d,
        units=ur.Hz,
    )
    assert nd.shape == (4, 4)
    assert nd.dtype.kind == "f"
    with pytest.raises(CastingError):
        nd.astype("int")
    nd0 = nd.astype("int", casting="unsafe")
    assert nd0.shape == (4, 4)
    nd1 = nd.astype("complex")
    assert nd1.shape == (4, 2)
    assert nd1.dtype.kind == "c"

    nd2 = nd.astype("quaternion")
    assert nd2.shape == (2, 2)
    assert nd2.dtype.kind == "V"
    assert nd2.dtype == typequaternion


def test_ndcomplexarray_components():
    d = np.arange(24).reshape(3, 2, 4)
    nd = NDComplexArray(d, dtype=np.complex128)
    assert nd.shape == (3, 2, 2)
    assert nd.is_complex
    ndr = nd.real
    assert ndr.is_real
    ndr = nd.R
    assert ndr.is_real
    ndi = nd.imag
    assert ndi.is_real
    ndi = nd.I
    assert ndi.is_real
    assert_array_equal(nd.RR, nd.real)
    assert_array_equal(nd.RI, nd.imag)
    with pytest.raises(CastingError):
        # only 2D
        # TODO: change this
        NDComplexArray(d, dtype=np.quaternion)
    nd = NDComplexArray(d[0], dtype=np.quaternion)
    assert nd.shape == (1, 2)
    assert nd.is_hypercomplex
    ndr = nd.real
    assert ndr.is_real
    ndr = nd.R
    assert ndr.is_real
    ndi = nd.imag
    assert ndi.is_hypercomplex
    ndi = nd.I
    assert ndi.is_hypercomplex
    assert nd.IR.is_real
    assert nd.RR.is_real
    assert nd.II.is_real
    assert nd.RI.is_real

    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)
    d3 = NDComplexArray(d)
    new = d3.copy()
    new.data = d3.real.data + 1j * d3.imag.data
    assert_equal(d3.data, new.data)

    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)
    d3 = NDComplexArray(d, dtype=typequaternion)
    d3r = d3.real
    assert d3r.dtype == np.float64
    assert d3r.shape == (1, 2)
    d3i = d3.imag
    assert d3i.dtype == typequaternion


def test_ndcomplexarray_copy(ndarraycplx, ndarrayquaternion):
    nd1 = ndarraycplx
    nd2 = copy(ndarraycplx)
    assert nd2 is not nd1
    assert nd2.shape == nd1.shape
    assert nd2.is_complex == nd1.is_complex
    assert nd2.ndim == nd1.ndim

    nd1 = ndarrayquaternion
    nd2 = copy(ndarrayquaternion)
    assert nd2 is not nd1
    assert nd2.shape == nd1.shape
    assert nd2.is_complex == nd1.is_complex
    assert nd2.ndim == nd1.ndim


def test_ndcomplexarray_limits(ndarraycplx, ndarrayquaternion):
    nd = NDComplexArray()
    assert nd.limits is None
    assert ndarraycplx.limits == ndarraycplx.real.limits
    assert ndarrayquaternion.limits is not None


def test_ndcomplexarray_set_complex():
    d = np.arange(24).reshape(3, 2, 4)
    nd = NDComplexArray(d)
    assert nd.shape == (3, 2, 4)
    nd1 = nd.set_complex()
    assert nd1.shape == (3, 2, 2)
    nd2 = nd1.set_complex()
    assert nd2 == nd1


def test_ndcomplexarray_set_hypercomplex():
    d = np.arange(24).reshape(3, 2, 4)
    d = as_quat_array(d)  # create a quaternion array
    nd = NDComplexArray(d)
    assert nd.shape == (3, 2)
    assert_array_equal(nd.real.data, [[0, 4], [8, 12], [16, 20]])
    nd1 = NDComplexArray(d)
    nd1 = nd1.set_hypercomplex()  # already hypercomplex
    assert_array_equal(nd1.real.data, nd.real.data)

    # use of _make_quaternion
    d = np.arange(24).reshape(3, 2, 4)
    nd = NDComplexArray(d)
    assert nd.shape == (3, 2, 4)
    nd1 = nd.set_hypercomplex(quat_array=True)
    assert nd1.shape == (3, 2)
    assert_array_equal(nd1.real.data, [[0, 4], [8, 12], [16, 20]])
    with pytest.raises(CastingError):
        nd2 = nd[..., :3]  # not enough data in the last dimension
        nd2.set_hypercomplex(quat_array=True)

    # use of interlaced float array
    with pytest.raises(CastingError):
        # too many dimensions
        nd.set_hypercomplex()

    d = np.arange(25).reshape(5, 5)
    nd = NDComplexArray(d)
    assert not nd.is_complex
    assert not nd.is_hypercomplex
    with pytest.raises(CastingError):
        # not suitable shape (last dimension)
        nd.set_hypercomplex()
    nd = nd[..., :4]
    with pytest.raises(CastingError):
        # not suitable shape (first dimension)
        nd.set_hypercomplex()
    nd = nd[:4]
    nd1 = nd.set_hypercomplex()
    assert nd1.is_hypercomplex
    assert nd1.shape == (2, 2)


def test_ndcomplexarray_setitem(ndarraycplx, ndarrayquaternion):
    nd = ndarraycplx.copy()
    # set item
    nd[1] = 2.0 + 1.0j
    assert nd[1, 0].value == (2.0 + 1.0j) * ur("meter / second")

    # set item with fancy indexing
    nd[[0, 1, 5]] = 2.0 + 1.0j
    assert np.all(nd[5].data == np.array([2.0 + 1.0j] * 4))

    with pytest.raises(TypeError):
        # cannot cast to complex128
        nd[1] = np.quaternion(1, 0, 1, 0)

    nd = ndarrayquaternion.copy()
    nd[1] = np.quaternion(1, 0, 1, 0)
    assert nd[1, 0].data == np.quaternion(1, 0, 1, 0)
    # TODO: comparison of quaternion quantity do not work. Pint problem
    # assert nd[1, 0].value == np.quaternion(1, 0, 1, 0)*ur('meter / second')
    assert nd[1, 0].data == np.quaternion(1, 0, 1, 0)


def test_ndcomplexarray_slicing_byindex_quaternion(ndarrayquaternion):
    ndc = ndarrayquaternion.copy()
    ndc1 = ndc[1, 1].real
    assert_approx_equal(ndc1.values.magnitude, 4.646475973719301, 3)


def test_ndcomplexarray_swapdims_transpose():
    # quaternion
    d = np.arange(24).reshape(3, 2, 4)
    d = as_quat_array(d)
    nd = NDComplexArray(d)
    nd1 = nd.swapdims(1, 0)
    assert nd1.shape == (2, 3)
    assert_array_equal(nd1.real.data, [[0, 8, 16], [4, 12, 20]])
    assert nd1[0, 0].values == quaternion(0, 2, 1, 3)

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)

    d3 = NDComplexArray(
        d,
        units=ur.Hz,
        dtype=typequaternion,
    )  # quaternion with units

    assert d3.shape == (2, 3)
    assert d3._data.shape == (2, 3)
    assert d3.has_complex_dims
    assert d3.is_hypercomplex

    w, x, y, z = as_float_array(d3.data).T

    d4 = d3.swapdims(0, 1)

    assert d4.shape == (3, 2)
    assert d4._data.shape == (3, 2)
    assert d4.has_complex_dims
    assert d4.is_hypercomplex

    wt, yt, xt, zt = as_float_array(d4.data).T
    assert_array_equal(xt, x.T)
    assert_array_equal(yt, y.T)
    assert_array_equal(zt, z.T)
    assert_array_equal(wt, w.T)

    d5 = d3.T
    assert_array_equal(d5.data, d4.data)


# ###########################
# TEST NDMaskedComplexArray #
# ###########################


@pytest.fixture(scope="module")
def ndarraymask():
    # return a simple ndarray with some data, units, and masks
    return NDMaskedComplexArray(ref_data, mask=ref_mask, units="m/s", copy=True)


# test docstring
def test_ndmaskedcomplexarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.ndarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.ndarray.NDMaskedComplexArray,
        exclude=["SA01", "EX01"],
    )


def test_ndmaskedcomplexarray_get_and_setitem(ndarray):
    nd = NDMaskedComplexArray(ndarray.copy())
    # set item mask
    nd[1] = MASKED
    assert nd.is_masked
    nd[1, 0] = NOMASK
    assert not nd[1, 0].mask
    nd[1, 0] = 1.0
    assert not nd[1, 0].mask
    assert nd[1, 0] == 1.0
    nd.remove_masks()
    assert not nd.is_masked
    assert nd[1, 0] == 1.0
    assert not nd[1, 0].mask
    assert nd.__getitem__((1, 0), return_index=True)[1] == (
        slice(1, 2, 1),
        slice(0, 1, 1),
    )


def test_ndmaskedcomplexarray_init_complex_with_mask():
    # test with complex with mask and units

    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)

    nd = NDMaskedComplexArray(
        d, units=ur.Hz, mask=[[False, True], [False, False]]
    )  # with units & mask
    assert nd.shape == (2, 2)
    assert nd._data.shape == (2, 2)
    assert nd.data.shape == (2, 2)
    assert nd.size == 4

    assert (nd.real.data == d.real).all()
    assert np.all(nd.data.real == d.real)
    assert (nd.imag.data == d.imag).all()
    assert np.all(nd.data.imag == d.imag)

    assert nd.dtype == np.complex128
    assert nd.has_complex_dims
    assert nd.mask.shape[-1] == nd.shape[-1]
    ndRR = nd.component("RR")
    assert not ndRR.has_complex_dims
    assert ndRR._data.shape == (2, 2)
    assert ndRR._mask.shape == (2, 2)

    assert isinstance(nd[1, 1].values, Quantity)
    assert nd[1, 1].values.magnitude == d[1, 1]


def test_ndmaskedcomplexarray_swapdims():

    d = np.arange(24).reshape(3, 2, 4)
    d = as_quat_array(d)
    nd = NDComplexArray(d)
    nd1 = nd.swapdims(1, 0)
    assert nd1.shape == (2, 3)
    assert_array_equal(nd1.real.data, [[0, 8, 16], [4, 12, 20]])
    assert nd1[0, 0].values == quaternion(0, 2, 1, 3)

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d3 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
        ],
    )  # with units & mask
    assert d3.shape == (4, 3)
    assert d3._data.shape == (4, 3)
    assert d3.has_complex_dims
    assert not d3.is_hypercomplex
    assert d3.dims == ["y", "x"]
    d4 = d3.swapdims(0, 1)
    assert d4.dims == ["x", "y"]
    assert d4.shape == (3, 4)
    assert d4._data.shape == (3, 4)
    assert d4.has_complex_dims
    assert not d4.is_hypercomplex

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d0 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[[False, True, False], [True, False, False]],
        dtype=typequaternion,
    )  # with units & mask
    assert d0.shape == (2, 3)
    assert (
        repr(d0) == "NDComplexArray (value): "
        "[quaternion] Hz (shape: (y:2(complex), x:3(complex)))"
    )
    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d3 = NDComplexArray(
        d,
        units=ur.Hz,
        mask=[
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
        ],
    )  # with units & mask
    assert d3.shape == (4, 3)
    assert d3._data.shape == (4, 3)
    assert d3.has_complex_dims
    assert not d3.is_hypercomplex
    assert d3.dims == ["y", "x"]
    d4 = d3.swapdims(0, 1)
    assert d4.dims == ["x", "y"]
    assert d4.shape == (3, 4)
    assert d4._data.shape == (3, 4)
    assert d4.has_complex_dims
    assert not d4.is_hypercomplex


def test_ndmaskedcomplexarray_uarray(ndarraymask):
    nd = NDMaskedComplexArray()
    assert nd.uarray is None
    assert nd.masked_data is None
    assert nd.values is None
    assert (ndarraymask.uarray == ndarraymask.values).all()
    assert isinstance(ndarraymask.masked_data, np.ndarray)
    assert ndarraymask.masked_data.mask[0, 0]
    assert ndarraymask.is_masked
    ndarraymask.remove_masks()
    assert not ndarraymask.masked_data.mask
    assert not ndarraymask.is_masked


# ######################
# TEST NDLabelledArray #
# ######################
# test docstring
def test_ndlabelledarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.ndarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.ndarray.NDLabelledArray,
        exclude=["SA01", "EX01"],
    )


def test_ndlabelledarray_init():
    nd = NDLabelledArray(labels=list("abcdefghij"), title="labelled")
    assert nd.is_labeled
    assert nd.ndim == 1
    assert nd.shape == (10,)


def test_ndlabelledarray_getitem():
    # slicing only-label array

    nd = NDLabelledArray(labels=list("abcdefghij"), title="labelled")
    assert nd[1].labels == ["b"]
    assert nd[1].values == "b"
    assert nd["b"].values == "b"
    assert nd["c":"d"].shape == (2,)
    assert_array_equal(nd["c":"d"].values, np.array(["c", "d"]))


def test_ndlabelledarray_sort():
    # labels and sort

    nd = NDLabelledArray(
        np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        name="wavelength",
    )

    assert nd.is_labeled

    d1 = nd._sort()
    assert d1.data[0] == 1000
    assert d1.data[-1] == nd.data[0]

    # check inplace
    d2 = nd._sort(inplace=True)
    assert nd.data[0] == d2.data[0] == 1000
    assert d2 is nd

    # check descend
    nd._sort(descend=True, inplace=True)
    assert nd.data[0] == 4000

    # check sort using label
    d3 = nd._sort(by="label", descend=True)
    assert d3.labels[0] == "j"
    assert d3 is not nd

    # multilabels
    # add a row of labels to d0
    nd.labels = "bc cd de ef ab fg gh hi ja ij ".split()

    d1 = nd._sort()
    assert d1.data[0] == 1000
    assert_array_equal(d1.labels[0], ["j", "ij"])

    d1._sort(descend=True, inplace=True)
    assert d1.data[0] == 4000
    assert_array_equal(d1.labels[0], ["a", "bc"])

    d1 = d1._sort(by="label[1]", descend=True)
    assert np.all(d1.labels[0] == ["i", "ja"])

    # other way
    d2 = d1._sort(by="label", pos=1, descend=True)
    assert np.all(d2.labels[0] == d1.labels[0])

    d3 = d1.copy()
    d3._labels = None
    with pytest.raises(KeyError):
        # no label!
        d3._sort(by="label", pos=1, descend=True)
