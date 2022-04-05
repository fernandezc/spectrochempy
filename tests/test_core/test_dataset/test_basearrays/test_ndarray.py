# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

from copy import copy, deepcopy
from os import environ

import numpy as np
import pytest

import spectrochempy
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.common.exceptions import (
    DimensionalityError,
    LabelsError,
    InvalidNameError,
    InvalidDimensionNameError,
    InvalidUnitsError,
    CastingError,
    ShapeError,
    UnknownTimeZoneError,
    UnitWarning,
)
from spectrochempy.core.units import Quantity, ur, Unit
from spectrochempy.core.common.constants import (
    INPLACE,
    TYPE_FLOAT,
    TYPE_INTEGER,
)
from spectrochempy.utils.testing import (
    RandomSeedContext,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_produces_log_warning,
)

from spectrochempy.utils import check_docstrings as td

typequaternion = np.dtype(np.quaternion)

# ------------------------------------------------------------------
# create reference arrays
# ------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref_mask = ref_data < -4
    ref_mask[0, 0] = True


# --------------------------------------------------------------------------------------
# Fixtures: some NDArray's
# --------------------------------------------------------------------------------------
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
# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause errors when checking docstrings",
)
def test_ndarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.basearrays.ndarray"
    result = td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.basearrays.ndarray.NDArray,
        exclude=["SA01", "EX01"],
    )


def test_ndarray_eq(ndarrayunit, ndarray):
    nd = ndarrayunit.copy()
    nd1 = ndarrayunit.copy()
    assert nd1 == nd
    assert nd1.name == nd.name
    nd1 = ndarrayunit.copy(keepname=False)
    assert nd1.name != nd.name
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
    nd0 = nd[0]
    nd0 = nd[-1]
    assert nd0.ndim == 2
    assert nd0.shape == (1, 8)
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
    assert ndf == nd
    # use with selection from other numpy functions
    nd = ndarray.copy()
    am = np.argmax(nd.data, axis=1)
    assert_array_equal(am, np.array([6, 3, 2, 2, 0, 0, 5, 3, 4, 7]))
    amm = nd.data[..., am]
    assert_array_equal(nd[..., am].data, amm)
    with pytest.raises(IndexError):
        # too many keys
        nd[0, 0, 1]
    with pytest.raises(IndexError):
        # not integer keys
        nd[0.0]
    # slicing with non integer:
    # values or quantity
    d = [np.arange(0, 2, 0.01)]
    # 1D array
    nd = NDArray(d[0])
    assert nd.shape == (200,)
    assert nd[0.01].data == 0.01
    # pseudo 1D-array
    nd = NDArray(d)
    assert nd.shape == (1, 200)
    assert nd[0, 0.01].data == 0.01
    # slice
    nd = NDArray(d[0])
    assert nd[0.01:0.04] == NDArray([0.01, 0.02, 0.03, 0.04])
    assert nd[0.01:0.04:2] == NDArray([0.01, 0.03])  # integer step
    assert nd[[0.00, 0.02, 0.03]] == NDArray([0.00, 0.02, 0.03])
    with pytest.raises(IndexError):
        nd[0.01:0.04:0.005]
    # datetime
    d = np.arange("2020", "2025", 6, dtype="<M8[M]")
    nd = NDArray(d)
    assert nd["2020-06"].data == np.datetime64("2020-07")
    assert nd[np.datetime64("2020-08")].data == np.datetime64("2020-07")
    assert nd["2020-01":"2022-01"].size == 5


def test_ndarray_init(refarray, ndarray):
    # initialisation with null array
    nd = NDArray()
    assert nd._implements("NDArray")
    assert nd._implements() == "NDArray"
    assert isinstance(nd, NDArray)
    assert nd.is_empty
    assert not nd.is_float
    assert not nd.is_integer
    assert nd.limits is None
    assert nd.id.startswith("NDArray_")
    assert not nd.has_data
    assert len(nd) == 0
    assert nd.shape == ()
    assert nd.id.startswith("NDArray")
    assert nd.title == "value"
    assert nd.ndim == 0
    assert nd.size == 0
    assert nd.dtype is None
    assert nd.is_unitless
    assert not nd.dims
    assert hash(nd) is not None
    assert repr(nd) == "NDArray (value): empty"
    # Passing attributes during initialisation.
    nd = NDArray(name="intensity", title="Raman intensity")
    assert nd.name == "intensity"
    assert nd.title == "Raman intensity"
    # Initialisation with a scalar
    nd = NDArray(25)
    assert nd.data == np.array(25)
    assert nd.data.dtype in TYPE_INTEGER
    # Initialisation with a scalar + cast to float
    nd = NDArray(25, dtype="float64")
    assert nd.data == np.array(25.0)
    assert nd.data.dtype in TYPE_FLOAT
    # Initialisation with a title
    nd = NDArray(13.0 * ur.tesla)
    assert nd.data == np.array(13.0)
    assert nd.data.dtype in TYPE_FLOAT
    assert nd.shape == ()
    assert nd.ndim == 0
    assert not nd.dims
    assert nd.units == "tesla"
    assert nd.values == 13.0 * ur.tesla
    # initialisation with a 1D array quantity
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
    assert nd.title == "value"
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
        title="intensity",
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
    assert str(nd).startswith(
        "NDArray intensity(value): [float64] m.s⁻¹ (shape: (y:10, x:8))"
    )
    nd1 = nd.copy()
    nd1._units = (nd1.values / ur("m/s")).units
    assert str(nd1).startswith(
        "NDArray intensity(value): [float64] dimensionless (shape: (y:10, x:8))"
    )
    nd2 = nd.copy()
    nd2._units = (nd2.values / ur("km/s")).units
    assert str(nd2).startswith(
        "NDArray intensity(value): [float64] scaled-dimensionless (0.001) (shape: ("
        "y:10, x:8))"
    )
    assert nd.summary.startswith("[32m         name")


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
    assert nd1.title == nd1.title
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
    assert nd4.name != nd1.name


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
    # try to change to an array with units
    with pytest.raises(DimensionalityError):
        nd.to("m")
    nd._units = ur.absorbance
    assert not nd.is_unitless  # no units
    assert nd.is_dimensionless
    nd._units = ur.km
    assert not nd.is_dimensionless


def test_ndarray_dims(ndarray):
    nd = ndarray.copy()
    assert nd.title == "value"  # default title for the dataset
    assert nd.dims == ["y", "x"]  # default name for dims
    nd.dims = ["yoyo", "xaxa"]  # user named axis
    assert nd.dims == ["yoyo", "xaxa"]
    nd1 = nd.copy()
    assert nd1.dims == nd.dims  # name should be copied


def test_ndarray_get_axis(ndarray):
    nd = ndarray.copy()
    assert nd._get_axis() == (1, "x")
    assert nd._get_axis(None) == (1, "x")
    assert nd._get_axis(None, allow_none=True) == (None, None)
    axis, dim = nd._get_axis(1)
    assert axis == 1
    assert dim == "x"
    axis, dim = nd._get_axis("y")
    assert axis == 0
    assert dim == "y"
    axis, dim = nd._get_axis("y", negative_axis=True)
    assert axis == -2
    assert dim == "y"
    axis, dim = nd._get_axis("x", "y", negative_axis=True)
    assert axis == [-1, -2]
    assert dim == ["x", "y"]
    axis, dim = nd._get_axis(("x", "y"), negative_axis=True)
    assert axis == [-1, -2]
    assert dim == ["x", "y"]
    axis, dim = nd._get_axis(["x", "y"], negative_axis=True)
    assert axis == [-1, -2]
    assert dim == ["x", "y"]
    # user named axis
    nd.dims = ["yoyo", "xaxa"]
    axis, dim = nd._get_axis("yoyo", negative_axis=True)
    assert axis == -2
    assert dim == "yoyo"
    with pytest.raises(ValueError):
        # axis not exits
        nd._get_axis("notexists", negative_axis=True)
    axis, dim = nd._get_axis(0, axis=1)  # conflict between args and kwargs (prioprity
    # to kwargs)
    assert axis == 1
    assert dim == "xaxa"
    with pytest.raises(InvalidDimensionNameError):
        nd._get_dims_index(1.1)
    assert nd._get_dims_index(-1) == (1,)


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
        title="wavelength",
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
    with pytest.raises(DimensionalityError):
        nd.ito(None)  # no change
    assert nd.units == ur("m^2/s^2")
    nd.ito(None, force=True)
    assert nd.is_unitless
    nd.ito("km/hour", force=True)
    assert nd.units == ur.km / ur.hour
    nd = nd.to_base_units()
    assert nd.units == ur("m/s")
    nd = nd.to_reduced_units()
    assert nd.units == ur("m/s")


def test_ndarray_absorbance_conversions():

    # Transmittance/absorbance
    nd = NDArray([100, 50, 25, 1, 0.1], units="transmittance", title="transmittance")
    assert nd.title == "transmittance"
    nd2 = nd.to("absorbance")
    nd3 = nd2.to("absolute_transmittance")
    nd4 = nd3.to("transmittance")
    assert_array_almost_equal(nd4.data, nd.data)


def test_ndarray_spectroscopy_context():
    nd = NDArray([1, 2, 3], units="1/s", title="frequency")
    nd.ito("1/cm")
    assert nd.title == "wavenumber"
    nd.ito("cm")
    assert nd.title == "wavelength"
    nd.ito("eV")
    assert nd.title == "energy"
    nd.ito("GHz")
    assert nd.title == "frequency"


def test_ndarray_limits():
    nd = NDArray([4, 5, 6, 3, 2, 1])
    assert_array_equal(nd.limits, np.array([1, 6]))
    assert_array_equal(nd.limits, nd.roi)


def test_ndarray_name():
    nd = NDArray()
    assert nd.name == nd.id
    nd = NDArray(name="x")
    assert nd.name == "x"
    with pytest.raises(InvalidNameError):
        nd.name = "contain.dot"


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


def test_ndarray_units():
    nd = NDArray([1, 2, 3], units="Hz")
    nd.units = 1 * ur.MHz
    with pytest.raises(InvalidUnitsError):
        # units incompatible
        nd.units = "km/hours"
    nd.units = ur.cm


def test_ndarray_urray(ndarrayunit):
    nd = NDArray()
    assert nd.uarray is None
    assert (ndarrayunit.uarray == ndarrayunit.values).all()
    assert ndarrayunit[1, 1].uarray.squeeze()[()] == ndarrayunit[1, 1].values


# bug fix
def test_ndarray_issue_23():
    nd = NDArray(np.ones((10, 10)))
    assert nd.shape == (10, 10)
    assert nd.dims == ["y", "x"]
    # slicing
    nd1 = nd[1]
    assert nd1.shape == (1, 10)
    assert nd1.dims == ["y", "x"]

    nd = NDArray(np.ones((10, 10, 2)))
    assert nd.shape == (10, 10, 2)
    assert nd.dims == ["z", "y", "x"]
    # slicing
    nd1 = nd[:, 1]
    assert nd1.shape == (10, 1, 2)
    assert nd1.dims == ["z", "y", "x"]


def test_ndarray_issue_13(ndarrayunit):
    nd = ndarrayunit[0]

    assert isinstance(nd[0], NDArray)

    # reproduce our bug (now solved)
    nd[0] = Quantity("10 cm.s^-1")

    with pytest.raises(DimensionalityError):
        nd[0] = Quantity("10 cm")
