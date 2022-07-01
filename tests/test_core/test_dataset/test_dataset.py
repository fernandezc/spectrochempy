#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

import numpy as np
import pytest
from pint.errors import UndefinedUnitError
from quaternion import quaternion
from os import environ

import spectrochempy as scp
from spectrochempy.core.common.complex import as_float_array, as_quat_array
from spectrochempy.core.common.exceptions import (
    UnknownTimeZoneError,
    CastingError,
    ShapeError,
    MissingCoordinatesError,
)
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy.utils.system import get_user_and_node
from spectrochempy.utils.testing import (
    RandomSeedContext,
    assert_array_almost_equal,
    assert_array_equal,
    assert_dataset_almost_equal,
    assert_dataset_equal,
    assert_equal,
    assert_produces_log_warning,
)
from spectrochempy.utils import optional

typequaternion = np.dtype(np.quaternion)

# --------------------------------------------------------------------------------------
# FIXTURES
# --------------------------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2


@pytest.fixture(scope="module")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, comment="An array", copy=True)


coord0_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="module")
def coord0():
    return coord0_.copy()


coord1_ = scp.Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream")


@pytest.fixture(scope="module")
def coord1():
    return coord1_.copy()


coord2_ = scp.Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="module")
def coord2():
    return coord2_.copy()


coord2b_ = scp.Coord(
    data=np.linspace(1.0, 20.0, 3),
    labels=["low", "medium", "high"],
    units="tesla",
    title="magnetic field",
)


@pytest.fixture(scope="module")
def coord2b():
    return coord2b_.copy()


coord0_2_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 9),
    labels=list("abcdefghi"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="module")
def coord0_2():
    return coord0_2_.copy()


coord1_2_ = scp.Coord(
    data=np.linspace(0.0, 60.0, 50), units="s", title="time-on-stream"
)


@pytest.fixture(scope="module")
def coord1_2():
    return coord1_2_.copy()


coord2_2_ = scp.Coord(
    data=np.linspace(200.0, 1000.0, 4),
    labels=["cold", "normal", "hot", "veryhot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="module")
def coord2_2():
    return coord2_2_.copy()


@pytest.fixture(scope="module")
def nd1d():
    # a simple ddataset
    return scp.NDDataset(ref_data[:, 1].squeeze()).copy()


@pytest.fixture(scope="module")
def nd2d():
    # a simple 2D ndarrays
    return scp.NDDataset(ref_data).copy()


@pytest.fixture(scope="module")
def ref_ds():
    # a dataset with coordinates
    return ref3d_data.copy()


@pytest.fixture(scope="module")
def ds1():
    # a dataset with coordinates
    return scp.NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coord2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="module")
def ds2():
    # another dataset
    return scp.NDDataset(
        ref3d_2_data,
        coordset=[coord0_2_, coord1_2_, coord2_2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="module")
def dsm():
    # dataset with coords containing several axis and a mask

    coordmultiple = scp.CoordSet(coord2_, coord2b_)
    return scp.NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coordmultiple],
        mask=ref3d_mask,
        title="absorbance",
        units="absorbance",
    ).copy()


coord_dates = scp.Coord(["2000", "2005", "2010"], title="date", dtype="datetime64[Y]")


coord_pressures = scp.Coord(np.arange(3), title="pressure")


@pytest.fixture(scope="module")
def dsdt64():
    # dataset with coords containing datetime
    coordmultiple = scp.CoordSet([coord_dates, coord_pressures])

    return scp.NDDataset(
        np.ones((3,)),
        coordset=coordmultiple,
        title="absorbance",
        units="absorbance",
    ).copy()


# --------------------------------------------------------------------------------------
# TEST SUITE FOR nddataset
# --------------------------------------------------------------------------------------

# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause errors when checking docstrings",
)
def test_nddataset_docstring():
    from spectrochempy.utils import check_docstrings as chd

    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.core.dataset.nddataset"
    chd.check_docstrings(
        module,
        obj=NDDataset,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


# test minimal constructor and dtypes
adata = (
    [],
    [None, 1.0],
    [np.nan, np.inf],
    [0, 1, 2],
    [0.0, 1.0, 3.0],
    [0.0 + 1j, 10.0 + 3.0j],
    [0.0 + 1j, np.nan + 3.0j],
)


@pytest.mark.parametrize("a", adata)
def test_dataset_init_1D(a):
    # 1D
    for arr in [a, np.array(a)]:
        ds = NDDataset(arr)
        assert ds.size == len(arr)
        shape = (ds.size,) if ds.size != 0 else ()
        assert ds.shape == shape
        if ds.size == 0:
            assert ds.dtype is None
            assert ds.dims == []
        else:
            assert ds.dtype in [np.int64, np.float64, np.complex128]
            assert ds.dims == ["x"]
        # force dtype
        try:
            ds = NDDataset(arr, dtype=np.float32)
            if ds.size == 0:
                assert ds.dtype is None
            else:
                assert ds.dtype == np.float32
        except CastingError as exc:
            if np.asarray(arr).dtype.kind in "cV":
                print(exc.message)
            else:
                raise exc

        assert ds.title == "value"

        assert ds.mask == scp.NOMASK
        assert ds.meta == {}
        assert ds.name.startswith("NDDataset")
        assert ds.author == get_user_and_node()
        assert ds.description == ""
        assert ds.history == []


arrdata = (
    np.array([[1, 1.0], [0, np.nan]]),
    np.random.rand(2, 3).astype("int64"),
    np.random.rand(2, 4),
)


@pytest.mark.parametrize("arr", arrdata)
def test_nddataset_init_2D(arr):
    # 2D
    ds = NDDataset(arr)
    assert ds.size == arr.size
    assert ds.shape == arr.shape
    if ds.size == 0:
        assert ds.dtype is None
        assert ds.dims == []
    else:
        assert ds.dtype in [np.int64, np.float64]
        assert ds.dims == ["y", "x"][-ds.ndim :]

    assert ds.title == "value"

    assert ds.mask == scp.NOMASK
    assert ds.meta == {}
    assert ds.name.startswith("NDDataset")
    assert ds.author == get_user_and_node()
    assert not ds.history
    assert ds.comment == ""
    # force dtype
    ds = NDDataset(arr, dtype=np.float32)
    if ds.size == 0:
        assert ds.dtype is None
    else:
        assert ds.dtype == np.float32
    if ds.shape[-1] % 2 == 0:  # must be even
        ds = NDDataset(arr, dtype=np.complex128)
        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.complex128
    else:

        with pytest.raises(ShapeError):
            ds = NDDataset(arr, dtype=np.complex128)
    if (
        arr.ndim == 2
        and (arr.shape[-1] % 2 == 0 or arr.dtype == np.complex)
        and (arr.shape[-2] % 2) == 0
    ):
        ds = NDDataset(arr, dtype=np.quaternion)

        if ds.size == 0:
            assert ds.dtype is None
        else:
            assert ds.dtype == np.quaternion
    else:

        with pytest.raises(ShapeError):
            ds = NDDataset(arr, dtype=np.quaternion)

    # test units
    ds1 = NDDataset(arr * scp.ur.km)
    ds2 = NDDataset(arr, units=scp.ur.km)
    assert ds1.units == scp.ur.km
    assert ds2.units == scp.ur.km
    assert_dataset_equal(ds1, ds2)
    # masking
    ds1[0] = scp.MASKED
    assert ds1.is_masked
    # init with another masked dataset
    ds2 = NDDataset(ds1)
    assert_dataset_equal(ds1, ds2)
    # check no coordinates
    assert ds2.coordset is None  # no coordinates
    assert ds2.x is None  # no coordinates
    with pytest.raises(AttributeError):
        ds2.t
    # dim attributes
    ds2 = NDDataset(arr, dims=["u", "w"])
    assert ds2.ndim == 2
    assert ds2.dims == ["u", "w"]


def test_nddataset_squeeze(ndarray):

    nd = NDDataset()
    assert nd._squeeze_ndim == 0
    nd = NDDataset(np.ones((1, 3, 1, 2)), name="value")
    assert (
        repr(nd) == "NDDataset (value): [float64] unitless (shape: (u:1, z:3, y:1, "
        "x:2))"
    )
    assert nd._squeeze_ndim == 2
    nd1 = nd.squeeze()
    assert repr(nd1) == "NDDataset (value): [float64] unitless (shape: (z:3, x:2))"
    nd2 = nd1.squeeze()
    nd2, idx = nd1.squeeze(return_index=True)
    assert not idx
    nd1, idx = nd.squeeze("u", "y", return_index=True)
    assert repr(nd1) == "NDDataset (value): [float64] unitless (shape: (z:3, x:2))"
    assert idx == (0, 2)
    nd1 = nd.squeeze(("u", "y"))
    assert repr(nd1) == "NDDataset (value): [float64] unitless (shape: (z:3, x:2))"
    nd1 = nd.squeeze(dims=("u", "y"))
    assert repr(nd1) == "NDDataset (value): [float64] unitless (shape: (z:3, x:2))"
    nd2 = nd.squeeze(dim="y")
    assert repr(nd2) == "NDDataset (value): [float64] unitless (shape: (u:1, z:3, x:2))"
    nd3 = nd.squeeze(dim=0)
    assert repr(nd3) == "NDDataset (value): [float64] unitless (shape: (z:3, y:1, x:2))"
    nd4 = nd.squeeze(axis=0)
    assert repr(nd4) == "NDDataset (value): [float64] unitless (shape: (z:3, y:1, x:2))"
    nd5 = nd.squeeze(keepdims=(2,))
    assert repr(nd5) == "NDDataset (value): [float64] unitless (shape: (z:3, y:1, x:2))"


def test_nddataset_swapdims():
    nd = NDDataset(np.ones((1, 3, 1, 2)), name="value")
    assert nd.dims == ["u", "z", "y", "x"]
    nd1 = nd.swapdims(0, 1)
    assert nd1.dims == ["z", "u", "y", "x"]
    nd2 = nd1.swapdims("y", "z")
    assert nd2.dims == ["y", "u", "z", "x"]
    assert nd2.shape == (1, 1, 3, 2)
    nds = nd[0, 0, 0].squeeze()
    assert nds.ndim == 1
    assert nds.swapdims(0, 1) == nds


def test_nddataset_timezone():
    nd = NDDataset(np.ones((1, 3, 1, 2)), name="value")
    assert nd.timezone is not None
    assert str(nd.timezone) == nd.local_timezone
    nd.timezone = "UTC"
    assert nd.timezone != nd.local_timezone
    with pytest.raises(UnknownTimeZoneError):
        nd.timezone = "XXX"


def test_nddataset_transpose():
    nd = NDDataset(np.ones((1, 3, 1, 2)), name="value")
    assert (
        repr(nd)
        == "NDDataset (value): [float64] unitless (shape: (u:1, z:3, y:1, x:2))"
    )
    nd1 = nd.T
    assert (
        repr(nd1)
        == "NDDataset (value): [float64] unitless (shape: (x:2, y:1, z:3, u:1))"
    )
    nd2 = nd.transpose()
    assert nd1 == nd2
    assert nd.shape == nd2.shape[::-1]
    # cannot transpose 1D
    nd = NDDataset([1, 2, 3])
    assert_array_equal(nd.T.data, nd.data)


def test_nddataset_swapdims_transpose():
    # quaternion
    d = np.arange(24).reshape(3, 2, 4)
    d = as_quat_array(d)
    nd = NDDataset(d)
    nd1 = nd.swapdims(1, 0)
    assert nd1.shape == (2, 3)
    assert_array_equal(nd1.real.data, [[0, 8, 16], [4, 12, 20]])
    assert nd1[0, 0].values == quaternion(0, 2, 1, 3)

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)

    d3 = NDDataset(
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


def test_ndmaskedcomplexarray_swapdims():

    d = np.arange(24).reshape(3, 2, 4)
    d = as_quat_array(d)
    nd = NDDataset(d)
    nd1 = nd.swapdims(1, 0)
    assert nd1.shape == (2, 3)
    assert_array_equal(nd1.real.data, [[0, 8, 16], [4, 12, 20]])
    assert nd1[0, 0].values == quaternion(0, 2, 1, 3)

    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d3 = NDDataset(
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
    d0 = NDDataset(
        d,
        units=ur.Hz,
        mask=[[False, True, False], [True, False, False]],
        dtype=typequaternion,
    )  # with units & mask
    assert d0.shape == (2, 3)
    assert (
        repr(d0) == "NDDataset (value): "
        "[quaternion] Hz (shape: (y:2(complex), x:3(complex)))"
    )
    np.random.seed(12345)
    d = np.random.random((4, 3)) * np.exp(0.1j)
    d3 = NDDataset(
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
    d4 = d3.transpose()
    assert d4.dims == ["x", "y"]
    assert d4.shape == (3, 4)
    assert d4._data.shape == (3, 4)
    assert d4.has_complex_dims
    assert not d4.is_hypercomplex


def test_nddataset_coordset():
    # init coordinates set at NDDataset initialization
    dx = np.random.random((10, 7, 3))
    coord0 = np.arange(10)
    coord1 = np.arange(7)
    coord2 = np.arange(3) * 100.0
    da = NDDataset(
        dx,
        coordset=(coord0, coord1, coord2),
        title="absorbance",
        coordtitles=["wavelength", "time-on-stream", "temperature"],
        coordunits=["cm^-1", "s", "K"],
    )
    assert da.shape == (10, 7, 3)
    assert da.coordset.titles == ["temperature", "time-on-stream", "wavelength"]
    assert da.coordset.names == ["x", "y", "z"]
    assert da.coordunits == ["K", "s", "cm⁻¹"]
    # order of dims follow data shape, but not necessarily the coord list (
    # which is ordered by name)
    assert da.dims == ["z", "y", "x"]
    assert da.coordset.names == sorted(da.dims)
    # transpose
    dat = da.T
    assert dat.dims == ["x", "y", "z"]
    # dims changed but did not change coords order !
    assert dat.coordset.names == sorted(dat.dims)
    assert dat.coordtitles == da.coordset.titles
    assert dat.coordunits == da.coordset.units

    # too many coordinates
    cadd = scp.Coord(labels=["d%d" % i for i in range(6)])
    coordtitles = ["wavelength", "time-on-stream", "temperature"]
    coordunits = ["cm^-1", "s", None]
    daa = NDDataset(
        dx,
        coordset=[coord0, coord1, coord2, cadd, coord2.copy()],
        title="absorbance",
        coordtitles=coordtitles,
        coordunits=coordunits,
    )
    assert daa.coordset.titles == coordtitles[::-1]
    assert daa.dims == ["z", "y", "x"]
    # with a CoordSet
    c0, c1 = scp.Coord(labels=["d%d" % i for i in range(6)]), scp.Coord(
        data=[1, 2, 3, 4, 5, 6]
    )
    cc = scp.CoordSet(c0, c1)
    cd = scp.CoordSet(x=cc, y=c1)
    ds = NDDataset([1, 2, 3, 6, 8, 0], coordset=cd, units="m")
    assert ds.dims == ["x"]
    assert ds.x == cc
    ds.history = "essai: 1"
    ds.history = "try:2"
    # wrong type
    with pytest.raises(TypeError):
        ds.coord[1.3]
    # extra coordinates
    with pytest.raises(AttributeError):
        ds.y
    # invalid_length
    coord1 = scp.Coord(np.arange(9), title="wavelengths")  # , units='m')
    coord2 = scp.Coord(np.arange(20), title="time")  # , units='s')
    with pytest.raises(ValueError):
        NDDataset(np.random.random((10, 20)), coordset=(coord1, coord2))


def test_nddataset_coords_indexer():
    dx = np.random.random((10, 100, 10))
    coord0 = np.linspace(4000, 1000, 10)
    coord1 = np.linspace(0, 60, 10)  # wrong length
    coord2 = np.linspace(20, 30, 10)
    with pytest.raises(ValueError):  # wrong length

        da = NDDataset(
            dx,
            coordset=[coord0, coord1, coord2],
            title="absorbance",
            coordtitles=["wavelength", "time-on-stream", "temperature"],
            coordunits=["cm^-1", "s", "K"],
        )
    coord1 = np.linspace(0, 60, 100)
    da = NDDataset(
        dx,
        coordset=[coord0, coord1, coord2],
        title="absorbance",
        coordtitles=["wavelength", "time-on-stream", "temperature"],
        coordunits=["cm^-1", "s", "K"],
    )
    assert da.shape == (10, 100, 10)
    coords = da.coordset
    assert len(coords) == 3
    assert_array_almost_equal(
        da.coordset[2].data, coord0, decimal=2, err_msg="get axis by index " "failed"
    )
    # we use almost as SpectroChemPy round the coordinate numbers
    assert_array_almost_equal(
        da.coordset["wavelength"].data,
        coord0,
        decimal=2,
        err_msg="get axis by title failed",
    )
    assert_array_almost_equal(
        da.coordset["time-on-stream"].data,
        coord1,
        decimal=4,
        err_msg="get axis by title failed",
    )
    assert_array_almost_equal(da.coordset["temperature"].data, coord2, decimal=3)
    da.coordset["temperature"] += 273.15 * ur.K
    assert_array_almost_equal(
        da.coordset["temperature"].data, coord2 + 273.15, decimal=3
    )


# ======================================================================================================================
# Methods
# ======================================================================================================================


def test_nddataset_str():
    arr1d = NDDataset([1, 2, 3])

    assert "[float64]" in str(arr1d)
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))
    assert str(arr2d) == "NDDataset: [float64] unitless (shape: (y:2, x:2))"


def test_nddataset_str_repr(ds1):
    arr1d = NDDataset(np.array([1, 2, 3]))
    assert repr(arr1d).startswith("NDDataset")
    arr2d = NDDataset(np.array([[1, 2], [3, 4]]))

    assert repr(arr2d).startswith("NDDataset")


def test_nddataset_mask_valid():
    NDDataset(np.random.random((10, 10)), mask=np.random.random((10, 10)) > 0.5)


def test_nddataset_copy_ref():
    """
    Tests to ensure that creating a new NDDataset object copies by *reference*.
    """
    a = np.ones((10, 10))
    nd_ref = NDDataset(a)
    a[0, 0] = 0
    assert nd_ref.data[0, 0] == 0


def test_nddataset_conversion():
    nd = NDDataset(np.array([[1, 2, 3], [4, 5, 6]]))
    assert nd.data.size == 6
    assert nd.data.dtype == np.dtype("float64")


def test_nddataset_invalid_units():
    with pytest.raises(UndefinedUnitError):
        NDDataset(np.ones((5, 5)), units="NotAValidUnit")


def test_nddataset_units(nd1d):
    nd = nd1d.copy()
    nd = np.fabs(nd)
    nd.units = "m"
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m**0.5
    nd.units = "cm"
    _ = np.sqrt(nd)
    nd.ito("m")
    nd2 = np.sqrt(nd)
    assert isinstance(nd2, type(nd))
    assert nd2.data[1] == np.sqrt(nd.data[1])
    assert nd2.units == ur.m**0.5


def test_nddataset_masked_array_input():
    a = np.random.randn(100)
    marr = np.ma.masked_where(a > 0, a)
    nd = NDDataset(marr)
    # check that masks and data match
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)
    # check that they are both by reference
    marr.mask[10] = ~marr.mask[10]
    marr.data[11] = 123456789
    assert_array_equal(nd.mask, marr.mask)
    assert_array_equal(nd.data, marr.data)


def test_nddataset_swapdims2(nd1d, nd2d, ref_ds, ds1):
    nd1 = nd1d.copy()
    nd2 = nd2d.copy()
    nd3 = ds1.copy()
    # swapdims needs 2D at least
    assert nd1.shape == (10,)
    nd1s = nd1.swapdims(1, 0)
    assert_equal(nd1s.data, nd1.data)
    nd2s = nd2.swapdims(1, 0)
    assert nd2s.dims == nd2.dims[::-1]
    assert nd3.shape == ref_ds.shape
    nd3s = nd3.swapdims(1, 0)
    ref = ref_ds
    refs = np.swapaxes(ref, 1, 0)
    assert nd3.shape == ref.shape  # original unchanged
    assert nd3s.shape == refs.shape
    assert nd3s is not nd3
    assert nd3s.dims[:2] == nd3.dims[:2][::-1]
    nd3s = nd3.swapdims(1, 0, inplace=True)
    assert nd3.shape == refs.shape  # original changed
    assert nd3s is nd3  # objects should be the same
    # use of the numpy method
    nd3s = np.swapaxes(nd3, 1, 0)
    assert nd3.shape == refs.shape  # original unchanged (but was already
    # swapped)
    assert nd3s.shape == ref.shape
    assert (
        nd3s is not nd3
    )  # TODO: add check for swapdims of all elements  # of a dataset such as meta


# ################################## TEST
# SLICING################################
def test_nddataset_loc2index(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    assert da.shape == ref.shape
    coords = da.coordset
    assert len(coords) == 3
    assert da._loc2index(3990.0, dim=0) == 0
    assert da._loc2index("j", dim=0) == 9
    with pytest.raises(IndexError):
        da._loc2index("z", 0)  # labels doesn't exist
    da._loc2index(5000, 0)
    assert da._loc2index(5000, 0) == (0, "out_of_limits")  # return the low limit
    assert da._loc2index(0, 0) == (9, "out_of_limits")  # return the high limit index


def test_nddataset_slicing_by_index(ref_ds, ds1):
    da = ds1
    ref = ref_ds
    # case where the index is an integer: the selection is by index starting
    # at zero
    assert da.shape == ref.shape
    coords = da.coordset
    plane0 = da[0]
    # should return a dataset of with dimension x of size 1
    assert type(plane0) == type(da)
    assert plane0.ndim == 3
    assert plane0.shape == (1, 100, 3)
    assert plane0.size == 300
    assert plane0.dims == ["z", "y", "x"]
    assert_dataset_equal(plane0.z, da.z[0])
    da1 = plane0.squeeze()
    assert da1.shape == (100, 3)
    assert da1.dims == ["y", "x"]
    plane1 = da[:, 0]
    assert type(plane1) == type(da)
    assert plane1.ndim == 3
    assert plane1.shape == (10, 1, 3)
    assert plane1.size == 30
    assert plane1.dims == ["z", "y", "x"]
    da2 = plane1.squeeze()
    assert da2.dims == ["z", "x"]
    assert_dataset_almost_equal(da2.z, coords[-1], decimal=2)  # remember
    # coordinates
    # are ordered by name!
    assert_dataset_almost_equal(da2.x, coords[0])
    # another selection
    row0 = plane0[:, 0]
    assert type(row0) == type(da)
    assert row0.shape == (1, 1, 3)
    # and again selection
    element = row0[..., 0]
    assert type(element) == type(da)
    assert element.dims == ["z", "y", "x"]
    # squeeze
    row1 = row0.squeeze()
    assert row1.mask == scp.NOMASK
    row1[0] = scp.MASKED
    assert row1.dims == ["x"]
    assert row1.shape == (3,)
    assert row1.mask.shape == (3,)
    element = row1[..., 0]
    assert element.x == coords[0][0]
    # now a slicing in multi direction
    da[1:4, 10:40:2, :2]
    # now again a slicing in multi direction (full selection)
    da[:, :, -2]
    # now again a slicing in multi direction (ellipsis)
    da[..., -1]
    da[-1, ...]


def test_nddataset_slicing_by_label(ds1):
    da = ds1
    # selection
    planeb = da["b"]
    assert type(planeb) == type(da)
    plane1 = da[1]
    assert_equal(planeb.data, plane1.data)
    assert planeb.ndim == 3
    assert planeb.size == 300
    bd = da["b":"f"]  # the last index is included
    assert bd.shape == (5, 100, 3)
    b1 = da[1:6]
    assert_equal(bd.data, b1.data)
    bc = da["b":"f", :, "hot"]
    assert bc.shape == (5, 100, 1)
    assert bc.z.labels[0] == "b"
    da[..., "hot"]  # TODO: find a way to use such syntax
    # hot2 = da[  #  # "x.hot"]  # assert hot == hot2


def test_nddataset_slicing_by_values(ds1):
    da = ds1
    x = da[3000.0]
    assert x.shape == (1, 100, 3)
    y = da[3000.0:2000.0, :, 210.0]
    assert y.shape == (4, 100, 1)
    # slicing by values should also work using reverse order
    da[2000.0:3000.0, :, 210.0]

    x = Coord(data=np.linspace(1000.0, 4000.0, num=6000), title="x")
    y = Coord(data=np.linspace(0.0, 10, num=5), title="y")
    data = np.random.rand(x.size, y.size)
    ds = NDDataset(data, coordset=[x, y])
    ds2 = ds[2000.0:3200.0, :]
    assert ds2.coordset.y.data.shape[0] == 2400, "taille axe 0 doit être 2400"
    assert ds2.data.shape[0] == 2400, "taille dimension 0 doit être 2400"


def test_nddataset_slicing_out_limits(caplog, ds1):
    import logging

    logger = logging.getLogger("SpectroChemPy")
    logger.propagate = True
    caplog.set_level(logging.DEBUG)
    da = ds1
    y1 = da[2000.0]
    assert str(y1) == "NDDataset: [float64] a.u. (shape: (z:1, y:100, x:3))"
    y2 = da[2000]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message.startswith(
        "ERROR | Empty array of shape (0, 100, 3) resulted from slicing."
    )
    y3 = da[:, 95:105]
    assert str(y3) == "NDDataset: [float64] a.u. (shape: (z:10, y:5, x:3))"
    da[5000.0:4001.0]
    assert y2 is None  # as we are out of limits
    assert caplog.records[-1].levelname == "ERROR"
    assert caplog.records[-1].message.startswith(
        "ERROR | Empty array of shape (0, 100, 3) resulted from slicing."
    )
    y5 = da[5000.0:3000.0]
    assert str(y5) == "NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))"


def test_nddataset_slicing_toomanyindex(ds1):
    da = ds1
    with pytest.raises(IndexError):
        da[:, 3000.0:2000.0, :, 210.0]


def test_nddataset_slicing_by_index_nocoords(ds1):
    da = ds1
    # case where the index is an integer:
    # the selection is by index starting at zero
    da.delete_coordset()  # clear coords
    plane0 = da[1]
    assert type(plane0) == type(da)  # should return a dataset
    assert plane0.ndim == 3
    assert plane0.size == 300


def test_nddataset_slicing_by_location_but_nocoords(ref_ds, ds1):
    da = ds1
    # case where the index is an integer:
    # the selection is by index starting at zero
    da.delete_coordset()  # clear coords
    # this cannot work (no coords for location)
    with pytest.raises(MissingCoordinatesError):
        _ = da[3666.7]


# slicing tests
def test_nddataset_simple_slicing():
    d1 = NDDataset(np.ones((5, 5)))
    assert d1.data.shape == (5, 5)
    assert d1.shape == (5, 5)
    d2 = d1[2:3, 2:3]
    assert d2.shape == (1, 1)
    assert d1 is not d2
    d3 = d1[2, 2]
    assert d3.shape == (1, 1)
    assert d3.squeeze().shape == ()
    d3 = d1[0]
    assert d3.shape == (1, 5)

    with pytest.raises(MissingCoordinatesError) as exc:
        _ = d1[0 * ur.cm]
    assert (
        exc.value.args[0]
        == "No coords have been defined. Slicing or selection by location (0.0) "
        "needs coords definition."
    )


def test_nddataset_slicing_with_mask():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = NDDataset(np.ones((5, 5)), mask=mask)
    assert d1[1].shape == (1, 5)
    assert d1[1, 1].mask


def test_nddataset_slicing_with_mask_units():
    mask = np.zeros((5, 5)).astype(bool)
    mask[1, 1] = True
    d1 = NDDataset(np.ones((5, 5)), mask=mask, units="m")
    assert d1[0].shape == (1, 5)


def test_nddataset_slicing_with_coords(ds1):
    da = ds1.copy()
    da00 = da[0, 0]
    assert da00.shape == (1, 1, 3)
    assert da00.coordset["x"] == da00.coordset[0]
    assert da00.coordset["x"] == da.coordset[0]


def test_nddataset_slicing_with_quantities(ds1):
    da = ds1.copy()

    da00 = da[1000.0 * ur("cm^-1"), 0]
    assert da00.shape == (1, 1, 3)
    assert da00.coordset["x"] == da00.coordset[0]
    assert da00.coordset["x"] == da.coordset[0]

    with pytest.raises(ValueError):
        _ = da[1000.0 * ur.K, 0]  # wrong units


def test_nddataset_mask_array_input():
    marr = np.ma.array([1.0, 2.0, 5.0])  # Masked array with no masked entries
    nd = NDDataset(marr)
    assert not nd.is_masked
    marr = np.ma.array([1.0, 2.0, 5.0], mask=[True, False, False])  # Masked array
    nd = NDDataset(marr)
    assert nd.is_masked


def test_nddataset_unmasked_in_operation_with_masked_numpy_array():
    ndd = NDDataset(np.array([1, 2, 3]))
    np_data = -np.ones_like(ndd)
    np_mask = np.array([True, False, True])
    np_arr_masked = np.ma.array(np_data, mask=np_mask)
    result1 = ndd * np_arr_masked
    assert result1.is_masked
    assert np.all(result1.mask == np_mask)
    # TODO: IndexError: in the future, 0-d boolean arrays will be
    #  interpreted as a valid boolean index
    # assert np.all(result1[~result1.mask].data == -ndd.data[~np_mask])
    result2 = np_arr_masked * ndd
    # Numpy masked  array return a masked array in this case
    # assert result2.is_masked
    assert np.all(
        result2.mask == np_mask
    )  # assert np.all(result2[  #  # ~result2.mask].data == -ndd.data[~np_mask])


@pytest.mark.parametrize("shape", [(10,), (5, 5), (3, 10, 10)])
def test_nddataset_mask_invalid_shape(shape):
    with pytest.raises(ValueError) as exc:
        with RandomSeedContext(789):
            NDDataset(np.random.random((10, 10)), mask=np.random.random(shape) > 0.5)
    assert exc.value.args[0] == "mask {} and data (10, 10) shape mismatch!".format(
        shape
    )


@pytest.mark.parametrize(
    "mask_in", [np.array([True, False]), np.array([1, 0]), [True, False], [1, 0]]
)
def test_nddataset_mask_init_without_np_array(mask_in):
    ndd = NDDataset(np.array([1, 1]), mask=mask_in)
    assert (ndd.mask == mask_in).all()


def test_nddataset_with_mask_acts_like_masked_array():
    # test for #2414
    input_mask = np.array([True, False, False])
    input_data = np.array([1.0, 2.0, 3.0])
    ndd_masked = NDDataset(input_data.copy(), mask=input_mask.copy())
    #   ndd_masked = np.sqrt(ndd_masked)
    other = -np.ones_like(input_data)
    result1 = np.multiply(ndd_masked, other)
    result2 = ndd_masked * other
    result3 = other * ndd_masked
    result4 = other / ndd_masked
    # Test for both orders of multiplication
    for result in [result1, result2, result3, result4]:
        assert result.is_masked
        # Result mask should match input mask because other has no mask
        assert np.all(
            result.mask == input_mask
        )  # TODO:IndexError: in the   #  # future, 0-d boolean arrays will be  #
        # interpreted  # as a  # valid  # boolean index
        # assert np.all(result[~result.mask].data == -   #  #  #
        # input_data[~input_mask])


def test_nddataset_creationdate():
    ndd = NDDataset([1.0, 2.0, 3.0])
    ndd2 = np.sqrt(ndd)
    assert ndd2._created is not None


def test_nddataset_title():
    ndd = NDDataset([1.0, 2.0, 3.0], title="xxxx")
    assert ndd.title == "xxxx"
    ndd2 = NDDataset(ndd, title="yyyy")
    assert ndd2.title == "yyyy"
    ndd2.title = "zzzz"
    assert ndd2.title == "zzzz"


def test_nddataset_real_imag():
    na = np.array(
        [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]]
    )
    nd = NDDataset(na)
    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)


def test_nddataset_comparison():
    ndd = NDDataset([1.0, 2.0 + 1j, 3.0])
    val = ndd * 1.2 - 10.0
    val = np.abs(val)
    assert np.all(val >= 6.0)


def test_nddataset_repr_html():
    dx = np.random.random((10, 100, 3))
    coord0 = scp.Coord(
        data=np.linspace(4000.0, 1000.0, 10),
        labels="a b c d e f g h i j".split(),
        mask=None,
        units="cm^-1",
        title="wavelength",
    )
    coord1 = scp.Coord(
        data=np.linspace(0.0, 60.0, 100),
        labels=None,
        mask=None,
        units="s",
        title="time-on-stream",
    )
    coord2 = scp.Coord(
        data=np.linspace(200.0, 300.0, 3),
        labels=["cold", "normal", "hot"],
        mask=None,
        units="K",
        title="temperature",
    )
    da = NDDataset(
        dx, coordset=[coord0, coord1, coord2], title="absorbance", units="absorbance"
    )
    da._repr_html_()


# ### Metadata ################################################################
def test_nddataset_with_meta(ds1):
    da = ds1.copy()
    meta = scp.Meta()
    meta.essai = ["try_metadata", 10]
    da.meta = meta
    # check copy of meta
    dac = da.copy()
    assert dac.meta == da.meta


# ### sorting #################################################################
def test_nddataset_sorting(ds1):  # ds1 is defined in conftest
    dataset = ds1[:3, :3, 0].copy()
    dataset.sort(inplace=True, dim="z")
    labels = np.array(list("abc"))
    assert_array_equal(dataset.coordset["z"].labels, labels)
    # nochange because the  axis is naturally inverted to force it
    # we need to specify descend
    dataset.sort(
        inplace=True, descend=False, dim="z"
    )  # order value in increasing order
    labels = np.array(list("cba"))
    assert_array_equal(dataset.coordset["z"].labels, labels)
    dataset.sort(inplace=True, dim="z")
    new = dataset.copy()
    new = new.sort(descend=False, inplace=False, dim="z")
    assert_array_equal(new.data, dataset.data[::-1])
    assert new[0, 0] == dataset[-1, 0]
    assert_array_equal(new.coordset["z"].labels, labels)
    assert_array_equal(new.coordset["z"].data, dataset.coordset["z"].data[::-1])
    # check for another dimension
    dataset = ds1.copy()
    new = ds1.copy()
    new.sort(dim="y", inplace=True, descend=False)
    assert_array_equal(new.data, dataset.data)
    assert new[0, 0, 0] == dataset[0, 0, 0]
    new = dataset.copy()
    new.sort(dim="y", inplace=True, descend=True)
    assert_array_equal(new.data, dataset.data[:, ::-1, :])
    assert new[0, -1, 0] == dataset[0, 0, 0]


# ### multiple axis
# #############################################################
def test_nddataset_multiple_axis(
    ref_ds, coord0, coord1, coord2, coord2b, dsm
):  # dsm is defined in conftest
    ref = ref_ds
    da = dsm.copy()
    coordm = scp.CoordSet(coord2, coord2b)
    # check indexing
    assert da.shape == ref.shape
    coords = da.coordset
    assert len(coords) == 3
    assert coords["z"] == coord0
    assert da.z == coord0
    assert da.coordset["wavenumber"] == coord0
    assert da.wavenumber == coord0
    assert da["wavenumber"] == coord0
    # for multiple coordinates
    assert da.coordset["x"] == coordm
    assert da["x"] == coordm
    assert da.x == coordm
    # but we can also specify, which axis should be returned explicitly
    # by an index or a label
    assert da.coordset["x_1"] == coord2b
    assert da.coordset["x_2"] == coord2
    assert da.coordset["x"][1] == coord2
    assert da.coordset["x"]._1 == coord2b
    assert da.x["_1"] == coord2b
    assert da["x_1"] == coord2b
    assert da.x_1 == coord2b
    x = da.coordset["x"]
    assert x["temperature"] == coord2
    assert da.coordset["x"]["temperature"] == coord2
    # even simpler we can specify any of the axis title and get it ...
    assert da.coordset["time-on-stream"] == coord1
    assert da.coordset["temperature"] == coord2
    da.coordset["magnetic field"] += 100 * ur.millitesla
    assert da.coordset["magnetic field"] == coord2b + 100 * ur.millitesla


def test_nddataset_coords_manipulation(dsm):
    dataset = dsm.copy()
    coord0 = dataset.coordset["y"]
    coord0 -= coord0[0]  # remove first element


def test_nddataset_square_dataset_with_identical_coordinates():
    a = np.random.rand(3, 3)
    c = scp.Coord(np.arange(3) * 0.25, title="time", units="us")
    nd = NDDataset(a, coordset=scp.CoordSet(x=c, y="x"))
    assert nd.x == nd.y


# ### Test masks ######
def test_nddataset_use_of_mask(dsm):
    nd = dsm
    nd[950.0:1260.0] = scp.MASKED


# ------------------------------------------------------------------
# additional tests made following some bug fixes
# ------------------------------------------------------------------
def test_nddataset_repr_html_bug_undesired_display_complex():
    da = NDDataset([1, 2, 3])
    da.title = "intensity"
    da.description = "Some experimental measurements"
    da.units = "dimensionless"
    assert "(complex)" not in da._repr_html_()
    pass


def test_nddataset_bug_fixe_figopeninnotebookwithoutplot():
    da = scp.NDDataset([1, 2, 3])
    da2 = np.sqrt(da**3)
    assert da2._fig is None  # no figure should open


# ################ Complex and Quaternion, and NMR ##################
def test_nddataset_create_from_complex_data():
    # 1D (complex)
    nd = NDDataset([1.0 + 2.0j, 2.0 + 0j])
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2,)
    assert nd.shape == (2,)
    # 2D (complex in the last dimension - automatic detection)
    nd = NDDataset(
        [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]]
    )
    assert nd.data.size == 6
    assert nd.size == 6
    assert nd.data.shape == (3, 2)
    assert nd.shape == (3, 2)
    # 2D quaternion
    nd = NDDataset(
        [
            [1.0, 2.0],
            [1.3, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=typequaternion,
    )
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2, 1)
    assert nd.shape == (2, 1)
    # take real part
    ndr = nd.real
    assert ndr.shape == (2, 1)
    assert not ndr.is_quaternion


def test_nddataset_set_complex_1D_during_math_op():
    nd = NDDataset([1.0, 2.0], coordset=[scp.Coord([10, 20])], units="meter")
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.shape == (2,)
    assert nd.units == ur.meter
    assert not nd.is_complex
    ndj = nd * 1j
    assert ndj.data.size == 2
    assert ndj.is_complex


def test_nddataset_create_from_complex_data_with_units():
    # 1D
    nd = NDDataset([1.0 + 2.0j, 2.0 + 0j])
    assert nd.data.size == 2
    assert nd.size == 2
    assert nd.data.shape == (2,)
    assert nd.shape == (2,)
    # add units
    nd.units = "m**-1"
    nd.ito("cm^-1")
    # 2D
    nd2 = NDDataset(
        [[1.0 + 2.0j, 2.0 + 0j], [1.3 + 2.0j, 2.0 + 0.5j], [1.0 + 4.2j, 2.0 + 3j]]
    )
    assert nd2.data.size == 6
    assert nd2.size == 6
    assert nd2.data.shape == (3, 2)
    assert nd2.shape == (3, 2)
    # add units
    nd2.units = "m**-1"
    nd2.ito("cm^-1")


def test_nddataset_real_imag_quaternion():
    na = np.array(
        [[1.0 + 2.0j, 2.0 + 0j, 1.3 + 2.0j], [2.0 + 0.5j, 1.0 + 4.2j, 2.0 + 3j]]
    )
    nd = NDDataset(na)
    # in the last dimension
    assert_array_equal(nd.real, na.real)
    assert_array_equal(nd.imag, na.imag)
    # in another dimension
    nd.set_hypercomplex(inplace=True)
    assert nd.is_quaternion
    assert nd.shape == (1, 3)
    na = np.array(
        [
            [1.0 + 2.0j, 2.0 + 0j],
            [1.3 + 2.0j, 2.0 + 0.5j],
            [1.0 + 4.2j, 2.0 + 3j],
            [5.0 + 4.2j, 2.0 + 3j],
        ]
    )
    nd = NDDataset(na)
    nd.set_hypercomplex(inplace=True)
    assert nd.is_quaternion
    assert_array_equal(nd.real.data, na[::2, :].real)
    nb = np.array(
        [
            [0.0 + 2.0j, 0.0 + 0j],
            [1.3 + 2.0j, 2.0 + 0.5j],
            [0.0 + 4.2j, 0.0 + 3j],
            [5.0 + 4.2j, 2.0 + 3j],
        ]
    )
    ndj = NDDataset(nb, dtype=quaternion)
    assert nd.imag == ndj


def test_nddataset_hypercomplex():
    na0 = np.array(
        [
            [1.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            [1.3, 2.0, 2.0, 0.5, 1.0, 1.0],
            [1, 4.2, 2.0, 3.0, 2.0, 2.0],
            [5.0, 4.2, 2.0, 3.0, 3.0, 3.0],
        ]
    )
    nd = NDDataset(na0)
    assert nd.shape == (4, 6)
    nd.dims = ["v", "u"]
    nd.set_coordset(v=np.linspace(-1, 1, 4), u=np.linspace(-10.0, 10.0, 6))
    nd.set_hypercomplex()
    # test swapdims
    nds = nd.swapdims(0, 1)
    assert_array_equal(nd.data.T, nds.data)
    assert nd.coordset[0] == nds.coordset[0]  # we do not swap the coords
    # test transpose
    nds = nd.T
    assert_array_equal(nd.data.T, nds.data)
    assert nd.coordset[0] == nds.coordset[0]


def test_nddataset_max_with_2D_quaternion(NMR_dataset_2D):
    # test on a 2D NDDataset
    nd2 = NMR_dataset_2D
    assert nd2.is_quaternion
    nd = nd2.RR
    nd.max()
    nd2.max()  # no axis specified
    nd2.max(dim=0)  # axis selected


def test_nddataset_max_min_with_1D(NMR_dataset_1D):
    # test on a 1D NDDataset
    nd1 = NMR_dataset_1D
    nd1[4] = scp.MASKED
    assert nd1.is_masked
    mx = nd1.max()
    assert (mx.real, mx.imag) == pytest.approx((2283.5096153847107, -2200.383064516033))
    # check if it works for real
    mx1 = nd1.real.max()
    assert mx1 == pytest.approx(2283.5096153847107)
    mi = nd1.min()
    assert (mi.real, mi.imag) == pytest.approx((-408.29714640199626, 261.1864143920416))


def test_nddataset_comparison_of_dataset(NMR_dataset_1D):
    # bug in notebook
    nd1 = NMR_dataset_1D.copy()
    nd2 = NMR_dataset_1D.copy()
    lb1 = nd1.em(lb=100.0 * ur.Hz)
    lb2 = nd2.em(lb=100.0 * ur.Hz)
    assert nd1 is not nd2
    assert nd1 == nd2
    assert lb1 is not lb2
    assert lb1 == lb2


def test_nddataset_complex_dataset_slicing_by_index():
    na0 = np.array([1.0 + 2.0j, 2.0, 0.0, 0.0, -1.0j, 1j] * 4)
    nd = NDDataset(na0)
    assert nd.shape == (24,)
    assert nd.data.shape == (24,)
    coords = (np.linspace(-10.0, 10.0, 24),)
    nd.set_coordset(coords)
    x1 = nd.x.copy()
    nd.coordset = coords
    x2 = nd.x.copy()
    assert x1 == x2
    # slicing
    nd1 = nd[0]
    assert nd1.shape == (1,)
    assert nd1.data.shape == (1,)
    # slicing range
    nd2 = nd[1:6]
    assert nd2.shape == (5,)
    assert nd2.data.shape == (5,)
    na0 = na0.reshape(6, 4)
    nd = NDDataset(na0)
    coords = scp.CoordSet(np.linspace(-10.0, 10.0, 6), np.linspace(-1.0, 1.0, 4))
    nd.set_coordset(**coords)
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)
    nd.coordset = coords
    # slicing 2D
    nd1 = nd[0]
    assert nd1.shape == (1, 4)
    assert nd1.data.shape == (1, 4)
    # slicing range
    nd1 = nd[1:3]
    assert nd1.shape == (2, 4)
    assert nd1.data.shape == (2, 4)
    # slicing range
    nd1 = nd[1:3, 0:2]
    assert nd1.shape == (2, 2)
    assert nd1.data.shape == (2, 2)
    nd.set_complex()
    assert nd.shape == (6, 4)
    assert nd.data.shape == (6, 4)


def test_nddataset_init_complex_1D_with_mask():
    # test with complex with mask and units
    np.random.seed(12345)
    d = np.random.random((5)) * np.exp(0.1j)
    d1 = NDDataset(d, units=ur.Hz)  # with units
    d1[1] = scp.MASKED
    assert d1.shape == (5,)
    assert d1._data.shape == (5,)
    assert d1.size == 5
    assert d1.dtype == np.complex128
    assert d1.has_complex_dims
    assert d1.mask.shape[-1] == 5
    assert d1[2].data == d[2]
    d1R = d1.real
    assert not d1R.has_complex_dims
    assert d1R._data.shape == (5,)
    assert d1R._mask.shape == (5,)


def test_nddataset_transpose_swapdims(ds1):
    nd = ds1.copy()
    ndt = nd.T
    assert nd[1] == ndt[..., 1].T
    # fix a bug with loc indexation
    nd1 = nd[4000.0:3000.0]
    assert str(nd1) == "NDDataset: [float64] a.u. (shape: (z:4, y:100, x:3))"
    nd2 = ndt[..., 4000.0:3000.0]
    assert str(nd2) == "NDDataset: [float64] a.u. (shape: (x:3, y:100, z:4))"
    assert nd1 == nd2.T


def test_nddataset_set_coordinates(nd2d, ds1):
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(x=np.arange(nx), y=np.arange(ny))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.x == np.arange(nx)
    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(np.arange(ny), np.arange(nx))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.x == np.arange(nx)
    # set coordinate with one set to None: should work!
    # set coordinates from tuple
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(np.arange(ny), None)
    assert nd.dims == ["y", "x"]
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    nd.transpose(inplace=True)
    assert nd.dims == ["x", "y"]
    assert nd.y == np.arange(ny)
    assert nd.x.is_empty
    assert nd.coordset == scp.CoordSet(np.arange(ny), None)
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.set_coordset(None, np.arange(nx))
    assert nd.dims == ["y", "x"]
    assert nd.x == np.arange(nx)
    assert nd.y.is_empty
    nd.set_coordset(y=np.arange(ny), x=None)
    # set up a single coordinates
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.x = np.arange(nx)
    nd.x = np.arange(nx)  # do it again - fix  a bug
    nd.set_coordtitles(y="intensity", x="time")
    assert repr(nd.coordset) == "CoordSet: [x:time, y:intensity]"
    # validation
    with pytest.raises(ValueError):
        nd.x = np.arange(nx + 5)
    with pytest.raises(AttributeError):
        nd.z = None
    # set coordinates all together
    nd = nd2d.copy()
    ny, nx = nd.shape
    nd.coordset = scp.CoordSet(u=np.arange(nx), v=np.arange(ny))
    assert nd.dims != ["u", "v"]  # dims = ['y','x']
    # set dim names
    nd.dims = ["u", "v"]
    nd.set_coordset(**scp.CoordSet(u=np.arange(ny), v=np.arange(nx)))
    assert nd.dims == ["u", "v"]


# ## issue 29
def test_nddataset_issue_29_mulitlabels():
    DS = NDDataset(np.random.rand(3, 4))
    with pytest.raises(ValueError):
        # shape data and label mismatch
        DS.set_coordset(
            DS.y,
            scp.Coord(
                title="xaxis", units="s", data=[1, 2, 3, 4], labels=["a", "b", "c"]
            ),
        )
    c = scp.Coord(
        title="xaxis", units="s", data=[1, 2, 3, 4], labels=["a", "b", "c", "d"]
    )
    DS.set_coordset(x=c)
    c = scp.Coord(
        title="xaxis",
        units="s",
        data=[1, 2, 3, 4],
        labels=[["a", "c", "b", "d"], ["e", "f", "g", "h"]],
    )
    d = DS.y
    DS.set_coordset(d, c)
    DS.x.labels = ["alpha", "beta", "omega", "gamma"]
    assert DS.x.labels.shape == (4, 3)
    # sort
    DS1 = DS.sort(axis=1, by="value", descend=True)
    assert_array_equal(DS1.x, [4, 3, 2, 1])
    # sort
    assert DS.dims == ["y", "x"]
    DS1 = DS.sort(dim="x", by="label", descend=False)
    assert_array_equal(DS1.x, [1, 3, 2, 4])
    DS1 = DS.sort(dim="x", by="label", pos=2, descend=False)
    assert_array_equal(DS1.x, [1, 2, 4, 3])
    DS.sort(dim="y")
    DS.y.labels = ["alpha", "omega", "gamma"]
    DS2 = DS.sort(dim="y")
    assert_array_equal(DS2.y.labels, ["alpha", "gamma", "omega"])


def test_nddataset_apply_funcs(dsm):
    # convert to masked array
    np.ma.array(dsm)
    dsm[1] = scp.MASKED
    np.ma.array(dsm)
    np.array(dsm)


def test_nddataset_take(dsm):
    pass


def test_nddataset_with_datetime64_coordinates(dsdt64):

    X = dsdt64.copy()
    # assert X.x ==
    assert X.x_0.dtype == np.dtype("datetime64[ns]")
    assert X.x.units is None
    # there is no units for this object as it is defined internally

    with assert_produces_log_warning(
        match="method `to` cannot be used with datetime object. Ignored!"
    ):
        X.y.to("tesla")
        # can not change to a foreign unit (of course)

    # subtract a datetime to a datetime array --> should be a float
    X.y = X.y - X.y[0]  # subtract the acquisition timestamp of the first spectrum
    assert X.y.dtype == np.dtype("float")
    assert X.y.units == ur.second

    X.y = X.y.to("minute")  # convert to minutes
    assert X.y.units == ur.minute
    X.y += 2
    # add 2 minutes
    assert X.y.units == ur.minute
    assert X.y[0].data == [2]


# Conversion


@pytest.mark.skipif(
    optional.import_optional_dependency("xarray", errors="ignore") is None,
    reason="need xarray package to run",
)
def test_nddataset_to_xarray(IR_dataset_2D):

    nd1 = IR_dataset_2D[0].squeeze()

    xnd1 = nd1.to_xarray()
    print(xnd1)
    assert xnd1.x.attrs["pint_units"] == nd1.x.units

    # 2D
    nd2 = IR_dataset_2D
    nd2[:, 1230.0:920.0] = scp.MASKED

    # add some attribute
    nd2.meta.pression = 34 * ur.pascal
    nd2.meta.temperature = 3000 * ur.K
    assert nd2.meta.temperature == 3000 * ur.K
    assert nd2.temperature == 3000 * ur.K  # alternative way to get the meta attribute

    assert nd2.meta.essai is None  # do not exist in dict
    with pytest.raises(AttributeError):
        nd2.essai2  # can not find this attribute

    # also for the coordinates
    nd2.y.meta.pression = 3 * ur.torr
    assert nd2.y.meta["pression"] == 3 * ur.torr
    assert nd2.y.pression == 3 * ur.torr  # alternative way to get the meta attribute

    assert nd2.y.meta.essai is None  # not found so it is None
    with pytest.raises(AttributeError):
        nd2.y.essai  # can't find such attribute

    # convert
    xnd2 = nd2.to_xarray()
    print(xnd2)
    assert xnd2.y.attrs["pint_units"] == "None"
    assert xnd2.y.attrs["meta_pression"] == nd2.y.meta["pression"].m
    assert xnd2.y.attrs["meta_pression"] == nd2.y.pression.m
    assert xnd2.y.attrs["pint_units_meta_pression"] == nd2.y.meta["pression"].u


####
def test_nddataset_squeeze_method():

    c1 = scp.Coord([2], units="m")
    c2 = scp.Coord.arange(3, units="m^2")
    c3 = scp.Coord([3], units="m^3")
    nd = scp.NDDataset(np.ones((1, 3, 1)))

    assert nd.shape == (1, 3, 1)

    nd.x = c3
    nd.y = c2
    nd.z = c1

    nd2 = nd.squeeze()

    assert nd2.shape == (3,)

    nd3 = nd.squeeze(0)

    assert nd3.shape == (3, 1)

    nd4 = nd.squeeze(keepdims=2)

    assert nd4.shape == (3, 1)
    assert str(nd4) == "NDDataset: [float64] unitless (shape: (y:3, x:1))"

    nd5 = nd.squeeze("x")
    assert nd5.shape == (1, 3)

    nd6 = nd.squeeze(keepdims="x")

    assert nd6.shape == (3, 1)


def test_nddataset_issue_310():

    import numpy as np

    import spectrochempy as scp

    D = scp.NDDataset(np.zeros((10, 5)))
    c5 = scp.Coord.linspace(start=0.0, stop=1000.0, num=5, name="xcoord")
    c10 = scp.Coord.linspace(start=0.0, stop=1000.0, num=10, name="ycoord")

    assert c5.name == "xcoord"
    D.set_coordset(x=c5, y=c10)
    assert c5.name == "x"
    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    E = scp.NDDataset(np.ndarray((10, 5)))
    E.dims = ["t", "s"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["t", "s"]

    E = scp.NDDataset(np.ndarray((5, 10)))
    E.dims = ["s", "t"]
    E.set_coordset(s=c5, t=c10)
    assert c5.name == "s"
    assert str(E.coordset) == "CoordSet: [s:<untitled>, t:<untitled>]"
    assert E.dims == ["s", "t"]

    assert str(D.coordset) == "CoordSet: [x:<untitled>, y:<untitled>]"
    assert D.dims == ["y", "x"]

    assert str(D[:, 1]) == "NDDataset: [float64] unitless (shape: (y:10, x:1))"


def test_nddataset_meta(ndarray):
    nd = ndarray.copy()
    nd.meta = {"essai": "un essai"}
    assert nd.meta.essai == "un essai"
    nd.meta = {"essai2": "un essai2"}
    assert list(nd.meta.keys()) == ["essai", "essai2"]

    # NMR
    nd.meta = {"some_metadata": "toto"}
    nd.meta.some_metadata = "titi"
    nd.meta.larmor = 100 * ur.MHz
    nd.ito("ppm", force=True)
    assert nd.units == ur.ppm
    nd.ito("MHz")
    assert nd.units == ur.MHz
    nd.meta.larmor = None


def test_nddataset_issue_20():
    # Description of bug #20
    # -----------------------
    # X = read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))
    #
    # # slicing a NDDataset with an integer is OK for the coord:
    # X[:,100].x
    # Out[4]: Coord: [float64] cm^-1
    #
    # # but not with an integer array :
    # X[:,[100, 120]].x
    # Out[5]: Coord: [int32] unitless
    #
    # # on the other hand, the explicit slicing of the coord is OK !
    # X.x[[100,120]]
    # Out[6]: Coord: [float64] cm^-1

    X = scp.read_omnic("CO@Mo_Al2O3.SPG")
    assert X.__str__() == "NDDataset: [float64] a.u. (shape: (y:19, x:3112))"

    # slicing a NDDataset with an integer is OK for the coord:
    assert X[:, 100].x.__str__() == "Coord: [float64] cm⁻¹ (size: 1)"

    # The explicit slicing of the coord is OK !
    assert X.x[[100, 120]].__str__() == "Coord: [float64] cm⁻¹ (size: 2)"

    # slicing the NDDataset with an integer array is also OK (fixed #20)
    assert X[:, [100, 120]].x.__str__() == X.x[[100, 120]].__str__()


def test_nddataset_issue_462():

    A = scp.random((10, 100))
    A.x = scp.Coord(np.arange(0.0, 100.0, 1), title="coord1")
    A.write("A.scp", confirm=False)
    B = scp.read("A.scp")
    assert B.x == A.x, "incorrect encoding/decoding"

    C = scp.random((10, 100))
    C.x = [
        scp.Coord(np.arange(0.0, 100.0, 1), title="coord1"),
        scp.Coord(np.arange(0.0, 1000.0, 10), title="coord2"),
    ]
    C.write("C.scp", confirm=False)
    D = scp.read("C.scp")
    assert len(D.x) == 2, "incorrect encoding/decoding"
