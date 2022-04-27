# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


from copy import copy
from os import environ

import numpy as np
import pytest
from pint.errors import DimensionalityError

from spectrochempy.core import debug_
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME
from spectrochempy.core.common.exceptions import (
    InvalidCoordinatesSizeError,
    InvalidCoordinatesTypeError,
    InvalidDimensionNameError,
    ShapeError,
)
from spectrochempy.core.dataset.coord import Coord, CoordSet
from spectrochempy.core.units import Quantity, ur
from spectrochempy.utils import check_docstrings as td
from spectrochempy.utils.testing import (
    assert_approx_equal,
    assert_array_equal,
    assert_produces_log_warning,
)


# ========
# FIXTURES
# ========
@pytest.fixture(scope="function")
def coord0():
    c = Coord(
        data=np.linspace(4000.0, 1000.0, 10),
        labels=list("abcdefghij"),
        units="cm^-1",
        title="wavenumber (coord0)",
    )
    return c


@pytest.fixture(scope="function")
def coord1():
    c = Coord(
        data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream (coord1)"
    )
    return c


@pytest.fixture(scope="function")
def coord2():
    c = Coord(
        data=np.linspace(200.0, 300.0, 3),
        labels=["cold", "normal", "hot"],
        units="K",
        title="temperature (coord2)",
    )
    return c


# ==========
# Coord Test
# ==========
# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause errors when checking docstrings",
)
def test_coord_docstring():
    import spectrochempy

    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.coord"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.coord.Coord,
        exclude=[
            "SA01",  # see also
            "EX01",  # examples
            "ES01",  # extended summary
        ],
    )


def test_coord_init():

    # simple coords

    a = Coord([1, 2, 3], name="x")
    assert a.id is not None
    assert not a.is_empty
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert not a.is_labeled
    assert a.units is None
    assert a.is_unitless
    assert a.name == "x"

    # set properties

    a.title = "xxxx"
    assert a.title == "xxxx"
    a.name = "y"
    assert a.name == "y"

    # now with labels

    x = np.arange(10)
    y = list("abcdefghij")
    a = Coord(x, labels=y, title="processors", name="x")
    assert a.title == "processors"
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)

    # any kind of object can be a label

    assert a.labels.dtype == "O"
    # even an array
    a._labels[3] = x
    assert a._labels[3][2] == 2

    # coords can be defined only with labels

    y = list("abcdefghij")
    a = Coord(labels=y, title="processors")
    assert a.title == "processors"
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)
    # any kind of object can be a label
    assert a.labels.dtype == "O"
    # even an array
    a._labels[3] = range(10)
    assert a._labels[3][2] == 2

    # coords with datetime in labels

    x = np.arange(10)
    y = [np.datetime64(f"2017-06-{(2 * (i + 1)):02d}") for i in x]

    a = Coord(x, labels=y, title="time")
    assert a.title == "time"
    assert isinstance(a.data, np.ndarray)
    assert isinstance(a.labels, np.ndarray)
    b = a._sort(by="label", descend=True)
    assert_array_equal(b.data, a.data[::-1])
    b = a._sort(by="label", descend=True, inplace=True)
    assert_array_equal(b.data, a.data)

    # actually y can also be put in data
    c = Coord(y, title="time")
    assert c.title == "time"
    assert isinstance(c.data, np.ndarray)
    assert isinstance(c.data[0], np.datetime64)
    assert c.dtype == np.dtype("datetime64[D]")
    c._sort(descend=True, inplace=True)
    assert_array_equal(b.labels, c.data)

    # but coordinates must be 1D

    with pytest.raises(ShapeError):
        # should raise an error as coords must be 1D
        Coord(data=np.ones((2, 10)))

    # unitless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        title="wavelength",
    )
    assert coord0.units is None
    assert coord0.data[0] == 4000.0
    assert coord0.coordinates[0] == 4000.0
    assert repr(coord0) == "Coord (wavelength): [float64] unitless (size: 10)"

    # dimensionless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units=ur.dimensionless,
        title="wavelength",
        name="toto",
    )
    assert coord0.units.dimensionless
    assert not coord0.is_unitless
    assert coord0.is_dimensionless
    assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord toto(wavelength): [float64] dimensionless (size: 10)"

    # scaled dimensionless coordinates

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="m/km",
        title="wavelength",
    )
    assert coord0.units.dimensionless
    assert (
        coord0.data[0] == 4000.0
    )  # <- displayed data to be multiplied by the scale factor
    assert (
        repr(coord0)
        == "Coord (wavelength): [float64] scaled-dimensionless (0.001) (size: 10)"
    )

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units=ur.m / ur.km,
        title="wavelength",
    )

    assert coord0.units.dimensionless
    assert (
        coord0.data[0] == 4000.0
    )  # <- displayed data to be multiplied by the scale factor
    assert (
        repr(coord0)
        == "Coord (wavelength): [float64] scaled-dimensionless (0.001) (size: 10)"
    )

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="m^2/s",
        title="wavelength",
    )
    assert not coord0.units.dimensionless
    assert coord0.units.scaling == 1.0
    assert coord0.data[0] == 4000.0
    assert repr(coord0) == "Coord (wavelength): [float64] m².s⁻¹ (size: 10)"

    # comparison

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        title="wavelength",
    )
    coord0b = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="a b c d e f g h i j".split(),
        title="wavelength",
    )
    coord1 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="a b c d e f g h i j".split(),
        title="titi",
    )
    coord2 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels="b c d e f g h i j a".split(),
        title="wavelength",
    )
    coord3 = Coord(
        data=np.linspace(4000, 1000, 10), labels=None, mask=None, title="wavelength"
    )

    assert coord0 == coord0b
    assert coord0 != coord1  # different title
    assert coord0 != coord2  # different labels
    assert coord0 != coord3  # one coord has no label

    # init from another coord

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        title="wavelength",
    )

    coord1 = Coord(coord0)
    assert coord1._data is coord0._data
    coord1 = Coord(coord0, copy=True)
    assert coord1._data is not coord0._data
    assert_array_equal(coord1._data, coord0._data)
    assert isinstance(coord0, Coord)
    assert isinstance(coord1, Coord)

    # sort

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        title="wavelength",
    )
    assert coord0.is_labeled
    ax = coord0._sort()
    assert ax.data[0] == 1000
    coord0._sort(descend=True, inplace=True)
    assert coord0.data[0] == 4000
    ax1 = coord0._sort(by="label", descend=True)
    assert ax1.labels[0] == "j"

    # copy

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        title="wavelength",
    )

    coord1 = coord0.copy()
    assert coord1 is not coord0

    assert_array_equal(coord1.data, coord0.data)
    assert_array_equal(coord1.labels, coord0.labels)
    assert coord1.units == coord0.units

    coord2 = copy(coord0)
    assert coord2 is not coord0

    assert_array_equal(coord2.data, coord0.data)
    assert_array_equal(coord2.labels, coord0.labels)
    assert coord2.units == coord0.units

    # automatic reversing for wavenumbers
    coord0 = Coord(data=np.linspace(4000, 1000, 10), units="cm^-1", title="wavenumbers")
    assert coord0.reversed

    # invalid data shape
    with pytest.raises(ShapeError):
        _ = Coord([[1, 2, 3], [3, 4, 5]])


def test_coord_larmor():
    c = Coord([1, 2, 3], name="x")
    assert c.larmor is None
    c.larmor = 100 * ur("MHz")
    assert c.larmor == 100 * ur("MHz")
    c = Coord([1, 2, 3], name="x", larmor=150 * ur("MHz"))
    assert c.larmor == 150 * ur("MHz")


def test_coord_is_descendant():
    c = Coord([1, 2, 3], name="x")
    assert not c.is_descendant
    c = Coord([3, 2, 1], name="x")
    assert c.is_descendant


def test_coord_str_repr_summary():

    nd = Coord(
        name="toto",
        data=np.arange("2020", "2030", dtype="<M8[Y]"),
        labels=list("abcdefghij"),
    )
    assert repr(nd) == "Coord toto(value): [datetime64[Y]] unitless (size: 10)"

    # repr_html and summary
    nd = Coord([1, 2, 3], labels=list("abc"))
    assert "coordinates:" in nd._cstr()[2] and "labels:" in nd._cstr()[2]

    nd = Coord(labels=list("abc"))
    assert "coordinates:" in nd._cstr()[2] and "labels:" not in nd._cstr()[2]

    nd = Coord(labels=list("abcdefghij"), name="z")
    assert nd.summary.startswith("\x1b[32m         name")
    assert (
        str(nd)
        == repr(nd)
        == "Coord z(value): [labels] [  a   b ...   i   j] (size: 10)"
    )

    nd = Coord(labels=[list("abcdefghij"), list("0123456789"), list("lmnopqrstx")])
    assert "labels[1]" in nd.summary
    assert "labels[1]" in nd._repr_html_()


def test_coord_to_xarray():
    xarray = pytest.importorskip("xarray")
    x = Coord([1, 2, 3], name="x")
    cx = x._to_xarray()
    assert isinstance(cx["x"], xarray.Variable)
    y = Coord(labels=["a", "b", "c"], name="y")
    cy = y._to_xarray()
    assert isinstance(cy["y"], xarray.Variable)
    y = Coord(labels=[["a", "b", "c"]], name="y")
    cy = y._to_xarray()
    assert isinstance(cy["y"], xarray.Variable)
    z = Coord(labels=[["a", "b", "c"], ["i", "j", "k"]], name="z")
    cz = z._to_xarray()
    assert isinstance(cz["z"], xarray.Variable)


def test_coord_slicing():
    # slicing by index

    coord0 = Coord(data=np.linspace(4000, 1000, 10), mask=None, title="wavelength")

    assert coord0[0] == 4000.0

    coord1 = Coord(
        data=np.linspace(4000, 1000, 10), units="cm^-1", mask=None, title="wavelength"
    )
    c1 = coord1[0]
    assert isinstance(c1.values, Quantity)
    assert coord1[0].values == 4000.0 * (1.0 / ur.cm)

    # slicing with labels

    labs = list("abcdefghij")

    coord0 = Coord(
        data=np.linspace(4000, 1000, 10),
        labels=labs,
        units="cm^-1",
        mask=None,
        title="wavelength",
    )

    assert coord0[0].values == 4000.0 * (1.0 / ur.cm)
    assert isinstance(coord0[0].values, Quantity)

    assert coord0[2] == coord0["c"]
    assert coord0["c":"d"] == coord0[2:4]  # label included

    # slicing only-labels coordinates

    y = list("abcdefghij")
    a = Coord(labels=y, name="x")
    assert a.name == "x"
    assert isinstance(a.labels, np.ndarray)
    assert_array_equal(a.values, a.labels)


def test_coord_linearize():
    from spectrochempy.utils.misc import spacings

    arr = np.linspace(0.1, 99.9, 100) * 10 + np.random.rand(100) * 0.001
    c = Coord(arr, linear=True, decimals=3)


def test_coord_default():
    # gettting attribute default on a single coordinate should not give an error.

    coord0 = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")

    assert coord0.default is coord0

    #


def test_coord_functions():

    # Creation function

    # class method
    c = Coord.arange(1, 20.0001, 1, units="K", title="temperature")
    assert str(c) == "Coord (temperature): [float64] K (size: 20)"

    # instance method
    c = Coord().arange(1, 20.0001, 1, units="K", title="temperature")
    assert str(c) == "Coord (temperature): [float64] K (size: 20)"

    c = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")
    assert c == Coord(np.linspace(200.0, 300.0, 3), units="K", title="temperature")

    c = Coord.logspace(200.0, 300.0, 3, units="K", title="temperature")
    assert str(c) == "Coord (temperature): [float64] K (size: 3)"

    c = Coord.geomspace(200.0, 300.0, 3, units="K", title="temperature")
    assert str(c) == "Coord (temperature): [float64] K (size: 3)"

    iterable = (x * x for x in range(5))
    d = Coord.fromiter(iterable, float, units="km")
    assert str(d) == "Coord (value): [float64] km (size: 5)"

    iterable = (x * x for x in range(5))
    d = Coord.fromiter(iterable, count=4, dtype=float, units="km")
    assert str(d) == "Coord (value): [float64] km (size: 4)"

    # Maths
    e = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")
    f = e + 2.0
    assert f[0].data == 202.0


def test_coord_ufuncs():

    coord = Coord(data=np.linspace(4000, 1000, 10), title="wavelength")
    coord1 = coord + 1000.0
    assert coord1[0].values == 5000.0
    assert coord1.title == "wavelength"

    # with units
    coord0 = Coord(data=np.linspace(4000, 1000, 10), units="cm^-1")
    coord1 = coord0 + 1000.0 * ur("cm^-1")
    assert coord1[0].values == 5000.0 * ur("cm^-1")

    # without units for the second operand
    coord1 = coord0 + 1000.0
    assert coord1[0].values == 5000.0 * ur("cm^-1")

    coord2 = coord - 500.0
    assert coord2[0].values == 3500.0

    coord2 = np.subtract(coord0, 500.0)
    assert coord2[0].values == 3500.0 * ur("cm^-1")

    coord2 = 500.0 - coord0
    assert coord2[0].values == -3500.0 * ur("cm^-1")

    coord2 += 3000
    assert coord2[0].values == -500.0 * ur("cm^-1")

    # test_ufunc on coord
    c = np.multiply(coord2, 2, coord2)
    assert c is coord2
    assert coord2[0].values == -1000.0 * ur("cm^-1")

    with pytest.raises(NotImplementedError):
        np.fmod(coord2, 2)

    with pytest.raises(NotImplementedError):
        np.multiply.accumulate(coord2)


def test_linearcoord_deprecated():
    from spectrochempy.core.dataset.coord import LinearCoord

    with assert_produces_log_warning(DeprecationWarning):
        _ = LinearCoord(title="z")


# ======================================================================================
# CoordSet Tests
# ======================================================================================
# test docstring
# # but this is not intended to work with the debugger - use run instead of debug!
# @pytest.mark.skipif(
#     environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
#     reason="debug mode cause errors when checking docstrings",
# )
def test_coordset_docstring():
    import spectrochempy
    from spectrochempy.utils import check_docstrings as td

    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.coord"
    result = td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.coord.CoordSet,
        exclude=["SA01", "ES01", "EX01"],
    )


def test_coordset_init(coord0, coord1, coord2):

    coord3 = coord2.copy()
    coord3.title = "My name is titi"

    # First syntax
    # Coordinates are sorted in the coordset
    coordsa = CoordSet(coord0, coord3, coord2)
    # by convention if the names are not specified, then the coordinates follows the
    # order of dims, so they are in reverse order with respect of the coordset where
    # the coords are ordered by names alphabetical order.

    assert coordsa.names == ["x", "y", "z"]
    assert coordsa.titles == [
        "temperature (coord2)",
        "My name is titi",
        "wavenumber (coord0)",
    ]
    assert coordsa.titles == coordsa.titles

    # Dims specified
    coordsa = CoordSet(coord0, coord3, coord2, dims=["x", "y", "z"])
    assert coordsa.names == ["x", "y", "z"]
    assert coordsa.titles == [
        "wavenumber (coord0)",
        "My name is titi",
        "temperature (coord2)",
    ]

    # Wrong dim specified
    with pytest.raises(InvalidDimensionNameError) as e:
        coordsa = CoordSet(coord0, coord3, coord2, dims=["toto", "y", "z"])
    assert "dim=`toto` was given!" in str(e.value)

    # Second syntax with a tuple of coordinates
    coordsb = CoordSet((coord0, coord3, coord2))
    assert coordsb.names == ["x", "y", "z"]

    # But warning
    # A list means that it is a sub-coordset (different meaning)
    coordsa1 = CoordSet([coord0[:3], coord3[:3], coord2[:3]])
    assert coordsa1.names == ["x"]
    assert coordsa1.x.names == ["_0", "_1", "_2"]
    assert coordsa1.x.titles == [
        "wavenumber (coord0)",
        "My name is titi",
        "temperature (coord2)",
    ]
    # Third syntax (probably the most safe as explicit
    coordsc = CoordSet(x=coord2, y=coord3, z=coord0)
    assert coordsc.names == ["x", "y", "z"]

    # We can also use a dictionary
    coordsc1 = CoordSet({"x": coord2, "y": coord3, "z": coord0})
    assert coordsc1.names == ["x", "y", "z"]

    coordsd = CoordSet(coord3, x=coord2, y=coord3, z=coord0)
    # conflict (keyw replace args)
    assert coordsb == coordsd
    assert coordsb == coordsc1

    coord4 = copy(coord2)
    coordsc = CoordSet([coord1[:3], coord2[:3], coord4[:3]])
    assert coordsa != coordsc

    # coordset as coordinates
    coordse = CoordSet(x=(coord1[:3], coord2[:3]), y=coord3, z=coord0)
    assert coordse["x"].titles == CoordSet(coord1, coord2).titles[::-1]
    assert coordse["x_0"] == coord1[:3]
    assert coordse["My name is titi"] == coord3

    # iteration
    for coord in coordsa:
        assert isinstance(coord, Coord)

    assert repr(coord0) == "Coord z(wavenumber (coord0)): [float64] cm⁻¹ (size: 10)"

    coords = CoordSet(coord0.copy(), coord0)

    assert repr(coords).startswith("CoordSet: [x:wavenumber")

    with pytest.raises(InvalidCoordinatesTypeError):
        _ = CoordSet(2, 3)  # Coord in CoordSet cannot be simple scalar

    coords = CoordSet(x=coord2, y=coord3, z=None)
    assert coords.names == ["x", "y", "z"]
    assert coords.z.is_empty

    coords = CoordSet(x=coord2, y=coord3, z=np.array((1, 2, 3)))
    assert coords.names == ["x", "y", "z"]
    assert coords.z.size == 3

    with pytest.raises(InvalidDimensionNameError):
        _ = CoordSet(x=coord2, y=coord3, fx=np.array((1, 2, 3)))
        # wrong key (must be a single char)

    with pytest.raises(InvalidCoordinatesTypeError):
        _ = CoordSet(x=coord2, y=coord3, z=3)
        # wrong coordinate value

    # set a coordset from another one
    coords = CoordSet(coordse.to_dict())
    assert coordse.names == ["x", "y", "z"]
    assert coords.names == ["x", "y", "z"]
    assert coords == coordse

    # not recommended
    coords2 = CoordSet(*coordse)
    # loose the names so the ordering may be different
    assert coords2.names == ["x", "y", "z"]
    assert coords.x == coords2.z

    # invalid size
    with pytest.raises(InvalidCoordinatesSizeError):  # size for same dim does not match
        _ = CoordSet(coord0, [coord1, coord2])


def test_coordset_coordset_example():

    coord0 = Coord.linspace(10.0, 100.0, 5, units="m", title="distance")
    coord1 = Coord.linspace(20.0, 25.0, 4, units="K", title="temperature")
    coord1b = Coord.linspace(1.0, 10.0, 4, units="millitesla", title="magnetic field")
    coord2 = Coord.linspace(0.0, 1000.0, 6, units="hour", title="elapsed time")
    cs = CoordSet(v=[coord1, coord1b])
    cs = CoordSet([coord1, coord1b])
    cs = CoordSet(t=coord0, u=coord2, v=[coord1, coord1b])
    assert str(cs.u) == "Coord u(elapsed time): [float64] hr (size: 6)"
    assert str(cs.v.parent) == "CoordSet: [_0:temperature, _1:magnetic field]"
    assert str(cs.v_0) == "Coord _0(temperature): [float64] K (size: 4)"


def test_coordset_implements_method(coord0, coord1):
    c = CoordSet(coord0, coord1)
    assert c._implements("CoordSet")
    assert c._implements() == "CoordSet"


def test_coordset_available_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.available_names == DEFAULT_DIM_NAME[:-3]

    c = CoordSet(coord0, coord0.copy(), [coord1, coord1.copy()])
    assert c.available_names == DEFAULT_DIM_NAME[:-3]


def test_coordset_coords_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.coords == [coord2, coord1, coord0]


def test_coordset_cstr(coord0, coord1):
    c = CoordSet(y=[coord0, coord0.copy()], x=coord1)
    c.y.set_titles("temperature", "wavenumbers")
    assert "(_0)*" in c._cstr()
    c.y.select(1)
    assert "(_1)*" in c._cstr()


def test_coordset_has_defined_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert not c.has_defined_name
    c.name = "Toto"
    assert c.has_defined_name


def test_coordset_is_empty_property():
    c = CoordSet()
    assert c._implements("CoordSet")
    assert c.is_empty


def test_coordset_is_same_dim_property(coord0):
    c = CoordSet([coord0, coord0.copy()])
    assert c.x.is_same_dim


def test_coordset_references_property():
    c = CoordSet(x=Coord.arange(10), y="x")
    assert isinstance(c.references, dict)
    assert c.references == dict(y="x")


def test_coordset_sizes_property(coord0, coord1, coord2):
    c = CoordSet(x=coord0, y="x")
    assert c.sizes == [coord0.size]

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.sizes == [coord1.size, coord0.size, coord2.size]

    # alias
    assert c.size == [coord1.size, coord0.size, coord2.size]


def test_coordset_names_property(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.names == ["x", "y", "z"]


def test_coordset_default_property(coord0, coord1):

    c = CoordSet(y=[coord0, coord0.copy()], x=coord1)
    assert c.default is None

    # single coordinate for dimension x
    assert c.x.default is c.x

    # multiple coordinates for dimension y
    assert c.y.default.name == "_0"
    c.y.set_titles("temperature", "wavenumbers")
    c.y.select(1)
    assert c.y.default.name == "_1"
    assert c.y.default.title == "wavenumbers"
    c.y.select(-2)
    assert c.y.default.name == "_0"
    # WARNING if one freeze the value of c.y, the default cannot be changed later on.
    cy = c.y
    assert cy.default.name == "_0"
    cy.select(1)
    assert cy.default.name == "_0"  # still _0 (not chanhed!)
    assert c.y.default.name == "_1"  # but this is ok!


def test_coordset_data_property(coord0, coord1):
    c = CoordSet([coord0, coord0.copy() * 2], coord1)
    assert_array_equal(c.y.data, coord0.data)
    assert_array_equal(c.x.data, coord1.data)
    with pytest.raises(AttributeError):
        c.data


def test_coordset_name_property(coord0):
    c = CoordSet(coord0)
    assert c.name == c.id

    c.name = "Toto"
    assert c.name == "Toto"


def test_coordset_titles_property(coord0, coord1):
    c = CoordSet(coord0, coord1)
    assert c.titles == ["time-on-stream (coord1)", "wavenumber (coord0)"]

    with pytest.deprecated_call():
        _ = c.titles


def test_coordset_titles_property(coord0, coord1, coord2):
    c = CoordSet(coord0, [coord1, coord1.copy()], coord2)
    assert c.titles == [
        "temperature (coord2)",
        ["time-on-stream (coord1)", "time-on-stream (coord1)"],
        "wavenumber (coord0)",
    ]


def test_coordset_labels_property(coord0, coord1, coord2):
    c = CoordSet(coord0, [coord1, coord1.copy()], coord2)
    assert_array_equal(c.labels[0], np.array(["cold", "normal", "hot"]))


def test_coordset_unit_property(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert c.units == ["s", ["cm⁻¹", "cm⁻¹"], "K"]
    assert c.y.units == ur("cm⁻¹")


def test_coordset_keys_method(coord0, coord1, coord2):
    c = CoordSet(x=coord0, z="x", y=[coord2, coord2.copy()])
    assert c.names == c.keys()[:2]  # 'z' is not included in names but in keys()


def test_coordset_select_method(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    assert c.y.default.name == "_0"
    c.y.select(1)
    assert c.y.default.name == "_1"
    # don't use this
    cy = c.y
    cy.select(0)  # change selection from _1 to _0
    # it is not change as expected
    cy.default.name == "_1"  # not _0


def test_coordset_set_method(coord0, coord1, coord2):
    c = CoordSet(x=coord0)
    c0 = c.copy()
    assert c == c0
    assert c.x == coord0

    c0.set()
    assert c == c0, "nothing happens"

    # set with kwargs

    c0.set(x=coord1)
    assert c0.x == coord1, "x is replaced by a new coordinate!"

    c0.set(y=coord2)
    assert c0.x == coord1
    assert c0.y == coord2

    # set with args
    # append
    c0.set(coord0)
    assert c0.coords == [coord1, coord2, coord0]

    # overwrite existing
    c0.set(coord0, coord0.copy(), mode="w")
    assert c0.coords == [coord0, coord0]

    # replace selected
    c0.set(y=coord1)
    assert c0.coords == [coord0, coord1]

    # with a coordset

    ca = CoordSet([coord0[:3], coord1[:3], coord2[:3]])
    c0.set(ca, mode="w")
    assert c0.coords == ca.coords


def test_coordset_set_titles_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.title = "bibi"
    c = CoordSet(coord2, [coord0, coord0b], coord1)
    assert c.titles == [
        "time-on-stream (coord1)",
        ["wavenumber (coord0)", "bibi"],
        "temperature (coord2)",
    ]

    c.set_titles("time", "dddd", "celsius")
    assert (
        str(c) == "CoordSet: [x:time, y:[_0:wavenumber (coord0), _1:bibi], z:celsius]"
    )

    c.set_titles(x="time", z="celsius", y_0="length")
    assert str(c) == repr(c) == "CoordSet: [x:time, y:[_0:length, _1:bibi], z:celsius]"

    c.set_titles("t", ("l", "g"), x="x")
    assert str(c) == "CoordSet: [x:x, y:[_0:l, _1:g], z:celsius]"

    c.set_titles(("t", ("l", "g")), z="z")
    assert str(c) == "CoordSet: [x:t, y:[_0:l, _1:g], z:z]"

    c.set_titles()  # nothing happens
    assert str(c) == "CoordSet: [x:t, y:[_0:l, _1:g], z:z]"


def test_coordset_set_units_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.title = "bibi"
    c = CoordSet(coord2, [coord0, coord0b], coord1)

    assert c.units == ["s", ["cm⁻¹", "cm⁻¹"], "K"]

    with pytest.raises(DimensionalityError) as e:  # because units doesn't match
        c.set_units(("s", ("m", "s")), z="radian")
    assert (
        str(e.value)
        == "Cannot convert from '[length]^-1' (centimeter^-1) to '[time]' (second)"
    )

    c.set_units(("km/s", ("m", "s")), z="radian", force=True)  # force change
    assert c.y_0.units == ur("m")


def test_coordset_update_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.title = "bibi"

    c = CoordSet(coord2, [coord0, coord0b], coord1)
    assert c.names == ["x", "y", "z"]
    assert (
        str(c.coords) == "[Coord x(time-on-stream (coord1)): [float64] s (size: 100), "
        "CoordSet: [_0:wavenumber (coord0), _1:bibi], "
        "Coord z(temperature (coord2)): [float64] K (size: 3)]"
    )

    c.update(x=coord2)
    assert (
        str(c.coords) == "[Coord x(temperature (coord2)): [float64] K (size: 3), "
        "CoordSet: [_0:wavenumber (coord0), _1:bibi], "
        "Coord z(temperature (coord2)): [float64] K (size: 3)]"
    )

    c.update(y_0=coord1)
    assert (
        str(c.coords) == "[Coord x(temperature (coord2)): [float64] K (size: 3), "
        "CoordSet: [_0:time-on-stream (coord1), _1:bibi], "
        "Coord z(temperature (coord2)): [float64] K (size: 3)]"
    )

    with pytest.raises(InvalidDimensionNameError):
        c.update(k=coord1)


def test_coordset_loc2index_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.title = "bibi"

    c = CoordSet(coord2, [coord0, coord1[:10]], coord0)

    with pytest.raises(AttributeError):
        # no loc2index for coordset except
        # if the coordset describe the same dimension (but this cannot be called
        # directly)
        c.loc2index(3000.0)

    idx = c.y.loc2index(3000.0)
    assert idx == 3
    assert c.y_0[idx].value == 3000 * ur("1/cm")

    c.y.select(2)  # change default
    idx2 = c.y.loc2index(4.8)
    assert_approx_equal(c.y_1[idx2].m, 4.848, significant=3)

    assert c.y.loc2index(20) == c.y_1.loc2index(20.0)


def test_coordset_call_method(coord0, coord1):
    coordsa = CoordSet(coord0, coord1)

    assert (
        str(coordsa) == "CoordSet: [x:time-on-stream (coord1), y:wavenumber (coord0)]"
    )
    a = coordsa(1, 0)
    assert a == coordsa

    b = coordsa(1)
    assert b == coord0  # reordering

    c = coordsa("x")
    assert c == coord1

    d = coordsa("time-on-stream (coord1)")
    assert d == coord1

    with pytest.raises(InvalidDimensionNameError):
        coordsa("x_a")  # do not exit

    coordsa("y_a")


def test_coordset_getitem_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0[:5], coord0[5:]], coord1)

    coord = c["temperature (coord2)"]
    assert (
        repr(coord) == str(coord) == "Coord z(temperature (coord2)): [float64] K ("
        "size: 3)"
    )
    assert coord.name == "z"
    assert c.z == coord

    assert c.y == c.y_0  # by default in a multicoordinate y is the first coordinate
    # or the one defined by select
    c.y.select(1)
    assert c.y == c.y_1

    coord = c["wavenumber (coord0)"]
    assert coord.name == "_0"

    coord = c["y_1"]
    assert coord.name == "_1"

    coord = c["_0"]
    assert coord.name == "_0"

    with pytest.raises(TypeError):
        c[3000.0]

    c.y.select(0)
    val = c.y[3000.0]
    assert val == 3000 * ur("1/cm")

    c.y.select(1)  # change default
    val = c.y[3000.0]  # out of limits
    assert int(val.m) == 2333


def test_coordset_del_item_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_0:wavenumber (coord0), _"
        "1:wavenumber (coord0)], z:temperature (coord2)]"
    )

    del c["temperature (coord2)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_0:wavenumber (coord0), _"
        "1:wavenumber (coord0)]]"
    )

    with pytest.raises(AttributeError):
        del c.y["wavenumber (coord0)"]  # c.y return the default coordinate not the
        # coordset
    # one must use to access the parent coordset
    del c.y.parent["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0)]]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y_1
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_0:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y_0
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    # execute some uncovered part of the coordset methods

    # _attributes
    assert "is_same_dim" in c._attributes()

    # __setattr__ error
    with pytest.raises(AttributeError):
        del c.temperature

    # __getattr__ error
    with pytest.raises(AttributeError):
        _ = c.temperature


def test_coordset_copy_method(coord0, coord1):
    coord2 = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")

    coordsa = CoordSet(coord0, coord1, coord2)

    coordsb = coordsa.copy()
    assert coordsa == coordsb
    assert coordsa is not coordsb
    assert coordsa(1) == coordsb(1)
    assert coordsa(1).name == coordsb(1).name

    # copy
    coords = CoordSet(coord0, coord0.copy())
    coords1 = coords[:]
    assert coords is not coords1

    import copy

    coords2 = copy.deepcopy(coords)
    assert coords == coords2  #


def test_coordset_str_repr_method(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream (coord1), y:[_0:wavenumber (coord0), _"
        "1:wavenumber (coord0)], z:temperature (coord2)]"
    )
    assert repr(coords) == str(coords)

    # _repr_html
    assert coords._repr_html_().startswith("<table style='background:transparent'>")


def test_coordset_set_item_method(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream (coord1), y:[_0:wavenumber (coord0), _"
        "1:wavenumber (coord0)], z:temperature (coord2)]"
    )
    coords.set_titles("time-on-stream", ("wavenumber", "wavenumber"), "temperature")

    # set item

    coords["z"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_0:wavenumber, _1:wavenumber], "
        "z:temperature (coord2)]"
    )

    with pytest.raises(KeyError):
        coords["temperature"] = coord1

    coords["temperature (coord2)"] = coord1
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_0:wavenumber, _1:wavenumber], "
        "z:time-on-stream (coord1)]"
    )

    coords["y_1"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_0:wavenumber, _1:temperature (coord2)], "
        "z:time-on-stream (coord1)]"
    )

    coords["_0"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_0:temperature (coord2), _"
        "1:temperature (coord2)], z:time-on-stream (coord1)]"
    )

    coords["t"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_0:temperature (coord2), _"
        "1:temperature (coord2)], z:time-on-stream (coord1), t:temperature (coord2)]"
    )

    coord2.title = "zaza"
    coords["temperature (coord2)"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_0:temperature (coord2), _"
        "1:temperature (coord2)], z:time-on-stream (coord1), t:zaza]"
    )

    coords["temperature (coord2)"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_0:zaza, _1:temperature (coord2)], "
        "z:time-on-stream (coord1), t:zaza]"
    )

    coords.set(coord1, coord0, coord2, mode="w")
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:wavenumber (coord0), z:time-on-stream (coord1)]"
    )

    coords.z = coord0
    assert (
        str(coords)
        == "CoordSet: [x:zaza, y:wavenumber (coord0), z:wavenumber (coord0)]"
    )

    coords.zaza = coord0
    assert (
        str(coords) == "CoordSet: [x:wavenumber (coord0), "
        "y:wavenumber (coord0), z:wavenumber (coord0)]"
    )
