# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Test coordset
"""
# flake8: noqa


from copy import copy

import numpy as np
import pytest
from pint.errors import DimensionalityError
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.units import ur
from spectrochempy.core.common.exceptions import (
    InvalidCoordinatesSizeError,
    InvalidCoordinatesTypeError,
    InvalidDimensionNameError,
)
from spectrochempy.utils.testing import (
    assert_approx_equal,
    assert_array_equal,
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


# ======================================================================================
# CoordSet Tests
# ======================================================================================
def test_coordset_init(coord0, coord1, coord2):

    coord3 = coord2.copy()
    coord3.title = "My name is titi"

    # First syntax
    # Coordinates are sorted in the coordset
    coordsa = CoordSet(coord0, coord3, coord2)
    # by convention if the names are not specified, then the coordinates follows the
    # order of dims, so they are in reverse order with respect of the coorset where
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
    assert coordsa1.x.names == ["_1", "_2", "_3"]

    # Third syntax
    coordsc = CoordSet(x=coord2, y=coord3, z=coord0)
    assert coordsc.names == ["x", "y", "z"]

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
    assert coordse["x"].titles == CoordSet(coord1, coord2).titles
    assert coordse["x_1"] == coord2
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
    coords = CoordSet(**coordse)
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


def test_coordset_implements_method(coord0, coord1):
    c = CoordSet(coord0, coord1)
    assert c.implements("CoordSet")
    assert c.implements() == "CoordSet"


# read_only properties


def test_coordset_available_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.available_names == DEFAULT_DIM_NAME[:-3]

    c = CoordSet(coord0, coord0.copy(), [coord1, coord1.copy()])
    assert c.available_names == DEFAULT_DIM_NAME[:-3]


def test_coordset_coords_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert c.coords == [coord2, coord1, coord0]


def test_coordset_has_defined_names_property(coord0, coord1, coord2):
    c = CoordSet(coord0, coord1, coord2)
    assert not c.has_defined_name
    c.name = "Toto"
    assert c.has_defined_name


def test_coordset_is_empty_property():
    c = CoordSet()
    assert c.implements("CoordSet")
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
    c = CoordSet([coord0, coord0.copy()], coord1)
    assert c.default is None
    cx = c.x
    assert cx.default is cx
    cy = c.y
    assert cy.default.name == "_1"


def test_coordset_data_property(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    assert_array_equal(c.y.data, coord0.data)
    assert_array_equal(c.x.data, coord1.data)
    assert c.data is None


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
    assert c.y.units == ["cm⁻¹", "cm⁻¹"]


# public methods


def test_coordset_keys_method(coord0, coord1, coord2):
    c = CoordSet(x=coord0, z="x", y=[coord2, coord2.copy()])
    assert c.names == c.keys()[:2]  # 'z' is not included in names but in keys()


def test_coordset_select_method(coord0, coord1):
    c = CoordSet([coord0, coord0.copy()], coord1)
    cy = c.y
    assert cy.default.name == "_1"
    cy.select(2)
    assert cy.default.name == "_2"


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
        str(c) == "CoordSet: [x:time, y:[_1:wavenumber (coord0), _2:bibi], z:celsius]"
    )

    c.set_titles(x="time", z="celsius", y_1="length")
    assert str(c) == repr(c) == "CoordSet: [x:time, y:[_1:length, _2:bibi], z:celsius]"

    c.set_titles("t", ("l", "g"), x="x")
    assert str(c) == "CoordSet: [x:x, y:[_1:l, _2:g], z:celsius]"

    c.set_titles(("t", ("l", "g")), z="z")
    assert str(c) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"

    c.set_titles()  # nothing happens
    assert str(c) == "CoordSet: [x:t, y:[_1:l, _2:g], z:z]"


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
    assert c.y_1.units == ur("m")


def test_coordset_update_method(coord0, coord1, coord2):
    coord0b = coord0.copy()
    coord0b.title = "bibi"

    c = CoordSet(coord2, [coord0, coord0b], coord1)
    assert c.names == ["x", "y", "z"]
    assert (
        str(c.coords) == "[Coord x(time-on-stream (coord1)): [float64] s (size: 100), "
        "CoordSet: [_1:wavenumber (coord0), _2:bibi], "
        "Coord z(temperature (coord2)): [float64] K (size: 3)]"
    )

    c.update(x=coord2)
    assert (
        str(c.coords) == "[Coord x(temperature (coord2)): [float64] K (size: 3), "
        "CoordSet: [_1:wavenumber (coord0), _2:bibi], "
        "Coord z(temperature (coord2)): [float64] K (size: 3)]"
    )

    c.update(y_1=coord1)
    assert (
        str(c.coords) == "[Coord x(temperature (coord2)): [float64] K (size: 3), "
        "CoordSet: [_1:time-on-stream (coord1), _2:bibi], "
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
    assert c.y_1[idx].value == 3000 * ur("1/cm")

    c.y.select(2)  # change default
    idx2 = c.y.loc2index(4.8)
    assert_approx_equal(c.y_2[idx2].m, 4.848, significant=3)

    assert c.y.loc2index(20) == c.y_2.loc2index(20.0)


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


def test_coordset_get_item_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0[:5], coord0[5:]], coord1)

    coord = c["temperature (coord2)"]
    assert (
        repr(coord) == str(coord) == "Coord z(temperature (coord2)): [float64] K ("
        "size: 3)"
    )
    assert coord.name == "z"

    coord = c["wavenumber (coord0)"]
    assert coord.name == "_1"

    coord = c["y_2"]
    assert coord.name == "_2"

    coord = c["_1"]
    assert coord.name == "_1"

    with pytest.raises(TypeError):
        c[3000.0]

    val = c.y[3000.0]
    assert val == 3000 * ur("1/cm")

    c.y.select(2)  # change default
    val = c.y[3000.0]  # out of limits
    assert int(val.m) == 2333


def test_coordset_del_item_method(coord0, coord1, coord2):
    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)

    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _"
        "2:wavenumber (coord0)], z:temperature (coord2)]"
    )

    del c["temperature (coord2)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _"
        "2:wavenumber (coord0)]]"
    )

    del c.y["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)]]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c["wavenumber (coord0)"]
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y_2
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    c = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    del c.y._1
    assert (
        str(c)
        == repr(c)
        == "CoordSet: [x:time-on-stream (coord1), y:[_2:wavenumber (coord0)], "
        "z:temperature (coord2)]"
    )

    # execute some uncovered part of the coordset methods

    # __dir__
    assert "is_same_dim" in c.__dir__()

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
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _"
        "2:wavenumber (coord0)], z:temperature (coord2)]"
    )
    assert repr(coords) == str(coords)

    # _repr_html
    assert coords._repr_html_().startswith("<table style='background:transparent'>")


def test_coordset_set_item_method(coord0, coord1, coord2):
    coords = CoordSet(coord2, [coord0, coord0.copy()], coord1)
    assert (
        str(coords)
        == repr(coords)
        == "CoordSet: [x:time-on-stream (coord1), y:[_1:wavenumber (coord0), _"
        "2:wavenumber (coord0)], z:temperature (coord2)]"
    )
    coords.set_titles("time-on-stream", ("wavenumber", "wavenumber"), "temperature")

    # set item

    coords["z"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], "
        "z:temperature (coord2)]"
    )

    with pytest.raises(KeyError):
        coords["temperature"] = coord1

    coords["temperature (coord2)"] = coord1
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:wavenumber], "
        "z:time-on-stream (coord1)]"
    )

    coords["y_2"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:wavenumber, _2:temperature (coord2)], "
        "z:time-on-stream (coord1)]"
    )

    coords["_1"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _"
        "2:temperature (coord2)], z:time-on-stream (coord1)]"
    )

    coords["t"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _"
        "2:temperature (coord2)], z:time-on-stream (coord1), t:temperature (coord2)]"
    )

    coord2.title = "zaza"
    coords["temperature (coord2)"] = coord2
    assert (
        str(coords) == "CoordSet: [x:time-on-stream, y:[_1:temperature (coord2), _"
        "2:temperature (coord2)], z:time-on-stream (coord1), t:zaza]"
    )

    coords["temperature (coord2)"] = coord2
    assert (
        str(coords)
        == "CoordSet: [x:time-on-stream, y:[_1:zaza, _2:temperature (coord2)], "
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
