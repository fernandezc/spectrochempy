# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


from copy import copy

import numpy as np
import pytest
from pint.errors import DimensionalityError
from traitlets import HasTraits, TraitError

import spectrochempy as scp
from spectrochempy.core import debug_
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import Quantity, ur
from spectrochempy.core.common.exceptions import (
    InvalidCoordinatesSizeError,
    InvalidCoordinatesTypeError,
    InvalidDimensionNameError,
    ShapeError,
)
from spectrochempy.utils.testing import (
    assert_approx_equal,
    assert_array_equal,
    assert_produces_log_warning,
    assert_units_equal,
)
from spectrochempy.utils.traits import Range

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


def test_coord_init():

    # simple coords

    a = Coord([1, 2, 3], name="x")
    assert a.id is not None
    assert not a.is_empty
    assert_array_equal(a.data, np.array([1, 2, 3]))
    assert not a.is_labeled
    assert a.units is None
    assert a.is_unitless
    debug_(a.meta)
    assert not a.meta
    assert a.name == "x"

    # set properties

    a.title = "xxxx"
    assert a.title == "xxxx"
    a.name = "y"
    assert a.name == "y"
    a.meta = None
    a.meta = {"val": 125}  # need to be an OrderedDic
    assert a.meta["val"] == 125

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


def test_linearcoord_deprecated():
    from spectrochempy.core.dataset.coord import LinearCoord

    with assert_produces_log_warning(DeprecationWarning):
        _ = LinearCoord(title="z")


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


def test_coord_functions():

    coord0 = Coord.linspace(200.0, 300.0, 3, units="K", title="temperature")
    coord1 = Coord(np.linspace(200.0, 300.0, 3), units="K", title="temperature")
    assert coord1 == coord0
