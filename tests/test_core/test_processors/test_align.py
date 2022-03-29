# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


""" Tests for the interpolate module

"""
import numpy as np
import pytest
from spectrochempy import align
import spectrochempy as scp
from spectrochempy.utils.testing import RandomSeedContext

# ========
# FIXTURES
# ========

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2

coord0_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)


coord1_ = scp.Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream")


coord2_ = scp.Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)

coord0_2_ = scp.Coord(
    data=np.linspace(4000.0, 1000.0, 9),
    labels=list("abcdefghi"),
    units="cm^-1",
    title="wavenumber",
)


coord1_2_ = scp.Coord(
    data=np.linspace(0.0, 60.0, 50), units="s", title="time-on-stream"
)


coord2_2_ = scp.Coord(
    data=np.linspace(200.0, 1000.0, 4),
    labels=["cold", "normal", "hot", "veryhot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def ds1():
    # a dataset with coordinates
    return scp.NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coord2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ds2():
    # another dataset
    return scp.NDDataset(
        ref3d_2_data,
        coordset=[coord0_2_, coord1_2_, coord2_2_],
        title="absorbance",
        units="absorbance",
    ).copy()


# align
# -------


def test_align(ds1, ds2):
    ds1c = ds1.copy()
    dss = ds1c.align(ds2, dim="x")  # first syntax
    # TODO: flag copy=False raise an error
    assert dss is not None
    ds3, ds4 = dss  # a tuple is returned

    assert ds3.shape == (10, 100, 6)

    # TODO: labels are not aligned

    dss2 = align(ds1, ds2, dim="x")  # second syntax
    assert dss2 == dss
    assert dss2[0] == dss[0]
    ds5, ds6 = dss2

    # align another dim
    dss3 = align(ds1, ds2, dim="z")  # by default it would be the 'x' dim
    ds7, ds8 = dss3
    assert ds7.shape == (17, 100, 3)

    # align two dims
    dss4 = align(ds1, ds2, dims=["x", "z"])
    ds9, ds10 = dss4

    # align inner
    a, b = align(ds1, ds2, method="inner")
