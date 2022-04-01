# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

import numpy as np
import pytest

import spectrochempy
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndmaskedcomplexarray import (
    NDMaskedComplexArray,
)
from spectrochempy.core.units import Quantity, ur
from spectrochempy.core.common.constants import (
    MASKED,
    NOMASK,
)
from spectrochempy.utils.testing import (
    RandomSeedContext,
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


# ###########################
# TEST NDMaskedComplexArray #
# ###########################


@pytest.fixture(scope="module")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, comment="An array", copy=True)


@pytest.fixture(scope="module")
def ndarraymask():
    # return a simple ndarray with some data, units, and masks
    return NDMaskedComplexArray(ref_data, mask=ref_mask, units="m/s", copy=True)


# test docstring
def test_ndmaskedcomplexarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.basearrays.ndmaskedcomplexarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.basearrays.ndmaskedcomplexarray.NDMaskedComplexArray,
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

    nd = NDMaskedComplexArray()
    assert str(nd) == f"NDMaskedComplexArray (value): empty"
    assert not nd.is_masked
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

    data = np.ma.array([1, 2], mask=[True, False])
    nd = NDMaskedComplexArray(data)
    assert np.all(nd.mask == [True, False])
    assert repr(nd) == "NDMaskedComplexArray (value): [int64] unitless (size: 2)"

    d = np.ones((2, 1)) * np.exp(0.1j)
    nd = NDMaskedComplexArray(d, dtype=typequaternion, mask=NOMASK)
    assert nd.shape == (1, 1)
    assert not nd.is_masked
    nd.mask = [[True]]
    assert nd.is_masked
    nd.mask = True
    nd.mask = NOMASK
    nd.mask = MASKED
    with pytest.raises(ValueError):
        # bad shape
        nd.mask = [True]


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


def test_ndmaskedcomplexarray_summary():
    data = np.ma.array([1, 2], mask=[True, False])
    nd = NDMaskedComplexArray(data)
    assert np.all(nd.mask == [True, False])
    assert repr(nd) == "NDMaskedComplexArray (value): [int64] unitless (size: 2)"
    assert "[34m[      --        2][39m" in nd.summary
