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
from quaternion import as_quat_array

import spectrochempy
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndcomplexarray import NDComplexArray
from spectrochempy.core.common.exceptions import (
    CastingError,
    ShapeError,
)
from spectrochempy.core.units import ur
from spectrochempy.core.common.constants import (
    TYPE_COMPLEX,
    TYPE_INTEGER,
)
from spectrochempy.utils.testing import (
    RandomSeedContext,
    assert_approx_equal,
    assert_array_equal,
    assert_equal,
)

from spectrochempy.utils import check_docstrings as td

typequaternion = np.dtype(np.quaternion)

# create reference arrays
# --------------------------------------------------------------------------------------

with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0


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
    module = "spectrochempy.core.dataset.basearrays.ndcomplexarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.basearrays.ndcomplexarray.NDComplexArray,
        exclude=["SA01", "EX01"],
    )


def test_ndcomplexarray_init():
    # test with complex data in the last dimension
    # A list
    nd = NDComplexArray()
    assert not nd.has_complex_dims
    assert not nd.is_complex
    assert not nd.is_hypercomplex
    assert str(nd) == "NDComplexArray (value): empty"
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
        NDArray(arr)
    nd1 = NDComplexArray(arr)
    nd2 = NDComplexArray(nd1)
    assert nd1 is not nd2  # shallow copy only data
    assert nd1.data is nd2.data  # by default copy is false

    # NDComplexarray can also accept other type at initialisation
    nd = NDComplexArray([25])
    assert nd.data == np.array([25])
    assert nd.data.dtype in TYPE_INTEGER

    with pytest.raises(ShapeError):
        # must have even number of colums
        NDComplexArray([25], dtype=np.complex128)
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
    with pytest.raises(ShapeError):  # not even
        NDComplexArray(d, dtype=typequaternion)

    d = np.random.random((3, 3)) * np.exp(0.1j)
    with pytest.raises(ShapeError):
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
    nd = NDComplexArray(d)
    with pytest.raises(AttributeError):
        nd.X
    with pytest.raises(ValueError):
        nd.R
    with pytest.raises(ValueError):
        nd.component("R")
    assert nd.imag is None
    assert nd.real is nd
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
    with pytest.raises(AttributeError):
        assert nd.X
    with pytest.raises(ValueError):
        nd.component("X")
    assert_array_equal(nd.RR, nd.real)
    assert_array_equal(nd.RI, nd.imag)
    with pytest.raises(ShapeError):
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
    with pytest.raises(AttributeError):
        nd.RX
    with pytest.raises(ValueError):
        nd.component("RX")
    np.random.seed(12345)
    d = np.random.random((2, 2)) * np.exp(0.1j)
    d3 = NDComplexArray(d)
    new = d3.copy()
    new.data = d3.real.data + 1j * d3.imag.data
    assert_equal(d3.data, new.data)
    d4 = NDComplexArray(d, dtype="complex128")
    new = d4.copy()
    new.data = d4.real.data + 1j * d4.imag.data
    assert_equal(d4.data, new.data)

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
    with pytest.raises(ShapeError):
        nd2 = nd[..., :3]  # not enough data in the last dimension
        nd2.set_hypercomplex(quat_array=True)

    # use of interlaced float array
    with pytest.raises(ShapeError):
        # too many dimensions
        nd.set_hypercomplex()

    d = np.arange(25).reshape(5, 5)
    nd = NDComplexArray(d)
    assert not nd.is_complex
    assert not nd.is_hypercomplex
    with pytest.raises(ShapeError):
        # not suitable shape (last dimension)
        nd.set_hypercomplex()
    nd = nd[..., :4]
    with pytest.raises(ShapeError):
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
