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
from spectrochempy.core.dataset.basearrays.ndlabeledarray import NDLabeledArray
from spectrochempy.core.common.exceptions import LabelsError, ShapeError
from spectrochempy.utils.testing import (
    assert_array_equal,
)

from spectrochempy.utils import check_docstrings as td


# ######################
# TEST NDLabeledArray  #
# ######################

# test docstring
def test_ndlabeledarray_docstring():
    td.PRIVATE_CLASSES = []  # override default to test private class docstring
    module = "spectrochempy.core.dataset.basearrays.ndlabeledarray"
    td.check_docstrings(
        module,
        obj=spectrochempy.core.dataset.basearrays.ndlabeledarray.NDLabeledArray,
        exclude=["SA01", "EX01"],
    )


def test_ndlabeledarray_init():
    nd = NDLabeledArray(labels=None)
    assert not nd.is_labeled
    assert nd.size == 0
    assert nd.shape == ()
    assert nd.is_empty
    assert nd.get_labels() is None

    # Without data
    nd = NDLabeledArray(labels=list("abcdefghij"))
    assert nd.is_labeled
    assert np.all(nd.data == nd.get_labels(level=0))
    assert nd.ndim == 1
    assert nd.shape == (10,)
    assert nd.size == 10
    assert list(nd.get_labels()) == list("abcdefghij")
    assert nd.get_labels(level=1) is None

    # With data
    nd = NDLabeledArray(data=np.arange(10), labels=list("abcdefghij"))
    assert nd.is_labeled
    assert nd.ndim == 1
    assert nd.data.shape == (10,)
    assert nd.shape == (10,)

    # 2D (not allowed)
    with pytest.raises(ShapeError):
        d = np.random.random((10, 10))
        NDLabeledArray(data=d, labels=list("abcdefghij"))

    # 2D but with the one of the dimensions being of size one.
    d = np.random.random((1, 10))
    nd = NDLabeledArray(data=d, labels=list("abcdefghij"))
    assert nd.is_labeled
    assert nd.ndim == 1
    assert nd.data.shape == (10,)
    assert nd.shape == (10,)

    d = np.random.random((10, 1))
    nd = NDLabeledArray(data=d, labels=list("abcdefghij"))
    assert nd.is_labeled
    assert nd.ndim == 1
    assert nd.data.shape == (10,)
    assert nd.shape == (10,)

    # multidimensional labels
    nd = NDLabeledArray(
        data=np.arange(10), labels=[list("abcdefghij"), list("klmnopqrst")]
    )
    assert nd.is_labeled
    assert nd.ndim == 1
    assert nd.data.shape == (10,)
    assert nd.shape == (10,)

    d = np.random.random((10, 1))
    nd = NDLabeledArray(data=d, labels=[list("abcdefghij"), list("klmnopqrst")])
    assert nd.is_labeled

    d = np.random.random((10, 1))
    l = np.array([list("abcdefghij"), list("klmnopqrst")])
    nd = NDLabeledArray(data=d, labels=l)
    assert nd.is_labeled

    # transposed labels
    nd = NDLabeledArray(data=d, labels=l.T)
    assert nd.is_labeled

    l = np.array([list("abcdefghijx"), list("klmnopqrstx")])
    with pytest.raises(LabelsError):
        NDLabeledArray(data=d, labels=l)


def test_ndlabeledarray_getitem():
    # slicing only-title array
    nd = NDLabeledArray(labels=list("abcdefghij"))
    assert nd[1].labels == ["b"]
    assert nd[1].values == "b"
    assert nd["b"].values == "b"
    assert nd["c":"d"].shape == (2,)
    assert nd.__getitem__("b", return_index=True)[-1] == (slice(1, 2, 1),)

    assert_array_equal(nd["c":"d"].values, np.array(["c", "d"]))
    assert_array_equal(nd["c":"j":2].values, np.array(["c", "e", "g", "i"]))

    # multilabels
    nd = NDLabeledArray(
        data=np.arange(10),
        labels=[list("abcdefghij"), list("0123456789"), list("lmnopqrstx")],
    )
    assert_array_equal(nd["o":"x":2].values, np.array([3, 5, 7, 9]))

    nd._data = np.arange("2020", "2030", dtype="<M8[Y]")
    assert_array_equal(
        nd["o":"x":2].values, np.arange("2023", "2030", 2, dtype="<M8[Y]")
    )
    assert_array_equal(
        nd["2023":"2029":2].values, np.arange("2023", "2030", 2, dtype="<M8[Y]")
    )


def test_ndlabeledarray_sort():
    # labels and sort

    nd = NDLabeledArray(
        np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        name="wavelength",
    )
    assert nd.is_labeled
    assert list(nd.get_labels()) == list("abcdefghij")

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
    # add rows of labels to d0
    nd.labels = "bc cd de ef ab fg gh hi ja ij ".split()
    nd.labels = list("0123456789")
    nd.labels = [list("klmnopqrst"), list("uvwxyzéèàç")]
    assert list(nd.get_labels(level=2)) == list("0123456789")
    d1 = nd._sort()
    assert d1.data[0] == 1000
    assert_array_equal(d1.labels[:, 0], ["j", "ij", "9", "t", "ç"])

    d1._sort(descend=True, inplace=True)
    assert d1.data[0] == 4000
    assert_array_equal(d1.labels[:, 0], ["a", "bc", "0", "k", "u"])

    d1 = d1._sort(by="label[1]", descend=True)
    assert np.all(d1.labels[:, 0] == ["i", "ja", "8", "s", "à"])

    # other way
    d2 = d1._sort(by="label", level=1, descend=True)
    assert np.all(d2.labels[:, 0] == d1.labels[:, 0])

    d3 = d1.copy()
    d3._labels = None
    with pytest.raises(KeyError):
        # no label!
        d3._sort(by="label", level=6, descend=True)


def test_ndlabeledarray_str_repr_summary():

    nd = NDLabeledArray(
        name="toto",
        data=np.arange("2020", "2030", dtype="<M8[Y]"),
        labels=list("abcdefghij"),
    )
    assert repr(nd) == "NDLabeledArray toto(value): [datetime64[Y]] unitless (size: 10)"

    # repr_html and summary
    nd = NDLabeledArray([1, 2, 3], labels=list("abc"))
    assert "data:" in nd._cstr()[2] and "labels:" in nd._cstr()[2]

    nd = NDLabeledArray(labels=list("abc"))
    assert "data:" in nd._cstr()[2] and "labels:" not in nd._cstr()[2]

    nd = NDLabeledArray(labels=list("abcdefghij"), name="z")
    assert nd.summary.startswith("\x1b[32m         name")
    assert (
        str(nd)
        == repr(nd)
        == "NDLabeledArray z(value): [labels] [  a   b ...   i   j] (size: 10)"
    )

    nd = NDLabeledArray(
        labels=[list("abcdefghij"), list("0123456789"), list("lmnopqrstx")]
    )
    assert "labels[1]" in nd.summary
    assert "labels[1]" in nd._repr_html_()


def test_ndlabeledarray_data():
    nd = NDLabeledArray(
        np.linspace(4000, 1000, 10),
        labels=list("abcdefghij"),
        units="s",
        name="wavelength",
    )

    assert np.all(nd.data == np.linspace(4000, 1000, 10))
    nd = NDLabeledArray(
        labels=[
            list("abcdefghij"),
        ],
        name="wavelength",
    )

    assert np.all(nd.data == nd.get_labels(level=0))
