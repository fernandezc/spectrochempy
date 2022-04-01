# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
This module implements the NDArray derived object NDLabeledArray
"""

import re

import numpy as np
import traitlets as tr
from traittypes import Array

from spectrochempy.core import info_, warning_
from spectrochempy.core.common.exceptions import LabelsError, LabelWarning
from spectrochempy.core.common.print import numpyprintoptions
from spectrochempy.core.dataset.basearrays.ndarray import NDArray, _docstring
from spectrochempy.core.units import Quantity

# Printing settings
# --------------------------------------------------------------------------------------
numpyprintoptions()


# ======================================================================================
# NDArray subclass : NDLabeledArray
# ======================================================================================
class NDLabeledArray(NDArray):
    __doc__ = _docstring.dedent(
        """
    A |NDArray| derived class with additional labels and related functionalities.

    The private |NDLabeledArray| class is an array (numpy |ndarray|-like)
    container, usually not intended to be used directly. In addition to the
    |NDArray| functionalities, this class adds labels and related methods and
    properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    labels : list, tuple, or |ndarray|-like object, optional
        Labels for the `data`.
        The labels array may have an additional dimension, meaning several
        series of labels for the same data.
    """
    )
    _docstring.get_sections(__doc__, base="NDLabeledArray")

    _labels = Array(allow_none=True)

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        self.labels = kwargs.pop("labels", None)

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __getitem__(self, items, return_index=False):
        new, keys = super().__getitem__(items, return_index=True)
        if self.is_labeled:
            if new._labels.ndim == 1:
                new._labels = np.array(self._labels[keys[0]])
            else:
                new._labels = np.array(self._labels[:, keys[0]])
        if not return_index:
            return new
        return new, keys

    def __str__(self):
        rep = super().__str__()
        if "[object]" in rep and self.is_labeled:
            # no data but labels
            lab = self.get_labels(level=0)
            data = f" {lab}"
            size = f" (size: {len(lab)})"
            dtype = "labels"
            body = f"[{dtype}]{data}{size}"
            rep = f"{rep.split(':', maxsplit=1)[0]}: {body}"
        return rep

    # ----------------------------------------------------------------------------------
    # Private properties and methods
    # ----------------------------------------------------------------------------------
    def _argsort(self, descend=False, by="value", level=None):
        # found the indices sorted by values or labels
        args = self.data
        if by == "value":
            args = np.argsort(self.data)
        elif "label" in by and not self.is_labeled:
            raise KeyError("no label to sort")
        elif "label" in by and self.is_labeled:
            labels = self.labels
            if len(self.labels.shape) > 1:
                # multidimensional labels
                if not level:
                    level = 0
                    # try to find a pos in the by string
                    pattern = re.compile(r"label\[(\d)]")
                    p = pattern.search(by)
                    if p is not None:
                        level = int(p[1])
                labels = self.labels[level]
            args = np.argsort(labels)
        if descend:
            args = args[::-1]
        return args

    def _attributes(self, removed=[]):
        return super()._attributes(removed) + ["labels"]

    def _is_labels_allowed(self):
        return self._squeeze_ndim <= 1

    def _interpret_strkey(self, key):

        if self.is_labeled:
            labels = self._labels
            indexes = np.argwhere(labels == key).flatten()
            if indexes.size > 0:
                return indexes[-1], None

        # key can be a date
        try:
            key = np.datetime64(key)
        except ValueError as exc:
            raise KeyError

        index, error = self._value_to_index(key)
        return index, error

    def _get_data_to_print(self, ufmt=" {:~K}"):
        units = ""
        data = self.values
        if isinstance(data, Quantity):
            data = data.magnitude
            units = ufmt.format(self.units) if self.has_units else ""
        return data, units

    @tr.default("_labels")
    def __labels_default(self):
        return None

    # def _repr_shape(self):
    #     if self.is_empty:
    #         return "size: 0"
    #     out = ""
    #     shape_ = (
    #         x for x in itertools.chain.from_iterable(list(zip(self.dims, self.shape)))
    #     )
    #     shape = (", ".join(["{}:{}"] * self.ndim)).format(*shape_)
    #     size = self.size
    #     out += f"size: {size}" if self.ndim < 2 else f"shape: ({shape})"
    #     return out
    #

    # def _repr_value(self):
    #     numpyprintoptions(precision=4, edgeitems=0, spc=1, linewidth=120)
    #     prefix = f"{type(self).__name__} ({self.name}): "
    #     units = ""
    #     if not self.is_empty:
    #         if self.data is not None:
    #             dtype = self.dtype
    #             data = ""
    #             units = " {:~K}".format(self.units) if self.has_units else " unitless"
    #         else:
    #             # no data but labels
    #             lab = self.get_labels()
    #             data = f" {lab}"
    #             dtype = "labels"
    #         body = f"[{dtype}]{data}"
    #     else:
    #         body = "empty"
    #     numpyprintoptions()
    #     return "".join([prefix, body, units])

    @property
    def _squeeze_ndim(self):
        # The number of dimensions of the squeezed`data` array (Readonly property).
        if self.data is None and self.is_labeled:
            return 1
        return super()._squeeze_ndim

    @staticmethod
    def _get_label_str(lab, idx=None, sep="\n"):
        arraystr = np.array2string(lab, separator=" ", prefix="")
        if idx is None:
            text = "       labels: "
        else:
            text = f"    labels[{idx}]: "
        text += f"\0{arraystr}\0{sep}"
        return text

    def _str_value(
        self, sep="\n", ufmt=" {:~K}", prefix="", header="         data: ... \n"
    ):
        text = super()._str_value(sep, ufmt, prefix, header)
        text += sep

        if self.is_labeled:
            lab = self.labels
            idx = 0
            if self.data is None:
                if lab.ndim > 1:
                    idx = 1
                else:
                    # already in data
                    lab = None
            if lab is not None:
                if lab.ndim == 1 and idx == 0:
                    text += self._get_label_str(lab, sep=sep)
                else:
                    for i in range(idx, len(lab)):
                        text += self._get_label_str(lab[i], i, sep=sep)
        return text.rstrip()

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @property
    def data(self):
        """
        The `data` array (|ndarray|).

        If there is no data but labels, then the labels are returned instead of data.
        """
        if self._data is None and self.is_labeled:
            return self.get_labels(level=0)
        return self._data

    @data.setter
    def data(self, data):
        self._set_data(data)

    def get_labels(self, level=0):
        """
        Get the labels at a given level.

        Used to replace `data` when only labels are provided, and/or for
        labeling axis in plots.

        Parameters
        ----------
        level : int, optional, default:0
            For multilabel array, specify the label row to extract.

        Returns
        -------
        |ndarray|
            The label array at the desired level or None.
        """
        if not self.is_labeled:
            return None

        if (self.labels.ndim > 1 and level > len(self.labels) - 1) or (
            self.labels.ndim == 1 and level > 0
        ):
            warning_("There is no such level in the existing labels", LabelWarning)
            return None

        if len(self.labels) > 1 and self.labels.ndim > 1:
            return self.labels[level]
        return self.labels

    @property
    @_docstring.dedent
    def is_empty(self):
        """%(is_empty)s"""
        if self.is_labeled:
            return False
        return super().is_empty

    @property
    def is_labeled(self):
        """
        Return whether the `data` array have labels.
        """
        # label cannot exist for now for nD dataset - only 1D dataset, such
        # as Coord can be labeled.
        if self._data is not None and self._squeeze_ndim > 1:
            return False
        return self._labels is not None and np.any(self.labels != "")

    @property
    def labels(self):
        """
        An array of labels for `data` (|ndarray| of str).

        An array of objects of any type (but most generally string), with the last
        dimension size equal to that of the dimension of data. Note that's labeling
        is possible only for 1D data. One classical application is the labeling of
        coordinates to display informative strings instead of numerical values.
        """
        return self._labels

    @labels.setter
    def labels(self, labels):

        if labels is None:
            return
        if not self._is_labels_allowed():
            raise LabelsError("We cannot set the labels for multidimentional data.")
        # make sure labels array is of type np.ndarray or Quantity arrays
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, subok=True, copy=True).astype(object, copy=False)
        if (self.data is not None) and labels.ndim > 1:
            # allow the fact that the labels
            # may have been passed in a transposed array
            if labels.shape[-1] != self.size and labels.shape[0] == self.size:
                labels = labels.T
        if self.data is not None and labels.shape[-1] != self.size:
            raise LabelsError(
                f"labels {labels.shape} and data {self.shape} " f"shape mismatch!"
            )
        if np.any(self._labels):
            info_(
                f"{type(self).__name__} is already a labeled array.\n"
                f"The explicitly provided labels will "
                f"be appended to the current labels"
            )
            labels = labels.squeeze()
            self._labels = self._labels.squeeze()
            self._labels = np.vstack((self._labels, labels))
        else:
            if self._copy:
                self._labels = labels.copy()
            else:
                self._labels = labels

    @property
    @_docstring.dedent
    def ndim(self):
        """%(ndim)s"""
        if self.data is None and self.is_labeled:
            return 1
        return super().ndim

    @property
    @_docstring.dedent
    def shape(self):
        """%(shape.summary)s

        The number of `data` element on each dimension (possibly complex or
        hypercomplex).
        For only labeled array, there is no data, so it is the 1D and the size
        is the size of the array of labels.
        """
        if self.data is None and self.is_labeled:
            # noinspection PyRedundantParentheses
            return (self.labels.shape[0],)
        if self.data is None:
            return ()
        return self.data.shape

    @property
    def size(self):
        """
        Size of the underlying `data` array - Readonly property (int).

        The total number of data elements (possibly complex or hypercomplex
        in the array).
        """

        if self.data is None and self.is_labeled:
            return self.labels.shape[-1]
        if self.data is None:
            return 0
        return self.data.size

    @property
    @_docstring.dedent
    def values(self):
        """%(values)s"""
        if self.data is not None:
            return self.uarray.squeeze()[()]
        if self.is_labeled:
            if self.labels.size == 1:
                return np.asscalar(self.labels)
            if self.labels.ndim == 1:
                return self.labels
            return self.labels[0]
        return None


# ======================================================================================
if __name__ == "__main__":
    """"""
