# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
This module implements the NDArray derived object NDMaskedComplexArray
"""

import numpy as np
import traitlets as tr
from traittypes import Array

from spectrochempy.core import info_
from spectrochempy.core.common.constants import (
    MASKED,
    NOMASK,
    MaskedConstant,
    typequaternion,
)
from spectrochempy.core.common.print import insert_masked_print, numpyprintoptions
from spectrochempy.core.dataset.basearrays.ndarray import _docstring
from spectrochempy.core.dataset.basearrays.ndcomplexarray import NDComplexArray

# Printing settings
# --------------------------------------------------------------------------------------
numpyprintoptions()


# ======================================================================================
# NDArray subclass : NDMaskedComplexArray
# ======================================================================================
class NDMaskedComplexArray(NDComplexArray):
    __doc__ = _docstring.dedent(
        """
    A |NDComplexArray| derived class with additional mask and related functionalities.

    The private |NDMaskedComplexArray| class is an array (numpy |ndarray|-like)
    container, usually not intended to be used directly. In addition to the
    |NDComplexArray| functionalities, this class adds a mask and related methods and
    properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    mask : array of bool or `NOMASK`, optional
        Mask for the data. The mask array must have the same shape as the
        data. The given array can be a list,
        a tuple, or a |ndarray|. Each values in the array must be `False`
        where the data are *valid* and True when
        they are not (like in numpy masked arrays). If `data` is already a
        :class:`~numpy.ma.MaskedArray`, or any object providing a `mask`, the mask
        defined by this parameter and the mask from the mask from the data will be
        combined (`mask` OR `data.mask`).
    """
    )
    _docstring.get_sections(__doc__, base="NDMaskedComplexArray")

    # masks
    _mask = tr.Union(
        (tr.Bool(), Array(tr.Bool(), allow_none=True), tr.Instance(MaskedConstant))
    )

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        mask = kwargs.pop("mask", NOMASK)
        if np.any(mask):
            self.mask = mask

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __getitem__(self, items, return_index=False):
        new, keys = super().__getitem__(items, return_index=True)
        if (new.data is not None) and new.is_masked:
            new._mask = new._mask[keys]
        else:
            new._mask = NOMASK
        if not return_index:
            return new
        return new, keys

    def __setitem__(self, items, value):
        if isinstance(value, (bool, np.bool_, MaskedConstant)):
            # in this case we set the mask not the data!
            keys = self._make_index(items)
            # the mask is modified, not the data
            if value is MASKED:
                value = True
            if not np.any(self._mask):
                self._mask = np.zeros(self._data.shape).astype(np.bool_)
            self._mask[keys] = value
            return
        # set data item case
        super().__setitem__(items, value)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _attributes(self, removed=[]):
        return super()._attributes(removed) + ["mask"]

    @tr.default("_mask")
    def __mask_default(self):
        return NOMASK if not self.has_data else np.zeros(self._data.shape).astype(bool)

    @tr.validate("_mask")
    def _mask_validate(self, proposal):
        pv = proposal["value"]
        mask = pv
        if mask is None or mask is NOMASK:
            return mask
        # no particular validation for now.
        if self._copy:
            return mask.copy()
        return mask

    def _set_data(self, data):
        mask = NOMASK
        if hasattr(data, "mask"):
            # an object with data and mask attributes
            if isinstance(data.mask, np.ndarray) and data.mask.shape == data.data.shape:
                mask = np.array(data.mask, dtype=np.bool_, copy=False)
        super()._set_data(data)
        if np.any(mask):
            self.mask = mask

    def _str_formatted_array(self, data, sep="\n", prefix="", units=""):
        ds = data.copy()
        if self.is_masked:
            dtype = self._data.dtype
            mask_string = f"--{dtype}"
            ds = insert_masked_print(ds, mask_string=mask_string)
        arraystr = np.array2string(ds, separator=" ", prefix=prefix)
        arraystr = arraystr.replace("\n", sep)
        text = "".join([arraystr, units])
        text += sep
        return text

    @staticmethod
    def _masked_data(data, mask):
        # This ensures that a masked array is returned.
        if not np.any(mask):
            mask = np.zeros(data.shape).astype(bool)
        data = np.ma.masked_where(mask, data)  # np.ma.masked_array(data, mask)
        return data

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def component(self, select="REAL"):
        """%(component)s"""
        new = super().component(select)
        if new is not None and self.is_masked:
            new._mask = self.mask
        else:
            new._mask = NOMASK
        return new

    @property
    def is_masked(self):
        """
        Whether the `data` array has masked values.
        """
        try:
            if np.all(self._mask == NOMASK) or self._mask is None:
                return False
            if isinstance(self._mask, (np.bool_, bool)):
                return self._mask
            if isinstance(self._mask, np.ndarray):
                return np.any(self._mask)
        except Exception:
            if self._data.dtype == typequaternion:
                return np.any(self._mask["R"])
        return False

    @property
    def mask(self):
        """
        Mask for the data (|ndarray| of bool).
        """
        if not self.is_masked:
            return NOMASK
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is NOMASK or mask is MASKED:
            pass
        elif isinstance(mask, (np.bool_, bool)):
            if not mask:
                mask = NOMASK
            else:
                mask = MASKED
        else:
            # from now, make sure mask is of type np.ndarray if it provided
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask, dtype=np.bool_)
            if not np.any(mask):
                # all element of the mask are false
                mask = NOMASK
            elif mask.shape != self.shape:
                raise ValueError(
                    f"mask {mask.shape} and data {self.shape} shape mismatch!"
                )
        # finally, set the mask of the object
        if isinstance(mask, MaskedConstant):
            self._mask = (
                NOMASK if not self.has_data else np.ones(self.shape).astype(bool)
            )
        else:
            if np.any(self._mask):
                # this should happen when a new mask is added to an existing one
                # mask to be combined to an existing one
                info_(
                    f"{type(self).__name__} is already a masked array.\n"
                    f"The new mask will be combined with the current array's mask."
                )
                self._mask |= mask  # combine (is a copy!)
            else:
                if self._copy:
                    self._mask = mask.copy()
                else:
                    self._mask = mask

    @property
    def masked_data(self):
        """
        The actual masked `data` array .
        """
        if not self.is_empty:
            return self._masked_data(self._data, self.mask)
        return self._data

    @property
    def real(self):
        """
        Return the real component of complex array (Readonly property).
        """
        new = super().real
        if new is not None and self.is_masked:
            new._mask = self.mask
        return new

    @property
    def imag(self):
        """
        Return the imaginary component of complex array (Readonly property).
        """
        new = super().imag
        if new is not None and self.is_masked:
            new._mask = self.mask
        return new

    def remove_masks(self):
        """
        Remove data masks.

        Remove all masks previously set on this array, unconditionnaly.
        """
        self._mask = NOMASK

    @property
    @_docstring.dedent
    def uarray(self):
        """%(uarray)s"""
        if self._data is not None:
            return self._uarray(self.masked_data, self.units)
        return None


# ======================================================================================
if __name__ == "__main__":
    """"""
