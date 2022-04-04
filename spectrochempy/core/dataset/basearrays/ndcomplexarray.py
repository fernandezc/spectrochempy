# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
This module implements the NDArray derived object NDComplexArray
"""

import itertools
import numpy as np
import traitlets as tr
from quaternion import as_float_array, as_quat_array

from spectrochempy.core.common.complex import as_quaternion
from spectrochempy.core.common.constants import (
    typequaternion,
)
from spectrochempy.core.common.exceptions import (
    CastingError,
    ShapeError,
)
from spectrochempy.core.common.print import (
    numpyprintoptions,
)
from spectrochempy.core.dataset.basearrays.ndarray import NDArray, _docstring


# Printing settings
# --------------------------------------------------------------------------------------
numpyprintoptions()


# ======================================================================================
# NDArray subclass : NDComplexArray
# ======================================================================================
class NDComplexArray(NDArray):
    __doc__ = _docstring.dedent(
        """
    A |NDArray| derived class with complex/quaternion related functionalities.

    The private |NDComplexArray| class is an array (numpy |ndarray|-like) container,
    usually not intended to be used directly. In addition to the |NDArray|
    functionalities, this class adds complex/quaternion related methods and properties.

    Parameters
    ----------
    %(NDArray.parameters)s

    Other Parameters
    ----------------
    %(NDArray.other_parameters)s
    """
    )

    _interleaved = tr.Bool(False)

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):
        dtype = np.dtype(kwargs.get("dtype", None))
        if dtype.kind in "cV":
            kwargs["dtype"] = None  # The treatment will be done after the NDArray
            # initialisation
        super().__init__(data, accepted_kind="iufcVM", **kwargs)
        if dtype.kind == "c":
            self.set_complex(inplace=True)
        if dtype.kind == "V":  # quaternion
            self.set_hypercomplex(inplace=True)

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __getattr__(self, item):
        if item in "RRRIIIRR":
            return self.component(select=item)
        # return super().__getattr__(item)
        raise AttributeError

    def __setitem__(self, items, value):
        if self.is_hypercomplex and np.isscalar(value):
            # sometimes do not work directly : here is a work around
            keys = self._make_index(items)
            self._data[keys] = np.full_like(self._data[keys], value).astype(
                np.dtype(np.quaternion)
            )
        else:
            super().__setitem__(items, value)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _make_complex(self, data):
        if self.is_complex:  # pragma: no cover
            return data  # normally, this is never used as checks are done before
            # calling this function
        if data.shape[-1] % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array of real data to be transformed to complex must have "
                "an even number of columns!.",
            )
        data = data.astype(np.float64, copy=False)
        data.dtype = np.complex128
        return data

    def _make_quaternion(self, data, quat_array=False):
        if data.dtype == typequaternion:
            # nothing to do
            return data
        # as_quat_array
        if quat_array:
            if data.shape[-1] != 4:
                raise ShapeError(
                    data.shape,
                    "An array of data to be transformed to quaternion "
                    "with the option `quat_array` must have its last dimension "
                    "size equal to 4.",
                )
            data = as_quat_array(data)
            return data
        # interlaced data
        if data.ndim % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array of data to be transformed to quaternion must be 2D.",
            )
            # TODO: offer the possibility to have more dimension (interleaved)
        if not self.is_complex:
            if data.shape[1] % 2 != 0:
                raise ShapeError(
                    data.shape,
                    "an array of real data to be transformed to quaternion "
                    "must have even number of columns!.",
                )
            # convert to double precision complex
            data = self._make_complex(data)
        if data.shape[0] % 2 != 0:
            raise ShapeError(
                data.shape,
                "an array data to be transformed to quaternion must have"
                " even number of rows!.",
            )
        rea = data[::2]
        ima = data[1::2]
        return as_quaternion(rea, ima)

    def _cplx(self):
        cplx = [False] * self.ndim
        if self.is_hypercomplex:
            cplx = [True, True]
        elif self.is_complex:
            cplx[-1] = True
        return cplx

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _get_component(data, select="REAL"):
        """
        Take selected components of a hypercomplex array (RRR, RIR, ...).

        Parameters
        ----------
        data : ndarray
        select : str, optional, default='REAL'
            If 'REAL', only real component in all dimensions will be selected.
            Else a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        component
            A component of the complex or hypercomplex array.
        """
        if select in ["REAL", "R"]:
            select = "R" * data.ndim

        if data.dtype == typequaternion:
            w, x, y, z = as_float_array(data).T
            w, x, y, z = w.T, x.T, y.T, z.T
            if select == "I":
                as_float_array(data)[..., 0] = 0
                # see imag (we just remove the real
                # part
            elif select in ["RR", "R"]:
                data = w
            elif select == "RI":
                data = x
            elif select == "IR":
                data = y
            elif select == "II":
                data = z
            else:
                raise ValueError(
                    f"something wrong: cannot interpret `{select}` for "
                    f"hypercomplex (quaternion) data!"
                )

        elif data.dtype.kind in "c":
            w, x = data.real, data.imag
            if select in ["R", "RR", "RRR"]:
                data = w
            elif select in ["I", "RI", "RRI"]:
                data = x
            else:
                raise ValueError(
                    f"something wrong: cannot interpret `{select}` for complex "
                    f"data!"
                )
        else:
            raise ValueError(
                f"No selection was performed because datasets with no complex data "
                f"have no `{select}` component. "
            )

        return data

    def _str_shape(self):
        out = ""
        cplx = self._cplx()
        shcplx = (
            x
            for x in itertools.chain.from_iterable(
                list(zip(self.dims, self.shape, cplx))
            )
        )
        shape = (
            (", ".join(["{}:{}{}"] * self.ndim))
            .format(*shcplx)
            .replace("False", "")
            .replace("True", "(complex)")
        )
        size = self.size
        sizecplx = "" if not self.has_complex_dims else "(complex)"
        out += (
            f"         size: {size}{sizecplx}\n"
            if self.ndim < 2
            else f"        shape: ({shape})\n"
        )
        return out

    @property
    def has_complex_dims(self):
        """
        Return whether at least one of the `data` array dimension is complex.
        """
        if not self.has_data:
            return False
        return self.is_complex or self.is_hypercomplex

    @_docstring.dedent
    def astype(self, dtype=None, casting="safe", inplace=False):
        """%(astype)s"""
        if self.has_data and dtype is not None:
            new = self.copy() if not inplace else self
            dtype = np.dtype(dtype)
            try:
                if dtype.kind not in "cV":  # not complex nor quaternion
                    new._data = new.data.astype(dtype, casting=casting, copy=False)
                elif dtype.kind == "c":  # complex
                    new._data = self._make_complex(new.data)
                else:  # quaternion
                    new._data = self._make_quaternion(new.data)
            except TypeError as exc:
                raise CastingError(dtype, exc)
            return new
        return self

    @_docstring.get_docstring(base="component")
    @_docstring.dedent
    def component(self, select="REAL"):
        """
        Take selected components of a hypercomplex array.

        By default, the real component is returned. Using the select keyword any
        components (e.g. RRR, RIR, ...) of the hypercomplex array can be selected.

        Parameters
        ----------
        select : str, optional, default='REAL'
            If 'REAL', only real component in all dimensions will be selected,
            else a string must specify which real (R) or imaginary (I) component
            has to be selected along a specific dimension. For instance,
            a string such as 'RRI' for a 2D hypercomplex array indicated
            that we take the real component in each dimension except the last
            one, for which imaginary component is preferred.

        Returns
        -------
        %(new)s

        Notes
        -----
        The definition is somewhat different from e.g., Bruker TOPSPIN, as we order the
        component in the order of the dimensions in dataset:
        e.g., for dims = ['y','x'], 'IR' means that the `y` component is
        imaginary while the `x` is real.
        """
        new = self.copy()
        new._data = self._get_component(new.data, select)
        return new

    @property
    def is_complex(self):
        """
        Return whether the array is complex.
        """
        if not self.has_data:
            return False
        return self._data.dtype.kind == "c"

    @property
    def is_hypercomplex(self):
        """
        Return whether the array is hypercomplex.
        """
        if not self.has_data:
            return False
        return self._data.dtype.kind == "V"

    # #
    #
    #
    #
    #
    # TODO: should we keep this possibility of storing complex number?
    #       For the moment, no functions use this.
    #  ..................................................................................
    # @property
    # def is_interleaved(self):
    #     """
    #     Return whether the hypercomplex array has interleaved data.
    #     """
    #     if not self.has_data:
    #         return False
    #     return self._interleaved

    @property
    def imag(self):
        """
        Return the imaginary component of the complex array.
        """
        new = self.copy()
        if not new.has_complex_dims:
            return None
        data = new.data
        if self.is_complex:
            new._data = data.imag.data
        elif self.is_hypercomplex:
            # this is a more complex situation than for real component
            # get the imaginary component (vector component)
            # q = a + bi + cj + dk  ->   qi = bi+cj+dk
            as_float_array(data)[..., 0] = 0  # keep only the imaginary component
            new._data = data  # .data
        return new

    @property
    def limits(self):
        """
        Return the range of the data.
        """
        if not self.has_data:
            return None
        if self.is_complex:
            return [self._data.real.min(), self._data.real.max()]
        if self.is_hypercomplex:
            data = as_float_array(self._data)[..., 0]
            return [data.min(), data.max()]
        return [self._data.min(), self._data.max()]

    @property
    def real(self):
        """
        Return the real component of a complex array.
        """
        new = self.copy()
        if not new.has_complex_dims:
            return self
        data = new.data
        if self.is_complex:
            new._data = data.real
        elif self.is_hypercomplex:
            # get the scalar component
            # q = a + bi + cj + dk  ->   qr = a
            new._data = as_float_array(data)[..., 0]
        return new

    @_docstring.dedent
    def set_complex(self, inplace=False):
        """
        Set the object data as complex.

        When nD-dimensional array are set to complex, we assume that it is along the
        first dimension. Two succesives rows are merged to form a complex rows. This
        means that the number of row must be even. If the complexity is to be applied
        in other dimension, either transpose/swapdims your data before applying this
        function in order that the complex dimension is the first in the array.

        Parameters
        ----------
        %(inplace)s

        Returns
        -------
        %(out)s

        """
        new = self if inplace else self.copy()
        if new.has_complex_dims:
            # not necessary in this case, it is already complex
            return new
        new._data = new._make_complex(new.data)
        return new

    @_docstring.dedent
    def set_hypercomplex(self, quat_array=False, inplace=False):
        """
        Set the object data as hypercomplex.

        Four components are created : RR, RI, IR, II if they not already exit.

        Parameters
        ----------
        quat_array : bool, optional, default: False
            If True, this indicates that the four coponents of quaternion are given in
            an extra dimension: e.g., an array with shape (3,2,4) will be transfored to
            a quaternion array with shape (3,2) with each element of type quaternion.
            If False, the components are supposed to be interlaced: e.g, an array
            with shape (6,4) with float dtype or an array of shape (6,2) of complex
            dtype, will both be transformed to an array of shape (3,2) with quaternion
            dtype.
        %(inplace)s

        Returns
        -------
        %(out)s
        """
        new = self if inplace else self.copy()
        new._data = new._make_quaternion(new.data, quat_array=quat_array)
        return new

    @property
    @_docstring.dedent
    def shape(self):
        """%(shape.summary)s

        The number of `data` element on each dimension (possibly complex).
        """
        return super().shape


# ======================================================================================
if __name__ == "__main__":  # pragma: nocover
    """"""
