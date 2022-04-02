# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Various methods and classes used in other part of the program.
"""

import numpy as np


def _get_n_decimals(val, accuracy):
    if abs(val) > 0.0:
        nd = int(np.log10(abs(val) * accuracy))
        nd = 1 if nd >= 0 else -nd + 1
        return nd
    return 3


def spacings(arr, decimals=3):
    """
    Return a scalar for the spacing in the one-dimensional input array
    (if it is uniformly spaced, else return an array of the different spacings.

    Parameters
    ----------
    arr : array-like
        An input 1D array
    decimals : Int, optional, default=3
        The number of rounding decimals to determine spacings

    Returns
    -------
    float or array
    """
    arr = np.asarray(arr)

    spacings = np.diff(arr)

    # we need to take into account only the significative digits
    ndecimals = _get_n_decimals(spacings.max(), 10 ** -(decimals + 1))
    spacings = list(set(np.around(spacings, ndecimals)))

    return spacings[0] if len(spacings) == 1 else spacings
