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
    else:
        return 3
    if nd >= 0:
        nd = 1
    else:
        nd = -nd + 1
    return nd


def spacings(arr):
    """
    Return a scalar for the spacing in the one-dimensional input array (if it is uniformly spaced,
    else return an array of the different spacings.

    Parameters
    ----------
    arr : 1D np.array

    Returns
    -------
    out : float or array
    """
    spacings = np.diff(arr)
    # we need to take into account only the significative digits
    # ( but round to some decimals doesn't work
    # for very small number
    #    mantissa, twoexp = np.frexp(spacings)
    #    mantissa = mantissa.round(6)
    #    spacings = np.ldexp(mantissa, twoexp)
    #    spacings = list(set(abs(spacings)))
    nd = _get_n_decimals(spacings.max(), 1.0e-3)
    spacings = list(set(np.around(spacings, nd)))

    if len(spacings) == 1:
        # uniform spacing
        return spacings[0]
    else:
        return spacings
