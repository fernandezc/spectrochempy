# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["MASKED", "NOMASK", "DEFAULT_DIM_NAME"]

import numpy as np
from numpy.ma import core

#: Used to mask a value in a NDDataset, e.g. nd[1]=MASKED
MASKED = core.masked

#: Used to unmask a value in a NDDataset, e.g. nd[1]=NOMASK
NOMASK = core.nomask

#: List of the default name for the dimensions: x, y, ...
DEFAULT_DIM_NAME = list("xyzuvwpqrstijklmnoabcdefgh")[::-1]

#: Smallest value for float numbers
EPSILON = epsilon = np.finfo(float).eps

#: TODO make a comment
INPLACE = "INPLACE"

# private
MaskedConstant = core.MaskedConstant
MaskedArray = core.MaskedArray
