# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np
import quaternion  # noqa: F401

hypercomplex = np.dtype(np.quaternion)
typequaternion = hypercomplex

# quaternion is a non-public alias of hypercomplex

TYPE_INTEGER = (int, np.int_, np.int32, np.int64, np.uint32, np.uint64)
TYPE_FLOAT = (float, np.float_, np.float32, np.float64)
TYPE_COMPLEX = (complex, np.complex_, np.complex64, np.complex128)
TYPE_BOOL = (bool, np.bool_)
