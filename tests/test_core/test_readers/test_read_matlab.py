# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

import pytest
import spectrochempy as scp
from spectrochempy import preferences as prefs

MATLABDATA = prefs.datadir / "matlabdata"


# @pytest.mark.skipif(
#     not MATLABDATA.exists(),
#     reason="Experimental data not available for testing",
# )
def test_read_matlab():

    A = scp.read_matlab(MATLABDATA / "als2004dataset.MAT")
    assert len(A) == 6
    assert A[3].shape == (4, 96)

    A = scp.read_matlab(MATLABDATA / "dso.mat")
    assert A.name == "Group sust_base line withoutEQU.SPG"
    assert A.shape == (20, 426)
