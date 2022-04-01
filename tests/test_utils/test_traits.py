# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa


import pytest
import traitlets as tr

from spectrochempy.utils.traits import Range


# ======================================================================================================================
# Range
# ======================================================================================================================
def test_range():
    class MyClass(tr.HasTraits):
        r = Range()  # Initialized with some default values

    c = MyClass()
    c.r = [10, 5]
    assert c.r == [5, 10]
    with pytest.raises(tr.TraitError):
        c.r = [10, 5, 1]
