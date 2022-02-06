# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

"""
Package containing various utilities classes and functions.
"""
# some useful constants
# ------------------------------------------------------------------
# import numpy as np

# masked arrays
# ------------------------------------------------------------------
# noinspection PyUnresolvedReferences
from numpy.ma.core import (
    masked as MASKED,
    nomask as NOMASK,
    MaskedArray,
    MaskedConstant,
)  # noqa: F401

# import util files content
# ------------------------------------------------------------------
# noinspection PyUnresolvedReferences
from .print import *
from .fake import *
from .file import *
from .jsonutils import *
from .misc import *
from .packages import *
from .plots import *
from .system import *
from .traits import *
from .zip import *
from .exceptions import *
from .version import *
from .print_versions import *
from .datetimeutils import *
