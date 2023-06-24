# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

subpackages = [
    "dataset",
    "project",
    "units",
    "plotters",
    "readers",
    "writers",
    "script",
]

import lazy_loader as lazy

__getattr__, __dir__, _ = lazy.attach(__name__, subpackages)
