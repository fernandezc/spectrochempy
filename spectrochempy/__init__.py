# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# Authors:
# christian.fernandez@ensicaen.fr
# arnaud.travert@ensicaen.fr
#
# This software is a computer program whose purpose is to provide a framework
# for processing, analysing and modelling *Spectro*scopic
# data for *Chem*istry with *Py*thon (SpectroChemPy). It is is a cross
# platform software, running on Linux, Windows or OS X.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================================
# flake8: noqa
"""
SpectroChemPy API.

SpectroChemPy is a framework for processing, analyzing and modeling Spectroscopic data
for Chemistry with Python.
It is a cross-platform software, running on Linux, Windows or OS X.
"""
import threading
import warnings

# setup warnings
# --------------
# warnings.filterwarnings(action="error", category=DeprecationWarning)
# warnings.filterwarnings(action="ignore", module="matplotlib")  # , category=UserWarning)
warnings.filterwarnings(
    action="once", module="spectrochempy", category=DeprecationWarning
)
warnings.filterwarnings(action="ignore", module="jupyter")  # , category=UserWarning)
warnings.filterwarnings(action="ignore", module="pykwalify")  # , category=UserWarning)


# --------------------------------------------------------------------------------------
# get API info
# --------------------------------------------------------------------------------------
from spectrochempy._api_info import api_info as info

name = info.name
icon = info.icon
description = info.description
version = info.version
release = info.release
release_date = info.release_date
copyright = info.copyright
url = info.url
authors = info.authors
contributors = info.contributors
license = info.license
cite = info.cite
long_description = info.long_description


# --------------------------------------------------------------------------------------
# Check for new release in a separate thread
# --------------------------------------------------------------------------------------
from spectrochempy._check_update import check_update

DISPLAY_UPDATE = threading.Thread(target=check_update, args=(version,))
# DISPLAY_UPDATE.start()
# do not leave trace of this method for the public API
del check_update

# --------------------------------------------------------------------------------------
# Set preferences (this also start the application
# --------------------------------------------------------------------------------------
from spectrochempy.application import preferences

# --------------------------------------------------------------------------------------
# Create the _api module if needed
# --------------------------------------------------------------------------------------
try:
    from spectrochempy._api import _api_methods
    from spectrochempy._dataset_methods import _dataset_methods
except (ImportError, ModuleNotFoundError):
    from spectrochempy._create_api import create_api

    _api_methods, _dataset_methods = create_api()

# # --------------------------------------------------------------------------------------
# # Lazily import objects when needed
# # --------------------------------------------------------------------------------------
# import lazy_loader
#
# subpackages = ["core" , "analysis", "processing", "widgets"]
#
# _getattr, _dir, _ = lazy_loader.attach(__name__, subpackages)

import traitlets as tr

def __getattr__(name):

    if name in _api_methods:
        return tr.import_item(_api_methods[name] + "." + name)
    else:
        # look also NDDataset attribute which can be used as API methods
        if name in _dataset_methods:
            from spectrochempy.core.dataset.nddataset import NDDataset

            return getattr(NDDataset, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    d = [
        "name",
        "icon",
        "description",
        "long_description",
        "version",
        "release",
        "release_date",
        "copyright",
        "url",
        "authors",
        "contributors",
        "license",
        "cite",
        "preferences",
    ]
    return d + _dir() # + list(_api_methods.keys()) + list(_dataset_methods.keys())
