# ======================================================================================
# Copyright (©) 2015-2025 LCS
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
"""
SpectroChemPy API Module.

SpectroChemPy is a framework for processing, analyzing and modeling Spectroscopic data
for Chemistry with Python. It is a cross-platform software, running on Linux, Windows or OS X.

This module serves as the main entry point for the SpectroChemPy package, providing:
- Configuration of warning filters
- Import of the public API
- Dynamic attribute access to NDDataset methods
"""

import lazy_loader as _lazy_loader

# --------------------------------------------------------------------------------------
# Lazy loading of sub-packages
# --------------------------------------------------------------------------------------
# Store the original __getattr__ from lazy_loader
original_getattr, *_ = _lazy_loader.attach_stub(__name__, __file__)

# Dictionary mapping top-level objects to their module paths
import threading
from functools import wraps
from queue import Queue

from spectrochempy import application
from spectrochempy.lazyimport.api_methods import _LAZY_IMPORTS
from spectrochempy.lazyimport.dataset_methods import _LAZY_DATASETS_IMPORTS

# Create synchronization objects
_preferences = None
_preferences_lock = threading.Lock()
_preferences_queue = Queue()

# --------------------------------------------------------------------------------------
# Display a loading message
# --------------------------------------------------------------------------------------
application.start.display_loading_message(3)

# --------------------------------------------------------------------------------------
# Warning configurations
# --------------------------------------------------------------------------------------
application.start.set_warnings()

# # --------------------------------------------------------------------------------------
# # Getting preferences
# # --------------------------------------------------------------------------------------
# preferences = application.preferences.preferences

# # --------------------------------------------------------------------------------------
# # Check for new release in a separate thread
# # --------------------------------------------------------------------------------------
# import threading

# check_update = application.check_update.check_update
# version = application.info.version

# check_update_frequency = preferences.check_update_frequency
# DISPLAY_UPDATE = threading.Thread(
#     target=check_update, args=(version, check_update_frequency)
# )
# if not application.application.NO_DISPLAY:
#     DISPLAY_UPDATE.start()

# # --------------------------------------------------------------------------------------
# # Download data in a separate thread
# # --------------------------------------------------------------------------------------
# download_full_testdata_directory = application.testdata.download_full_testdata_directory

# DOWNLOAD_TESTDATA = threading.Thread(
#     target=download_full_testdata_directory,
#     args=(preferences.datadir,),
# )
# DOWNLOAD_TESTDATA.start()

# --------------------------------------------------------------------------------------
# Plugin manager
# --------------------------------------------------------------------------------------
# from .plugins.pluginmanager import PluginManager

# plugin_manager = PluginManager()
# plugin_manager.discover_plugins()

# # initialize all auto-initializable plugins
# for plugin in plugin_manager.available_plugins.values():
#     if plugin.auto_initialize:
#         plugin.initialize(manager=plugin_manager)

# __all__.append("plugin_manager")


def get_preferences():
    """Lazy load preferences only when needed."""
    global _preferences
    with _preferences_lock:
        if _preferences is None:
            from . import application

            _preferences = application.preferences.preferences
            _preferences_queue.put(_preferences)
        return _preferences


def requires_preferences(func):
    """
    Ensure preferences are loaded before function runs.

    This decorator guarantees that preferences are initialized before the decorated
    function is executed.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        prefs = get_preferences()
        return func(prefs, *args, **kwargs)

    return wrapper


@requires_preferences
def check_update_thread(preferences):
    """Check for updates in a separate thread."""
    from . import application

    check_update = application.check_update.check_update
    version = application.info.version

    if not application.application.NO_DISPLAY:
        check_update(version, preferences.check_update_frequency)


@requires_preferences
def download_testdata_thread(preferences):
    """Download testdata in a separate thread."""
    from . import application

    download_full_testdata_directory = (
        application.testdata.download_full_testdata_directory
    )
    download_full_testdata_directory(preferences.datadir)


# Start threads
DISPLAY_UPDATE = threading.Thread(target=check_update_thread)
DOWNLOAD_TESTDATA = threading.Thread(target=download_testdata_thread)

DISPLAY_UPDATE.start()
DOWNLOAD_TESTDATA.start()

# ------------------------------------------------------------------------------
# Display welcome message
# ------------------------------------------------------------------------------
import sys

version = application.info.version
copyright = application.info.copyright
welcome_string = f"SpectroChemPy's API - v.{version}\n©Copyright {copyright}"

from .utils.system import is_notebook

if is_notebook():  # pragma: no cover
    # Only in Jupyter notebook.
    application.info.display_info_string(message=welcome_string.strip())
else:
    if "/bin/" not in sys.argv[0]:  # deactivate for console scripts
        print(welcome_string.strip())  # noqa: T201


# Override __getattr__ to handle both submodules and direct class access
def __getattr__(name):
    """
    Lazily import modules or classes when accessed.

    This function enables direct access to classes like `scp.Coord`
    without importing them until they are actually used.
    """
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[name])
        return getattr(module, name)

    # Look also NDDataset attribute which can be used as API methods
    if name in _LAZY_DATASETS_IMPORTS:
        from spectrochempy.core.dataset.nddataset import NDDataset

        return getattr(NDDataset, name)

    # Fall back to original __getattr__ for submodules
    try:
        return original_getattr(name)
    except AttributeError as err:
        raise AttributeError(
            f"module 'spectrochempy' has no attribute '{name}'"
        ) from err


# we don't use __all__ and __dir__ returned _lazy_loader.attach_stub

__all__ = list(_LAZY_IMPORTS.keys())


def __dir__() -> list[str]:
    # displays the list of available attributes in the top-level package
    return __all__


# ------------------------------------------------------------------------------

if __name__ == "main":
    pass
