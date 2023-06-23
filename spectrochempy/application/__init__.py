# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
# --------------------------------------------------------------------------------------
# Setup the environment.
# --------------------------------------------------------------------------------------

from spectrochempy.application.envsetup import setup_environment

NO_DISPLAY, SCPY_STARTUP_LOGLEVEL = setup_environment()
__all__ = ["NO_DISPLAY"]

# --------------------------------------------------------------------------------------
# Define an instance of the SpectroChemPy application.
# --------------------------------------------------------------------------------------
from spectrochempy.application.application import SpectroChemPy

app = SpectroChemPy(log_level=SCPY_STARTUP_LOGLEVEL)
info_ = app.info_
debug_ = app.debug_
warning_ = app.warning_
error_ = app.error_
config_manager = app.config_manager
config_dir = app.config_dir
log_dir = app.log_dir
save_dialog = app.save_dialog
open_dialog = app.open_dialog
__all__ += [
    "app",
    "info_",
    "debug_",
    "warning_",
    "error_",
    "save_dialog",
    "open_dialog",
]

# --------------------------------------------------------------------------------------
# Start the application: this fire the generation of configurable options.
# --------------------------------------------------------------------------------------
app.start()
from spectrochempy.application.preferences_set import PreferencesSet

preferences = PreferencesSet()
__all__ += ["preferences"]

# --------------------------------------------------------------------------------------
# Add some utility functions and variables.
# --------------------------------------------------------------------------------------
def set_loglevel(level="WARNING"):
    if isinstance(level, str):
        import logging

        level = getattr(logging, level)
    preferences.log_level = level


def get_loglevel():
    return preferences.log_level


DATADIR = preferences.datadir

__all__ += ["set_loglevel", "get_loglevel", "DATADIR"]

# Write configurable defaults
from spectrochempy.application.general_preferences import GeneralPreferences
from spectrochempy.application.plot_preferences import PlotPreferences

configurables = [GeneralPreferences, PlotPreferences]
app.make_default_config_file(configurables)
# TODO: configurable from analysis to add


# --------------------------------------------------------------------------------------
# Setup for pytest and sphinx
# --------------------------------------------------------------------------------------
# import warnings

# warnings.filterwarnings(action="ignore", module="matplotlib")  # , category=UserWarning)
# warnings.filterwarnings(action="error", category=DeprecationWarning)

if NO_DISPLAY:
    from os import environ

    import matplotlib as mpl

    mpl.use("template", force=True)

    # set test file and folder in environment
    # set a test file in environment
    environ["TEST_FILE"] = str(DATADIR / "irdata" / "nh4y-activation.spg")
    environ["TEST_FOLDER"] = str(DATADIR / "irdata" / "subdir")
    environ["TEST_NMR_FOLDER"] = str(
        DATADIR / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
    )
