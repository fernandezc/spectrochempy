# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Loading an IR (omnic SPG) experimental file
===========================================

Here we load an experimental SPG file (OMNIC) and plot it.

"""

# %%
import spectrochempy as scp

# %%
# Loading and stacked plot of the original

datadir = scp.preferences.datadir
dataset = scp.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
dataset.plot_stack(style="paper")

# %%
# change the unit of y-axis, the y origin as well as the title of the axis

dataset.y.to("hour")
dataset.y -= dataset.y[0]
dataset.y.title = "acquisition time"

dataset.plot_stack()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
