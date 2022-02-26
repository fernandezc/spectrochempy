# %%
"""
Load and save NDDataset
=======================

To import data from different programs, there are several `readers` that can be used, such as `read_omnic`, 'read_opus`, ...
In this example we show how to load and save data in the proprietary SpectroChempy format (extension *.scp). 

"""

# %%
import spectrochempy as scp

# %% [markdown]
# Let's start by importing data from Omnic

# %%
datadir = scp.preferences.datadir
dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
dataset

# %%
# Display content:

dataset

scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)

# %%
