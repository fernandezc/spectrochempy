# Write a plugin for spectrochempy
# =================================
#
# This guide will show you how to write a plugin for spectrochempy.
#

# %%
import spectrochempy as scp

# %%
x = scp.read(["irdata/irdata.spg", "x.spa", "y.spa"])  # sould be ok
y = scp.read_omnic("irdata/irdata.spg")  # ok
y = scp.read_omnic("irdata/irdata.srs")  # ok
z = scp.read_spg("irdata/irdata.spg")  # ok
u = scp.read_spa("irdata/xxxx.spa")  # ok
v = scp.read_spa("irdata/irdata.spg")  # this will fail
pass

# %%
