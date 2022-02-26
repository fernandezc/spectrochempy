# -*- coding: utf-8 -*-
# %%
"""
Import / Export to netCDF and xarray
====================================
In this example, we show how to import/export netCDF files.

"""
# %%
# `NetCDF`_ (Network Common Data Form)
# is a file format designed to support the creation,
# access and sharing of scientific data. It is widely used among the oceanographic and
# atmospheric communities to store variables, such as temperature, pressure, wind speed,
# and wave height.
#
# Less known and used in the chemistry community,
# it is not lacking of interest to store data and structure of NDDataset
# in Spectrochempy.
#
# `NetCDF`_ data are :
#
# * self-describing, meaning that a netCDF file includes information about the data it
#   contains, such as when the data elements were captured and what units of
#   measurement were used;
# * portable, or cross-platform, in the sense that a netCDF file created on one type
#   of operating system can often be read by software on another type of operating
#   system;
# * scalable, in the sense that it is possible to efficiently access a small part
#   of a large netCDF file without having to read the whole file.
#
# In SpectroChemPy we do not use yet all the features, but saving and reading such
# data is possible. This for instance makes easy the exchange of data between
# `xarray`_ and SpectroChemPy.
#
# .. _xarray: https://xarray.pydata.org/en/stable/getting-started-guide/why-xarray.html
# .. _NetCDF: http://www.unidata.ucar.edu/software/netcdf

# %%
# .. warning:
#
#   This example works only if the xarray package is installed.
#   If it is not the case, do it using conda, or pip.

# %%
# Let's start this example by reading some Infrared spectroscopic data.
import spectrochempy as scp

datadir = scp.preferences.datadir
nd = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
_ = nd.plot_map()
nd

# %%
# Write NetCDF file
# -----------------
# To export a dataset to the netCDF format, simply use the `write_netcdf` method.
# Pass a filename where to save (generally with the extension *.nc and with the
# confirm=False argument in order to avoid opening a dialog if the file already exists.
# The function return the full path of the created file.
f = nd.write_netcdf("netcdf_example.nc", confirm=False)

# %%
# Read NetCDF file (created by SpectrochemPy)
# -------------------------------------------
# To read the file, and create a new dataset from it:
othernd = scp.read_netcdf(f)
othernd

# %%
# we can check that the newly created dataset from the netCDF file is equivalent to
# the original one
assert othernd == nd

# %%
# Read NetCDF file (created by SpectroChemPy) in xarray
# -----------------------------------------------------
# Now we can test the opening of SpectroChemPy netCDF files by xarray
import xarray as xr

xrd = xr.open_dataarray(f)

# %%
# xrd is a
# `xarray.DataArray <https://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray>`_
# object.
xrd

# %%
# It can be used as usually by xarray.
_ = xrd.plot()

# %%
xrd["y"] = (xrd.y - xrd.y[0]).astype(float) / 1.0e9  # base unit being in nanosecond
# xarray use as in spectrochempy the long_name to label the axes automatically.
xrd.y.attrs["long_name"] = "time / s"
xrd.x.attrs["long_name"] = "wavenumber / cm^-1"
_ = xrd.plot()

# %%
# Read xarray NetCDF file (created by xarray) in SpectroChemPy
# ------------------------------------------------------------
# For this, we will use a dataset present in the `xarray tutorial`_.
# .. _`xarray tutorial`: https://xarray.pydata.org/en/stable/user-guide/plotting.html#imports
#
# Open a xarray dataset: here we follow the xarray tutorial and use
# use the North American air temperature dataset.
airtemps = xr.tutorial.open_dataset("air_temperature")
airtemps

# %%
# Convert to celsius
air = airtemps.air - 273.15

# %%
# Copy attributes to get nice figure labels and change Kelvin to Celsius
air.attrs = airtemps.air.attrs
air.attrs["units"] = "deg C"

# %%
# Let extract one xarray.DataArray from this dataset (a section a longitude 10)
air2d = air.isel(lon=10)
air2d

# %%
# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)

# %%
# The xarray plot is:
_ = air2d.plot()

# %%
# Let's save it into a NetCDF format
air2d.to_netcdf("xarray_example.nc")

# %%
# and then, try to read it in SpectroChemPy
nd = scp.read_netcdf("xarray_example.nc")
nd

# %%
