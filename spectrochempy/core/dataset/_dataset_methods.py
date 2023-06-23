# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
#
#
#    ###################################################################################
#    #           DO NOT MODIFY THIS FILE BECAUSE IT IS CREATED AUTOMATICALLY.          #
#    #   ANY MODIFICATION OF THIS FILE WILL BE CANCELLED AFTER THE COMMIT IN GITHUB.   #
#    ###################################################################################
#
#
# flake8: noqa
"""
NDDataset methods from API.
"""

# --------------------------------------------------------------------------------------
# API methods that are also NDDataset methods
# --------------------------------------------------------------------------------------
dataset_methods = {
    "plot_with_transposed": "spectrochempy.core.plotters.multiplot",
    "plot_1D": "spectrochempy.core.plotters.plot1d",
    "plot_pen": "spectrochempy.core.plotters.plot1d",
    "plot_scatter": "spectrochempy.core.plotters.plot1d",
    "plot_bar": "spectrochempy.core.plotters.plot1d",
    "plot_multiple": "spectrochempy.core.plotters.plot1d",
    "plot_scatter_pen": "spectrochempy.core.plotters.plot1d",
    "plot_2D": "spectrochempy.core.plotters.plot2d",
    "plot_map": "spectrochempy.core.plotters.plot2d",
    "plot_stack": "spectrochempy.core.plotters.plot2d",
    "plot_image": "spectrochempy.core.plotters.plot2d",
    "plot_3D": "spectrochempy.core.plotters.plot3d",
    "plot_surface": "spectrochempy.core.plotters.plot3d",
    "plot_waterfall": "spectrochempy.core.plotters.plot3d",
    "em": "spectrochempy.core.processors.apodization",
    "gm": "spectrochempy.core.processors.apodization",
    "sp": "spectrochempy.core.processors.apodization",
    "sine": "spectrochempy.core.processors.apodization",
    "sinm": "spectrochempy.core.processors.apodization",
    "qsin": "spectrochempy.core.processors.apodization",
    "general_hamming": "spectrochempy.core.processors.apodization",
    "hamming": "spectrochempy.core.processors.apodization",
    "hann": "spectrochempy.core.processors.apodization",
    "triang": "spectrochempy.core.processors.apodization",
    "bartlett": "spectrochempy.core.processors.apodization",
    "blackmanharris": "spectrochempy.core.processors.apodization",
    "autosub": "spectrochempy.core.processors.autosub",
    "BaselineCorrection": "spectrochempy.core.processors.baseline",
    "ab": "spectrochempy.core.processors.baseline",
    "abc": "spectrochempy.core.processors.baseline",
    "dc": "spectrochempy.core.processors.baseline",
    "basc": "spectrochempy.core.processors.baseline",
    "fft": "spectrochempy.core.processors.fft",
    "ifft": "spectrochempy.core.processors.fft",
    "mc": "spectrochempy.core.processors.fft",
    "ps": "spectrochempy.core.processors.fft",
    "ht": "spectrochempy.core.processors.fft",
    "savgol_filter": "spectrochempy.core.processors.filter",
    "detrend": "spectrochempy.core.processors.filter",
    "smooth": "spectrochempy.core.processors.filter",
    "simps": "spectrochempy.core.processors.integrate",
    "trapz": "spectrochempy.core.processors.integrate",
    "simpson": "spectrochempy.core.processors.integrate",
    "trapezoid": "spectrochempy.core.processors.integrate",
    "interpolate": "spectrochempy.core.processors.interpolate",
    "pk": "spectrochempy.core.processors.phasing",
    "pk_exp": "spectrochempy.core.processors.phasing",
    "zf_auto": "spectrochempy.core.processors.zero_filling",
    "zf_double": "spectrochempy.core.processors.zero_filling",
    "zf_size": "spectrochempy.core.processors.zero_filling",
    "zf": "spectrochempy.core.processors.zero_filling",
    "download_iris": "spectrochempy.core.readers.download",
    "download_nist_ir": "spectrochempy.core.readers.download",
    "read": "spectrochempy.core.readers.importer",
    "read_dir": "spectrochempy.core.readers.importer",
    "read_remote": "spectrochempy.core.readers.importer",
    "read_carroucell": "spectrochempy.core.readers.read_carroucell",
    "read_csv": "spectrochempy.core.readers.read_csv",
    "read_jcamp": "spectrochempy.core.readers.read_jcamp",
    "read_jdx": "spectrochempy.core.readers.read_jcamp",
    "read_dx": "spectrochempy.core.readers.read_jcamp",
    "read_labspec": "spectrochempy.core.readers.read_labspec",
    "read_matlab": "spectrochempy.core.readers.read_matlab",
    "read_mat": "spectrochempy.core.readers.read_matlab",
    "read_omnic": "spectrochempy.core.readers.read_omnic",
    "read_spg": "spectrochempy.core.readers.read_omnic",
    "read_spa": "spectrochempy.core.readers.read_omnic",
    "read_srs": "spectrochempy.core.readers.read_omnic",
    "read_opus": "spectrochempy.core.readers.read_opus",
    "read_quadera": "spectrochempy.core.readers.read_quadera",
    "read_soc": "spectrochempy.core.readers.read_soc",
    "read_ddr": "spectrochempy.core.readers.read_soc",
    "read_sdr": "spectrochempy.core.readers.read_soc",
    "read_hdr": "spectrochempy.core.readers.read_soc",
    "read_spc": "spectrochempy.core.readers.read_spc",
    "read_topspin": "spectrochempy.core.readers.read_topspin",
    "read_bruker_nmr": "spectrochempy.core.readers.read_topspin",
    "read_zip": "spectrochempy.core.readers.read_zip",
    "write": "spectrochempy.core.writers.exporter",
    "write_csv": "spectrochempy.core.writers.write_csv",
    "write_excel": "spectrochempy.core.writers.write_excel",
    "write_xls": "spectrochempy.core.writers.write_excel",
    "write_jcamp": "spectrochempy.core.writers.write_jcamp",
    "write_jdx": "spectrochempy.core.writers.write_jcamp",
    "write_matlab": "spectrochempy.core.writers.write_matlab",
    "write_mat": "spectrochempy.core.writers.write_matlab",
}
