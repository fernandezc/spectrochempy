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
_dataset_methods = {
    "get_baseline": "spectrochempy.analysis.baseline.baseline",
    "basc": "spectrochempy.analysis.baseline.baseline",
    "detrend": "spectrochempy.analysis.baseline.baseline",
    "asls": "spectrochempy.analysis.baseline.baseline",
    "snip": "spectrochempy.analysis.baseline.baseline",
    "rubberband": "spectrochempy.analysis.baseline.baseline",
    "ab": "spectrochempy.analysis.baseline.baseline_deprecated",
    "abc": "spectrochempy.analysis.baseline.baseline_deprecated",
    "simps": "spectrochempy.analysis.integration.integrate",
    "trapz": "spectrochempy.analysis.integration.integrate",
    "simpson": "spectrochempy.analysis.integration.integrate",
    "trapezoid": "spectrochempy.analysis.integration.integrate",
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
    "load_iris": "spectrochempy.core.readers.download",
    "download_nist_ir": "spectrochempy.core.readers.download",
    "read": "spectrochempy.core.readers.importer",
    "read_dir": "spectrochempy.core.readers.importer",
    "write": "spectrochempy.core.writers.exporter",
    "em": "spectrochempy.processing.fft.apodization",
    "gm": "spectrochempy.processing.fft.apodization",
    "sp": "spectrochempy.processing.fft.apodization",
    "sine": "spectrochempy.processing.fft.apodization",
    "sinm": "spectrochempy.processing.fft.apodization",
    "qsin": "spectrochempy.processing.fft.apodization",
    "general_hamming": "spectrochempy.processing.fft.apodization",
    "hamming": "spectrochempy.processing.fft.apodization",
    "hann": "spectrochempy.processing.fft.apodization",
    "triang": "spectrochempy.processing.fft.apodization",
    "bartlett": "spectrochempy.processing.fft.apodization",
    "blackmanharris": "spectrochempy.processing.fft.apodization",
    "pk": "spectrochempy.processing.fft.phasing",
    "pk_exp": "spectrochempy.processing.fft.phasing",
    "rs": "spectrochempy.processing.fft.shift",
    "ls": "spectrochempy.processing.fft.shift",
    "roll": "spectrochempy.processing.fft.shift",
    "cs": "spectrochempy.processing.fft.shift",
    "fsh": "spectrochempy.processing.fft.shift",
    "fsh2": "spectrochempy.processing.fft.shift",
    "dc": "spectrochempy.processing.fft.shift",
    "zf_auto": "spectrochempy.processing.fft.zero_filling",
    "zf_double": "spectrochempy.processing.fft.zero_filling",
    "zf_size": "spectrochempy.processing.fft.zero_filling",
    "zf": "spectrochempy.processing.fft.zero_filling",
    "savgol_filter": "spectrochempy.processing.filter.filter",
    "savgol": "spectrochempy.processing.filter.filter",
    "smooth": "spectrochempy.processing.filter.filter",
    "whittaker": "spectrochempy.processing.filter.filter",
}
