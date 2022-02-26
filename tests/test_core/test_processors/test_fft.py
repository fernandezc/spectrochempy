# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa

from spectrochempy.core.units import ur


def test_nmr_fft_1D(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    dataset1D.x.ito("s")
    new = dataset1D.fft(tdeff=8192, size=2 ** 15)
    new2 = new.ifft()


def test_nmr_fft_1D_our_Hz(NMR_dataset_1D):
    dataset1D = NMR_dataset_1D.copy()
    dataset1D /= dataset1D.real.data.max()  # normalize
    LB = 10.0 * ur.Hz
    GB = 50.0 * ur.Hz
    dataset1D.gm(gb=GB, lb=LB)
    new = dataset1D.fft(size=32000, ppm=False)
