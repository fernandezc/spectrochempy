# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = []


import numpy as np

from spectrochempy.utils import NGreen, NRed, NBlue, NBlack

# ............................................................................
def make_label(ss, lab='<no_axe_label>'):
    """ make a label from title and units

    """
    if ss.title:
        label = ss.title  # .replace(' ', r'\ ')
    else:
        label = lab

    if ss.units is not None and str(ss.units) != 'dimensionless':
        #if str(ss.units) == 'absorbance':
        #    units = r'/\ a.u.'
        #else:
        units = r"/\ {:~L}".format(ss.units)
        units = units.replace('%',r'\%')
    else:
        units = ''

    label = r"%s $\mathrm{%s}$" % (label, units)
    return label


def make_attr(key):
    name = 'M_%s' % key[1]
    k = r'$\mathrm{%s}$' % name

    if 'P' in name:
        m = 'o'
        c = NBlack
    elif 'A' in name:
        m = '^'
        c = NBlue
    elif 'B' in name:
        m = 's'
        c = NRed

    if '400' in key:
        f = 'w'
        s = ':'
    else:
        f = c
        s = '-'

    return m, c, k, f, s