# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


"""Preferences for SpectroChemPy
"""

from apptools.preferences.ui.api import PreferencesPage

__all__ = ['RootPreferencesPage', 'NDDatasetPreferencesPage', 
           'PlotPreferencesPage']

from .preferences import *

################################################################################
# `RootPreferencesPage` class
################################################################################
class RootPreferencesPage(PreferencesPage, RootPreferences):
    """ Preferences page for SpectroChemPy. """

    #### 'PreferencesPage' interface ##########################################

    # The page's category (e.g. 'General/Appearance'). The empty string means
    # that this is a top-level page.
    category = ''

    # The page's help identifier (optional). If a help Id *is* provided then
    # there will be a 'Help' button shown on the preference page.
    help_id = ''

    # The page name (this is what is shown in the preferences dialog.
    name = 'General'

    # The path to the preferences node that contains the preferences.
    preferences_path = 'spectrochempy'



################################################################################
# `PlotPreferencesPage` class
################################################################################
class PlotPreferencesPage(PreferencesPage, PlotPreferences):
    """ Preferences page for SpectroChemPy. """

    #### 'PreferencesPage' interface ##########################################

    # The page's category (e.g. 'General/Appearance'). The empty string means
    # that this is a top-level page.
    category = 'Appereance'

    # The page's help identifier (optional). If a help Id *is* provided then
    # there will be a 'Help' button shown on the preference page.
    help_id = ''

    # The page name (this is what is shown in the preferences dialog.
    name = 'Plot'

    # The path to the preferences node that contains the preferences.
    preferences_path = 'spectrochempy.plot'


################################################################################
# `NDDatasetPreferencesPage` class
################################################################################
class NDDatasetPreferencesPage(PreferencesPage, NDDatasetPreferences):
    """ Preferences page for SpectroChemPy. """

    #### 'PreferencesPage' interface ##########################################

    # The page's category (e.g. 'General/Appearance'). The empty string means
    # that this is a top-level page.
    category = 'Data'

    # The page's help identifier (optional). If a help Id *is* provided then
    # there will be a 'Help' button shown on the preference page.
    help_id = ''

    # The page name (this is what is shown in the preferences dialog.
    name = 'Dataset'

    # The path to the preferences node that contains the preferences.
    preferences_path = 'spectrochempy.dataset'
