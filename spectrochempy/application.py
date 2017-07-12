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

"""
This module define the application on which the API rely


"""

import os
import sys
import logging
from copy import deepcopy

from traitlets.config.configurable import Configurable
from traitlets.config.application import Application, catch_config_error

from traitlets import (
    Bool, Unicode, Int, List, Dict, default, observe
)

from IPython.core.magic import UsageError
from IPython import get_ipython

import matplotlib as mpl


# local
# =============================================================================

from spectrochempy.utils import is_kernel
from spectrochempy.utils import get_config_dir, get_pkg_data_dir
from spectrochempy.version import get_version
from spectrochempy.core.plotters.plottersoptions import PlotOptions
from spectrochempy.core.readers.readersoptions import ReadOptions
from spectrochempy.core.writers.writersoptions import WriteOptions
from spectrochempy.core.processors.processorsoptions import ProcessOptions
from spectrochempy.utils.file import get_pkg_data_filename

# For wild importing using the *, we limit the methods, objetcs, ...
# that this method exposes
# ------------------------------------------------------------------------------
__all__ = ['scp']

# ==============================================================================
# PYTHONPATH
# ==============================================================================
# in case spectrochempy was not yet installed using setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ==============================================================================
# Main application and configurators
# ==============================================================================

class SpectroChemPy(Application):



    # info _____________________________________________________________________

    name = Unicode(u'SpectroChemPy')
    description = Unicode(u'This is the main SpectroChemPy application ')

    version = Unicode('').tag(config=True)
    release = Unicode('').tag(config=True)
    copyright = Unicode('').tag(config=True)

    classes = List([PlotOptions,])

    # configuration parameters  ________________________________________________

    reset_config = Bool(False,
            help='should we restaure a default configuration?').tag(config=True)

    config_file_name = Unicode(None,
                                  help="Load this config file").tag(config=True)

    @default('config_file_name')
    def _get_config_file_name_default(self):
        return self.name + u'_config.py'

    config_dir = Unicode(None,
                     help="Set the configuration dir location").tag(config=True)

    @default('config_dir')
    def _get_config_dir_default(self):
        return get_config_dir()


    data_dir = Unicode(help="Set a data directory where to look for data").tag(
            config=True)

    @default('data_dir')
    def _get_data_dir_default(self):
        # look for the testdata path in package tests
        return get_pkg_data_dir('testdata','tests')

    info_on_loading = Bool(True,
                                help='display info on loading').tag(config=True)

    running = Bool(False,
                   help="Is SpectrochemPy running?").tag(config=True)

    debug = Bool(False,
                 help='set DEBUG mode, with full outputs').tag(config=True)

    quiet = Bool(False,
                 help='set Quiet mode, with minimal outputs').tag(config=True)


    # --------------------------------------------------------------------------
    # Initialisation of the plot options
    # --------------------------------------------------------------------------

    def _init_plotoptions(self):

        # Pass config to other classes for them to inherit the config.
        self.plotoptions = PlotOptions(config=self.config)

        # set default matplotlib options
        mpl.rc('text', usetex=self.plotoptions.use_latex)

        if self.plotoptions.latex_preamble == []:
            self.plotoptions.latex_preamble = [
                r'\usepackage{siunitx}',
                r'\sisetup{detect-all}',
                r'\usepackage{times}',  # set the normal font here
                r'\usepackage{sansmath}',
                # load up the sansmath so that math -> helvet
                r'\sansmath'
            ]

    # --------------------------------------------------------------------------
    # Initialisation of the application
    # --------------------------------------------------------------------------

    @catch_config_error
    def initialize(self, argv=None):
        """
        Initialisation function for the API applications

        Parameters
        ----------
        argv :  List, [optional].
            List of configuration parameters.

        Returns
        -------
        app : Application.
            The application handler.

        """
        # matplotlib use directive to set before calling matplotlib backends
        # ------------------------------------------------------------------
        # we performs this before any call to matplotlib that are performed
        # later in this application

        # if we are building the docs, in principle it should be done using
        # the make_scp_docs.py located in the scripts folder
        if not 'make_scp_docs.py' in sys.argv[0]:
            # the normal backend
            mpl.use('Qt5Agg')
            mpl.rcParams['backend.qt5'] = 'PyQt5'
        else:
            # 'agg' backend is necessary to build docs with sphinx-gallery
            mpl.use('agg')

        ip = get_ipython()
        if ip is not None:

            if is_kernel():

                # set the ipython matplotlib environments
                try:
                    ip.magic('matplotlib nbagg')
                except UsageError:
                    try:
                        ip.magic('matplotlib osx')
                    except:
                        ip.magic('matplotlib qt')
            else:
                try:
                    ip.magic('matplotlib osx')
                except:
                    ip.magic('matplotlib qt')

        # parse the argv
        # --------------

        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)

        _do_parse = True
        for arg in ['egg_info', '--egg-base', 
                    'pip-egg-info', 'develop', '-f', '-x']:
            if arg in sys.argv:
                _do_parse = False

        # print("*" * 50, "\n", sys.argv, "\n", "*" * 50)

        if _do_parse:
            self.parse_command_line(argv)

        # Get options from the config file
        # --------------------------------

        if self.config_file_name :

            config_file = os.path.join(self.config_dir, self.config_file_name)
            self.load_config_file(config_file)

        # add other options
        # -----------------

        self._init_plotoptions()

        # Test, Sphinx,  ...  detection
        # ------------------------------

        _do_not_block = self.plotoptions.do_not_block

        for caller in ['make_scp_docs.py', 'pytest', 'py.test', 'docrunner.py']:

            if caller in sys.argv[0]:

                # this is necessary to build doc
                # with sphinx-gallery and doctests

                _do_not_block = self.plotoptions.do_not_block = True
                self.log.warning(
                    'Running {} - set do_not_block: {}'.format(
                                                         caller, _do_not_block))

        self.log.debug("DO NOT BLOCK : %s " % _do_not_block)

        # version
        # --------
        self.version, self.release, self.copyright = get_version()

        # Possibly write the default config file
        # ---------------------------------------
        self._make_default_config_file()

        # exception handler
        # ------------------
        if ip is not None:

            def custom_exc(shell, etype, evalue, tb, tb_offset=None):
                if self.log_level == logging.DEBUG:
                    shell.showtraceback((etype, evalue, tb),
                                        tb_offset=tb_offset)
                else:
                    self.log.error("%s: %s" % (etype.__name__, evalue))

            ip.set_custom_exc((Exception,), custom_exc)

        else:

            def exceptionHandler(exception_type, exception, traceback,
                                 debug_hook=sys.excepthook):
                if self.log_level == logging.DEBUG:
                    debug_hook(exception_type, exception, traceback)
                else:
                    self.log.error(
                            "%s: %s" % (exception_type.__name__, exception))

            sys.excepthook = exceptionHandler

    # --------------------------------------------------------------------------
    # start the application
    # --------------------------------------------------------------------------

    def start(self, **kwargs):
        """

        Parameters
        ----------
        kwargs : options to pass to the application.

        Examples
        --------

        >>> app = SpectroChemPy()
        >>> app.initialize()
        >>> app.start(
            reset_config=True,   # option for restoring default configuration
            debug=True,          # debugging logs
            )

        """

        try:

            if self.running:
                self.log.debug('API already started. Nothing done!')
                return

            for key in kwargs.keys():
                if hasattr(self, key):
                    setattr(self, key, kwargs[key])

            self.log_format = '%(highlevel)s %(message)s'

            if self.quiet:
                self.log_level = logging.CRITICAL

            if self.debug:
                self.log_level = logging.DEBUG
                self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'


            info_string = u"""
        SpectroChemPy's API
        Version   : {}
        Copyright : {}
            """.format(self.version, self.copyright)

            if self.info_on_loading and \
                    not self.plotoptions.do_not_block:

                print(info_string)

            self.log.debug("The application was launched with ARGV : %s" % str(sys.argv))

            self.running = True

            return True

        except:

            return False

    # --------------------------------------------------------------------------
    # Store default configuration file
    # --------------------------------------------------------------------------
    def _make_default_config_file(self):
        """auto generate default config file."""

        fname = config_file = os.path.join(self.config_dir,
                                           self.config_file_name)

        if not os.path.exists(fname) or self.reset_config:
            s = self.generate_config_file()
            self.log.warning("Generating default config file: %r"%(fname))
            with open(fname, 'w') as f:
                f.write(s)

    # --------------------------------------------------------------------------
    # Events from Application
    # --------------------------------------------------------------------------

    @observe('log_level')
    def _log_level_changed(self, change):
        self.log_format = '%(highlevel)s %(message)s'
        if change.new == logging.DEBUG:
            self.log_format = '[%(name)s %(asctime)s]%(highlevel)s %(message)s'
        self.log.level = self.log_level
        self.log.debug("changed default loglevel to {}".format(change.new))

scp = SpectroChemPy()
scp.initialize()

#TODO: look at the subcommands capabilities of traitlets
if __name__ == "__main__":

    scp.start(
            reset_config=True,
            log_level = logging.INFO,
    )

    # ==============================================================================
    # Logger
    # ==============================================================================
    log = scp.log

    log.info('Name : %s ' % scp.name)

    scp.plotoptions.use_latex = True

    log.info(scp.plotoptions.latex_preamble)
