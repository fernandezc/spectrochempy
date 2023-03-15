# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import logging
import warnings
from os import environ
from pathlib import Path

import traitlets as tr

from spectrochempy.application.metaconfigurable import MetaConfigurable


# ======================================================================================
# General Preferences
# ======================================================================================
class GeneralPreferences(MetaConfigurable):
    """
    Preferences that apply to the |scpy| application in general.

    They should be accessible from the main API.
    """

    # ----------------------------------------------------------------------------------
    # Non configurable attributes
    # ----------------------------------------------------------------------------------
    name = tr.Unicode("GeneralPreferences")
    description = tr.Unicode("General options for the SpectroChemPy application")
    updated = tr.Bool(False)

    # ----------------------------------------------------------------------------------
    # Configurable attributes
    # ----------------------------------------------------------------------------------
    datadir = tr.Union(
        (tr.Instance(Path), tr.Unicode()),
        help="Directory where to look for data by default",
    ).tag(config=True)

    csv_delimiter = tr.Enum(
        [",", ";", r"\t", " "], default_value=",", help="CSV data delimiter"
    ).tag(config=True)

    project_directory = tr.Union(
        (tr.Instance(Path), tr.Unicode()),
        help="Directory where projects are stored by default",
    ).tag(config=True)

    use_qt = tr.Bool(
        help="Use QT for dialog instead of TK which is the default. "
        "If True the PyQt libraries must be installed",
    ).tag(config=True)

    # GUI
    # show_close_dialog = tr.Bool(
    #     True,
    #     help="Display the close project dialog project changing "
    #     "or on application exit",
    # ).tag(config=True)
    #
    # last_project = tr.Union(
    #     (tr.Instance(Path, allow_none=True), tr.Unicode()), help="Last used project"
    # ).tag(config=True)
    #
    # port = tr.Integer(7000, help="Dash server port").tag(config=True)
    #
    #
    # autoload_project = tr.Bool(
    #     True, help="Automatic loading of the last project at startup"
    # ).tag(config=True)
    #
    # autosave_project = tr.Bool(
    #     True, help="Automatic saving of the current project"
    # ).tag(config=True)
    #
    # workspace = tr.Union(
    #     (tr.Instance(Path), tr.Unicode()), help="Workspace directory by default"
    # ).tag(config=True)
    #
    # databases_directory = tr.Union(
    #     (tr.Instance(Path), tr.Unicode()),
    #     help="Directory where to look for database files such as csv",
    # ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Private methods and properties
    # ----------------------------------------------------------------------------------
    @tr.default("project_directory")
    def _get_default_project_directory(self):
        # Determines the SpectroChemPy project directory name and creates the directory if it doesn't exist.
        # This directory is typically ``$HOME/spectrochempy/projects``, but if the SCP_PROJECTS_HOME environment
        # variable is set and the `$SCP_PROJECTS_HOME` directory exists, it will be that directory.
        # If neither exists, the former will be created.

        # first look for SCP_PROJECTS_HOME
        pscp = environ.get("SCP_PROJECTS_HOME")
        if pscp is not None and Path(pscp).exists():
            return Path(pscp)

        pscp = Path.home() / ".spectrochempy" / "projects"

        pscp.mkdir(exist_ok=True)

        if pscp.is_file():
            raise IOError("Intended Projects directory is actually a file.")

        return pscp

    @tr.default("workspace")
    def _workspace_default(self):
        # the spectra path in package data
        return Path.home()

    @tr.default("datadir")
    def _datadir_default(self):
        from spectrochempy.utils.file import pathclean

        return pathclean(self.parent.datadir.path)

    @tr.observe("datadir")
    def _datadir_changed(self, change):
        from spectrochempy.utils.file import pathclean

        self.parent.datadir.path = pathclean(change["new"])

    @tr.validate("datadir")
    def _datadir_validate(self, proposal):
        # validation of the datadir attribute
        from spectrochempy.utils.file import pathclean

        datadir = proposal["value"]
        if isinstance(datadir, str):
            datadir = pathclean(datadir)
        return datadir

    @property
    def log_level(self):
        """
        Logging level (int).
        """
        return self.parent.log_level

    @log_level.setter
    def log_level(self, value):
        if isinstance(value, str):
            value = getattr(logging, value, None)
            if value is None:  # pragma: no cover
                warnings.warn(
                    "Log level not changed: invalid value given\n"
                    "string values must be 'DEBUG', 'INFO', 'WARNING', "
                    "or 'ERROR'"
                )
        self.parent.log_level = value

    @tr.default("use_qt")
    def _use_qt(self):
        from spectrochempy.utils import optional

        pyqt = optional.import_optional_dependency("PyQt5.QtWidgets", errors="ignore")
        if pyqt is not None:
            return True
        return False

    # GUI
    # @tr.default("databases_directory")
    # def _databases_directory_default(self):
    #     # the spectra path in package data
    #     from spectrochempy.utils.file import pathclean
    #     from spectrochempy.utils.packages import get_pkg_path
    #
    #     return pathclean(get_pkg_path("databases", "scp_data"))
    # @tr.observe("last_project")
    # def _last_project_changed(self, change):
    #     if change.name in self.traits(config=True):
    #         self.config_manager.update(
    #             self.config_file_name,
    #             {
    #                 self.__class__.__name__: {
    #                     change.name: change.new,
    #                 }
    #             },
    #         )

    # ----------------------------------------------------------------------------------
    # Class Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(section="GeneralPreferences", **kwargs)
