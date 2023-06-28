# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module defines the `application` on which the API rely.

It also defines the default application preferences and IPython magic functions.
"""

import inspect
import io
import json
import logging

# import pprint
import sys
import traceback
import warnings
from contextlib import contextmanager
from os import environ
from pathlib import Path

import matplotlib as mpl
import traitlets as tr
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config.application import Application
from traitlets.config.configurable import Config
from traitlets.config.manager import BaseJSONConfigManager

from spectrochempy._api_info import api_info
from spectrochempy.application.datadir import DataDir


# --------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------
def get_config_dir():
    """
    Determines the SpectroChemPy configuration directory name and
    creates the directory if it doesn't exist.

    This directory is typically ``$HOME/.spectrochempy/config``,
    but if the
    SCP_CONFIG_HOME environment variable is set and the
    ``$SCP_CONFIG_HOME`` directory exists, it will be that
    directory.

    If neither exists, the former will be created.

    Returns
    -------
    config_dir : str
        The absolute path to the configuration directory.
    """
    from spectrochempy.utils.file import find_or_create_spectrochempy_dir

    # first look for SCPY_CONFIG_HOME
    scp = environ.get("SCPY_CONFIG_HOME")

    if scp is not None and Path(scp).exists():  # pragma: no cover
        return Path(scp)

    config = find_or_create_spectrochempy_dir() / "config"
    if not config.exists():  # pragma: no cover
        config.mkdir(exist_ok=True)

    return config


def get_log_dir():
    """
    Determines the SpectroChemPy log output directory name and
    creates the directory if it doesn't exist.

    This directory is typically ``$HOME/.spectrochempy/logs``,
    but if the
    SCPY_LOGS environment variable is set and the
    ``$SCPY_LOGS`` directory exists, it will be that
    directory.

    If neither exists, the former will be created.

    Returns
    -------
    log_dir : str
        The absolute path to the log directory.
    """
    from spectrochempy.utils.file import find_or_create_spectrochempy_dir

    # first look for SCP_LOGS
    log_dir = environ.get("SCP_LOGS")

    if log_dir is not None and Path(log_dir).exists():  # pragma: no cover
        return Path(log_dir)

    log_dir = find_or_create_spectrochempy_dir() / "logs"
    if not log_dir.exists():  # pragma: no cover
        log_dir.mkdir(exist_ok=True)

    return log_dir


# ======================================================================================
# The main SpectoChemPy Application
# ======================================================================================
class SpectroChemPy(Application):
    """
    This class SpectroChemPy is the main class, containing most of the setup,
    configuration and more.
    """

    name = tr.Unicode("SpectroChemPy")
    description = tr.Unicode("Main application")

    # ----------------------------------------------------------------------------------
    # Non configurable attributes
    # ----------------------------------------------------------------------------------
    # config
    config_file_name = tr.Unicode(None, help="Configuration file name")
    config_dir = tr.Instance(Path, help="Set the configuration directory location")
    config_manager = tr.Instance(BaseJSONConfigManager)

    # ----------------------------------------------------------------------------------
    # Configurable attributes
    # ----------------------------------------------------------------------------------
    # log
    log_dir = tr.Instance(Path, help="The log output directory location").tag(
        config=True
    )
    log_format = tr.Unicode(
        "%(highlevel)s %(message)s",
        help="The Logging format template",
    ).tag(config=True)
    logging_config = tr.Dict(
        {
            "handlers": {
                "string": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": "INFO",
                    "stream": io.StringIO(),
                },
            },
            "loggers": {
                "SpectroChemPy": {
                    "level": "DEBUG",
                    "handlers": ["console", "string"],
                },
            },
        }
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------------------------
    _running = tr.Bool(False)
    _loaded_config_files = tr.List()
    _from_warning_ = False

    # used to create configuration files
    classes = tr.List([])

    # ----------------------------------------------------------------------------------
    # Initialisation of the application
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log_level = kwargs.pop("log_level", None)
        self._initialize()
        if log_level is not None:
            self.log_level = log_level

    # ----------------------------------------------------------------------------------
    # Error/warning capture
    # ----------------------------------------------------------------------------------
    def _ipython_catch_exceptions(self, shell, etype, evalue, tb, tb_offset=None):
        # output the full traceback only in DEBUG mode
        if self.log_level == logging.DEBUG:
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        else:
            self.log.error(f"{etype.__name__}: {evalue}")

    def _catch_exceptions(self, etype, evalue, tb=None):
        # output the full traceback only in DEBUG mode
        with self._fmtcontext():
            if self.log_level == logging.DEBUG:
                # print(etype, type(etype))
                if isinstance(etype, str):
                    # probably the type was not provided!
                    evalue = etype
                    etype = Exception
                self.log.error(f"{etype.__name__}: {evalue}")
                if tb:
                    format_exception = traceback.format_tb(tb)
                    for line in format_exception:
                        parts = line.splitlines()
                        for p in parts:
                            self.log.error(p)
            else:
                self.log.error(f"{etype.__name__}: {evalue}")

    def _custom_warning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        with self._fmtcontext():
            self._formatter(message)
            self.log.warning(f"({category.__name__}) {message}")

    # ----------------------------------------------------------------------------------
    # Initialisation of the configurables
    # ----------------------------------------------------------------------------------
    def _init_all_preferences(self):
        # Get preferences from the config file
        if not self.config:
            self.config = Config()

        configfiles = []
        if self.config_dir:
            lis = self.config_dir.iterdir()
            for fil in lis:
                if fil.suffix == ".py":
                    pyname = self.config_dir / fil
                    configfiles.append(pyname)
                elif fil.suffix == ".json":
                    jsonname = self.config_dir / fil
                    # check integrity of the file
                    with jsonname.open() as f:
                        try:
                            json.load(f)
                        except json.JSONDecodeError:
                            jsonname.unlink()
                            continue
                    configfiles.append(jsonname)

            for cfgname in configfiles:
                self.load_config_file(cfgname)
                if cfgname not in self._loaded_config_files:
                    self._loaded_config_files.append(cfgname)

        from spectrochempy.application.general_preferences import GeneralPreferences
        from spectrochempy.application.plot_preferences import PlotPreferences

        self.general_preferences = GeneralPreferences(config=self.config, parent=self)
        self.plot_preferences = PlotPreferences(config=self.config, parent=self)

    # ----------------------------------------------------------------------------------
    # Private methods and properties
    # ----------------------------------------------------------------------------------
    @tr.default("config_dir")
    def _config_dir_default(self):
        return get_config_dir()

    @tr.default("log_dir")
    def _log_dir_default(self):
        return get_log_dir()

    @tr.default("config_manager")
    def _config_manager_default(self):
        return BaseJSONConfigManager(config_dir=str(self.config_dir))

    def _formatter(self, *args):
        # We need a custom formatter (maybe there is a better way to do this suing
        # the logging library directly?)

        rootfolder = Path(__file__).parent
        st = 2
        if "_showwarnmsg" in inspect.stack()[2][3]:
            st = 4 if self._from_warning_ else 3

        filename = Path(inspect.stack()[st][1])
        try:
            module = filename.relative_to(rootfolder)
        except ValueError:
            module = filename
        line = inspect.stack()[st][2]
        func = inspect.stack()[st][3]

        # rotatingfilehandler formatter (DEBUG)
        formatter = logging.Formatter(
            f"<%(asctime)s:{module}/{func}::{line}> %(message)s"
        )
        self.log.handlers[1].setFormatter(formatter)

    @contextmanager
    def _fmtcontext(self):
        fmt = self.log_format, self.log.handlers[1].formatter
        try:
            yield fmt
        finally:
            self.log_format = fmt[0]
            self.log.handlers[1].setFormatter(fmt[1])

    def _initialize(self):
        """
        Initialisation function for the API applications.
        """
        # Parse the argv.
        # if we are running this under ipython and jupyter notebooks
        # deactivate potential command line arguments
        # (such that those from jupyter which cause problems here)
        ipy = get_ipython() if InteractiveShell.initialized() else None

        # if ipy is None:
        # # remove argument not known by spectrochempy
        # if (
        #     "make.py" in sys.argv[0]
        #     or "pytest" in sys.argv[0]
        #     or "validate_docstrings" in sys.argv[0]
        # ):  # building docs
        #     options = []
        #     for item in sys.argv[:]:
        #         for k in list(self.flags.keys()):
        #             if item.startswith("--" + k) or k in ["--help", "--help-all"]:
        #                 options.append(item)
        #             continue
        #         for k in list(self.aliases.keys()):
        #             if item.startswith("-" + k) or k in [
        #                 "h",
        #             ]:
        #                 options.append(item)
        #     self.parse_command_line(options)
        # else:  # pragma: no cover
        #     self.parse_command_line(sys.argv)

        # Warning handler
        # we catch warnings and error for a lighter display to the end-user.
        # except if we are in debugging mode

        warnings.showwarning = self._custom_warning

        # exception handler
        if ipy is not None:  # pragma: no cover
            ipy.set_custom_exc((Exception,), self._ipython_catch_exceptions)
        else:
            sys.excepthook = self._catch_exceptions

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------

    def start(self, configurables=None):
        """
        Start SpectroChemPy application main loop and int preferences
        """
        if self._running:
            # API already started. Nothing done!
            return True

        self.datadir = DataDir()

        # Get preferences from the config file and init everything
        self._init_all_preferences()

        # force update of rcParams
        for rckey in mpl.rcParams.keys():
            key = rckey.replace("_", "__").replace(".", "_").replace("-", "___")
            try:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key)
            except ValueError:
                mpl.rcParams[rckey] = getattr(self.plot_preferences, key).replace(
                    "'", ""
                )
            except AttributeError:
                # print(f'{e} -> you may want to add it to PlotPreferences.py')
                pass

        self.plot_preferences.set_latex_font(self.plot_preferences.font_family)

        # Eventually write the default config file for all configurables
        # ------------------------------------------------------------------------------
        # self._make_default_config_file(configurables=configurables)

        self._running = True

        welcome_string = (
            f"SpectroChemPy's API - v.{api_info.version}\n"
            f"©Copyright {api_info.copyright}"
        )
        ipy = get_ipython()
        if ipy is not None and "TerminalInteractiveShell" not in str(
            ipy
        ):  # pragma: no cover
            api_info._display_info_string(message=welcome_string.strip())
        else:
            if "/bin/scpy" not in sys.argv[0]:  # deactivate for console scripts
                print(welcome_string.strip())

        self.debug_(
            f"API loaded with log level set to "
            f"{logging.getLevelName(self.log_level)}- application is ready"
        )

    def make_default_config_file(self, configurables=None):
        """auto generate default config files."""

        # remove old configuration file spectrochempy_cfg.py
        fname = self.config_dir / "spectrochempy_cfg.py"  # Old configuration file
        if fname.exists():
            fname.unlink()

        # create a configuration file for each configurables
        if configurables:
            self.classes.extend(configurables)

        config_classes = list(self._classes_with_config_traits(self.classes))
        for cls in config_classes:
            name = cls.__name__
            fname = self.config_dir / f"{name}.cfg.py"
            if fname.exists():
                continue
            """generate default config file from Configurables"""
            lines = [f"# Configuration file for SpectroChemPy::{name}"]
            lines.append("")
            lines.append("c = get_config()  # noqa")
            lines.append("")
            lines.append(cls.class_config_section([cls]))
            sfil = "\n".join(lines)
            self.log.info(f"Generating default config file: {fname}")
            with open(fname, "w") as fil:
                fil.write(sfil)

    def info_(self, msg, *args, **kwargs):
        """
        Formatted info message.
        """
        with self._fmtcontext():
            self._formatter(msg)
            self.log.info(msg, *args, **kwargs)

    def debug_(self, msg, *args, **kwargs):
        """
        Formatted debug message.
        """
        with self._fmtcontext():
            self._formatter(msg)
            self.log.debug("DEBUG | " + msg, *args, **kwargs)

    def error_(self, *args, **kwargs):
        """
        Formatted error message.
        """
        if isinstance(args[0], Exception):
            e = args[0]
            etype = type(e)
            emessage = str(e)
        elif len(args) == 1 and isinstance(args[0], str):
            from spectrochempy.utils import exceptions

            etype = exceptions.SpectroChemPyError
            emessage = str(args[0])
        elif len(args) == 2:
            etype = args[0] if args else kwargs.get("type", None)
            emessage = (
                args[1] if args and len(args) > 1 else kwargs.get("message", None)
            )
        else:
            raise KeyError("wrong arguments have been passed to error_")
        self._catch_exceptions(etype, emessage, None)

    def warning_(self, msg, *args, **kwargs):
        """
        Formatted warning message.
        """
        self._from_warning_ = True
        warnings.warn(msg, *args, **kwargs)
        self._from_warning_ = False

    # ----------------------------------------------------------------------------------
    # Dialog functions
    # ----------------------------------------------------------------------------------
    def save_dialog(
        self, filename=None, caption="Save as...", filters=("All Files (*)"), **kwargs
    ):  # pragma: no cover
        """
        Return a file where to save.
        """
        if self.general_preferences.use_qt:
            import pyqt

            parent = kwargs.pop(
                "Qt_parent", None
            )  # in case this is launched from spectrochempy_gui

            FileDialog = pyqt.QFileDialog
            _ = pyqt.QApplication([])

            f = _QTFileDialogs._save_filename(
                FileDialog,
                parent=parent,
                filename=filename,
                caption=caption,
                filters=filters,
            )
        else:
            from tkinter import filedialog

            f = _TKFileDialogs()._save_filename(
                filedialog, filename=filename, caption=caption, filters=filters
            )

        from spectrochempy.utils.paths import pathclean

        return pathclean(f)

    def open_dialog(
        self, single=True, directory=None, filters=("All Files (*)"), **kwargs
    ):  # pragma: no cover
        """
        Return one or several files to open.
        """
        if self.general_preferences.use_qt:
            import pyqt

            parent = kwargs.pop(
                "Qt_parent", None
            )  # in case this is launched from spectrochempy_gui

            FileDialog = pyqt.QFileDialog
            _ = pyqt.QApplication([])
            klass = _QTFileDialogs
        else:
            from tkinter import filedialog as FileDialog

            klass = _TKFileDialogs()
            parent = klass.root

        if directory is None:
            directory = ""
        if filters == "directory":
            caption = "Select a folder"
            f = klass._open_existing_directory(
                FileDialog, parent=parent, caption=caption, directory=str(directory)
            )
        elif single:
            f = klass._open_filename(FileDialog, parent=parent, filters=filters)
        else:
            f = klass._open_multiple_filenames(
                FileDialog, parent=parent, filters=filters
            )

        from spectrochempy.utils.paths import pathclean

        return pathclean(f)


class _QTFileDialogs:  # pragma: no cover
    @classmethod
    def _open_existing_directory(
        cls, FileDialog, parent=None, caption="Select a folder", directory=None
    ):
        from spectrochempy.application import preferences

        if directory is None:
            directory = str(preferences.datadir)

        options = FileDialog.DontResolveSymlinks | FileDialog.ShowDirsOnly
        directory = FileDialog.getExistingDirectory(
            parent=parent, caption=caption, directory=directory, options=options
        )

        if directory:
            return directory

        return None

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_filename(
        cls,
        FileDialog,
        parent=None,
        directory=None,
        caption="Select file",
        filters=None,
    ):
        from spectrochempy.application import preferences

        if directory is None:
            directory = str(preferences.datadir)

        filename, _ = FileDialog.getOpenFileName(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
        )
        if filename:
            return filename

        return None

    # noinspection PyRedundantParentheses
    @classmethod
    def _open_multiple_filenames(
        cls,
        FileDialog,
        parent=None,
        directory=None,
        caption="Select file(s)",
        filters=None,
    ):
        """
        Return one or several files to open
        """
        from spectrochempy.application import preferences

        if directory is None:
            directory = str(preferences.datadir)

        files, _ = FileDialog.getOpenFileNames(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
        )
        if files:
            return files

        return None

    @classmethod
    def _save_filename(
        cls,
        FileDialog,
        parent=None,
        filename=None,
        caption="Save as...",
        filters=None,
    ):
        directory = str(filename)

        options = (
            FileDialog.DontConfirmOverwrite
        )  # bug : this seems to work only with DontUseNativeDialog on OSX.
        # TODO: Check on windows and Linux
        # second problems: if we confirm overwrite here a new dialog is opened,
        # and thus the main one do not close on exit!
        filename, _ = FileDialog.getSaveFileName(
            parent=parent,
            caption=caption,
            directory=directory,
            filter=";;".join(filters),
            options=options,
        )
        if filename:
            return filename

        return None


class _TKFileDialogs:  # pragma: no cover
    def __init__(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.geometry("0x0+0+0")
        root.deiconify()
        root.lift()
        root.focus_force()
        self.root = root

    @staticmethod
    def _open_existing_directory(
        filedialog, parent=None, caption="Select a folder", directory=""
    ):
        directory = filedialog.askdirectory(
            # parent=parent,
            initialdir=directory,
            title=caption,
        )

        if directory:
            return directory

        return None

    @staticmethod
    def filetypes(filters):
        # convert QT filters to TK
        import re

        regex = r"(.*)\((.*)\)"
        filetypes = []
        for _filter in filters:
            matches = re.finditer(regex, _filter)
            match = list(matches)[0]
            g = list(match.groups())
            g[1] = g[1].replace("[0-9]", "")
            g[1] = g[1].replace("1[r|i]", "*.*")
            g[1] = g[1].replace("2[r|i]*", "*.*")
            g[1] = g[1].replace("3[r|i]*", "*.*")
            g[1] = g[1].replace(" ", ",")
            g[1] = tuple(set(g[1].split(",")))
            filetypes.append((g[0], (g[1])))
        return filetypes

    # noinspection PyRedundantParentheses
    def _open_filename(
        self,
        filedialog,
        parent=None,
        filters=None,
    ):
        filename = filedialog.askopenfilename(
            # parent=parent,
            filetypes=self.filetypes(filters),
            title="Select file to open",
        )

        if parent is not None:
            parent.destroy()

        if filename:
            return filename

        return None

    # noinspection PyRedundantParentheses
    def _open_multiple_filenames(self, filedialog, parent=None, filters=None):
        """
        Return one or several files to open
        """
        filename = filedialog.askopenfilenames(
            # parent=parent,
            filetypes=self.filetypes(filters) + [("all files", ("*"))],
            title="Select file(s) to open",
        )

        if parent is not None:
            parent.destroy()

        if filename:
            return filename

        return None

    def _save_filename(
        self,
        filedialog,
        # parent=None,
        filename="",
        caption="Save as...",
        filters=None,
    ):
        from spectrochempy.utils.paths import pathclean

        dftext = ""
        directory = "."
        if filename:
            filename = pathclean(filename)
            directory = filename.parent
            dftext = filename.suffix

        if not dftext:
            dftext = ".scp"

        filename = filedialog.asksaveasfilename(
            # parent=parent,
            title=caption,
            initialdir=str(directory),
            initialfile=filename.name,
            defaultextension=dftext,
            filetypes=self.filetypes(filters),
        )
        #        if parent is not None:
        #            parent.destroy

        if filename:
            return pathclean(filename)

        return None
