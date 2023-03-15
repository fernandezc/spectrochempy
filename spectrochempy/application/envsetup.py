# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import sys
from os import environ
from pathlib import Path

import matplotlib as mpl
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from traitlets import import_item


# --------------------------------------------------------------------------------------
def setup_environment():
    NO_DISPLAY = False
    _IN_IPYTHON = InteractiveShell.initialized()
    _IP = get_ipython()
    _KERNEL = getattr(_IP, "kernel", None)
    _IN_COLAB = "google.colab" in str(_IP)

    # Are we buildings the docs ?
    if Path(sys.argv[0]).name in ["make.py"]:  # pragma: no cover
        # if we are building the documentation, in principle it should be done
        # using the make.py located at the root of the spectrochempy package.
        NO_DISPLAY = True
        mpl.use("agg", force=True)

    # is there a --nodisplay flag
    if "--nodisplay" in sys.argv:
        _ = sys.argv.pop(sys.argv.index("--nodisplay"))  # pragma: no cover
        NO_DISPLAY = True

    # Are we running pytest?
    _IN_PYTEST = "pytest" in sys.argv[0] or "py.test" in sys.argv[0]
    if _IN_PYTEST:
        # if we are testing we also like a silent work with no figure popup!
        NO_DISPLAY = True

        # OK, but if we are doing individual function testing in PyCharm
        # it is interesting to see the plots and the file dialogs,
        # except if we set explicitly --nodisplay argument!
        # if len(sys.argv) > 1 and not any([arg.endswith(".py") for arg in
        # sys.argv[1:]]) and '--nodisplay' not in sys.argv:
        if (
            len(sys.argv) > 1
            and any((arg.split("::")[0].endswith(".py") for arg in sys.argv[1:]))
            and "--nodisplay" not in sys.argv
        ):  # pragma: no cover
            # individual module testing
            NO_DISPLAY = False

    # Are we running in PyCharm scientific mode?
    _IN_PYCHARM_SCIMODE = mpl.get_backend() == "module://backend_interagg"

    if (
        not (_IN_IPYTHON and _KERNEL) and not _IN_PYCHARM_SCIMODE and not NO_DISPLAY
    ):  # pragma: no cover
        backend = mpl.rcParams["backend"]  # 'Qt5Agg'
        mpl.use(backend, force=True)

    # Terminal output colors
    # ----------------------
    if not _IN_IPYTHON:
        # needed in Windows terminal - but must not be inited in Jupyter notebook
        from colorama import init as initcolor

        initcolor()

    # Matplotlib output
    # -----------------
    if _IN_IPYTHON and _KERNEL and not NO_DISPLAY:  # pragma: no cover
        try:
            if (
                "ipykernel_launcher" in sys.argv[0]
                and "--InlineBackend.rc={'figure.dpi': 96}" in sys.argv
            ):
                # We are running from NBSphinx - the plot must be inline to show up.
                _IP.run_line_magic("matplotlib", "inline")
            else:
                if _IN_COLAB:  # pragma: no cover
                    # allow using matplotlib widget
                    # from google.colab import output
                    output = import_item("google.colab").output
                    output.enable_custom_widget_manager()
                _IP.run_line_magic("matplotlib", "widget")
        except Exception:
            _IP.run_line_magic("matplotlib", "qt")

    SCPY_STARTUP_LOGLEVEL = environ.get("SCPY_STARTUP_LOGLEVEL", None)

    if SCPY_STARTUP_LOGLEVEL is None:
        if _IN_PYTEST:
            SCPY_STARTUP_LOGLEVEL = "DEBUG"
        else:
            SCPY_STARTUP_LOGLEVEL = "INFO"

    return NO_DISPLAY, SCPY_STARTUP_LOGLEVEL
