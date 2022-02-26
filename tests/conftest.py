# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================

# flake8: noqa
import pathlib
import pytest

# initialize a ipython session before calling spectrochempy
# ---------------------------------------------------------


@pytest.fixture(scope="session")
def session_ip():
    try:
        from IPython.testing.globalipapp import start_ipython

        return start_ipython()
    except ImportError:
        return None


@pytest.fixture(scope="module")
def ip(session_ip):
    yield session_ip


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover
    """whole test run finishes."""

    # cleaning
    cwd = pathlib.Path(__file__).parent.parent

    for f in list(cwd.glob("**/*.?scp")):
        f.unlink()
    for f in list(cwd.glob("**/*.jdx")):
        f.unlink()
    for f in list(cwd.glob("**/*.json")):
        if f.name != ".zenodo.json":
            f.unlink()
    for f in list(cwd.glob("**/*.log")):
        f.unlink()
    for f in list(cwd.glob("**/*.nc")):
        f.unlink()
    docs = cwd / "docs"
    for f in list(docs.glob("**/*.ipynb")):
        f.unlink()


try:
    # work only if spectrochempy is installed
    import spectrochempy
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "You must install spectrochempy and its dependencies before executing tests!"
    )

from spectrochempy import preferences as prefs, NDDataset, set_loglevel, DEBUG

# put in debug mode for development
set_loglevel(DEBUG)
