# This file is adapted from pandas (pandas.compat._optional)
# see https://github.com/pandas-dev/pandas/blob/master/pandas/compat/_optional.py
# BSD 3-Clause License

from __future__ import annotations

import contextlib
import importlib
import sys
import types
import warnings

from packaging.version import Version

VERSIONS = {
    "xarray": "*",
    "cantera": "2.5.1",
    "PyQt5": "*",
}

# A mapping from import name to package name (on PyPI) for packages where
# these two names are different.

INSTALL_MAPPING = {
    "jinja2": "Jinja2",
}


def get_module_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        version = getattr(module, "__VERSION__", None)
    if version is None:
        with contextlib.suppress(Exception):
            version = importlib.metadata.version(module.__name__).split("+")[0]
    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    return version


def import_optional_dependency(
    name: str,
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
):
    r"""
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : `str`
        The module name.
    extra : `str`
        Additional text to include in the ImportError message.
    errors : `str` {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore : If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. `io/html.py`)

    min_version : `str`, default: `None`
        Specify a minimum version that is different from the global
        minimum version required.

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'``.

    """
    if errors not in {"warn", "raise", "ignore"}:
        raise ValueError("errors must be one of {'warn', 'raise', 'ignore'}")

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use conda or pip to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as imp:
        if errors == "raise":
            raise ImportError(msg) from imp
        return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version is not None and minimum_version != "*":
        version = get_module_version(module_to_get)
        if Version(version) < Version(minimum_version):
            msg = (
                f"SpectroChemPy requires version '{minimum_version}' or newer of "
                f"'{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
                return None
            if errors == "raise":
                raise ImportError(msg)

    return module
