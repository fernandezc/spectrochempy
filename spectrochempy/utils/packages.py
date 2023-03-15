# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import os
import sys
from pkgutil import walk_packages

from traitlets import import_item


# ======================================================================================
# PACKAGE and API UTILITIES
# ======================================================================================
def list_packages(package):
    """
    Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709.

    names = [package.__name__]
    for __, name, __ in walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):
        names.append(name)

    return sorted(names)


def generate_api(api_path):
    # name of the package

    dirname, name = os.path.split(os.path.split(api_path)[0])

    if not dirname.endswith("spectrochempy"):
        dirname, _name = os.path.split(dirname)
        name = _name + "." + name
    pkgs = sys.modules["spectrochempy.%s" % name]
    api = sys.modules["spectrochempy.%s" % name]

    pkgs = list_packages(pkgs)

    all_ = []

    for pkg in pkgs:
        if pkg.endswith("api") or "test" in pkg:
            continue
        try:
            pkg = import_item(pkg)
        except Exception:
            if not hasattr(pkg, "__all__"):
                continue
            raise ImportError(pkg)
        if not hasattr(pkg, "__all__"):
            continue

        a = getattr(pkg, "__all__", [])

        all_ += a
        for item in a:

            # set general method for the current package API
            obj = getattr(pkg, item)
            setattr(api, item, obj)

    return all_


def get_pkg_path(data_name, package=None):
    data_name = os.path.normpath(data_name)

    path = os.path.dirname(import_item(package).__file__)
    path = os.path.join(path, data_name)

    if not os.path.isdir(path):  # pragma: no cover
        return os.path.dirname(path)

    return path


# ======================================================================================
if __name__ == "__main__":
    pass
