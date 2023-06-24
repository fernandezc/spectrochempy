# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Create the spectrochempy.__api__ file
"""
import sys
from pathlib import Path
from pkgutil import walk_packages

from traitlets import import_item

api_file_template = """# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
#
#
#    ###################################################################################
#    #           DO NOT MODIFY THIS FILE BECAUSE IT IS CREATED AUTOMATICALLY.          #
#    #   ANY MODIFICATION OF THIS FILE WILL BE CANCELLED AFTER THE COMMIT IN GITHUB.   #
#    ###################################################################################
#
#
# flake8: noqa
\"\"\"
SpectroChemPy API.
\"\"\"
import sys

from spectrochempy.utils.lazy_imports import LazyImporter


_api_methods = {
%(txt_api_methods)s
}
"""

dataset_methods_template = """# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
#
#
#    ###################################################################################
#    #           DO NOT MODIFY THIS FILE BECAUSE IT IS CREATED AUTOMATICALLY.          #
#    #   ANY MODIFICATION OF THIS FILE WILL BE CANCELLED AFTER THE COMMIT IN GITHUB.   #
#    ###################################################################################
#
#
# flake8: noqa
\"\"\"
NDDataset methods from API.
\"\"\"

# --------------------------------------------------------------------------------------
# API methods that are also NDDataset methods
# --------------------------------------------------------------------------------------
_dataset_methods = {
%(txt_dataset_methods)s
}
"""

exclude = ["spectrochempy"]
exclude_startswith = ["~"]
exclude_within = [
    "spectrochempy.core.utils",
    "spectrochempy.examples",
    "spectrochempy.extern",
]


def list_packages(package):
    """
    Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709.

    names = []  # package.__name__]
    for __, name, __ in walk_packages(
        package.__path__, prefix=package.__name__ + ".", onerror=lambda x: None
    ):

        s = name.split(".")[-1]
        if (
            name in exclude
            or any(s.startswith(ss) for ss in exclude_startswith)
            or any(ss in name for ss in exclude_within)
        ):
            continue
        names.append(name)

    return sorted(names)


def create_api():
    root = Path(__file__).parent.parent
    apifile = root / "spectrochempy" / "_api.py"
    if apifile.exists():
        apifile.unlink()  # we want a completely clean API
    datasetfile = root / "spectrochempy" / "_dataset_methods.py"
    if datasetfile.exists():
        datasetfile.unlink()  # we want a completely clean API

    # spectrochempy = import_item("spectrochempy")

    modules = list_packages(sys.modules["spectrochempy"])
    _api_methods = {}
    _dataset_methods = {}

    for module in modules:

        x = import_item(module)
        members = []
        if hasattr(x, "__all__"):
            members = x.__all__
        # print(module)
        for member in members:
            print("-> ", member)
            if member in _api_methods:
                print(f"Duplicate API method: {member} - skipping")
                continue
            _api_methods[member] = module

        methods = []
        if hasattr(x, "__dataset_methods__"):
            methods = x.__dataset_methods__
        for method in methods:
            _dataset_methods[method] = module

    # Now create the file
    txt_api_methods = ""
    for obj, module in _api_methods.items():
        txt_api_methods += f'    "{obj}": "{module}",\n'

    apifile.write_text(api_file_template % {"txt_api_methods": txt_api_methods})

    # Create also a file for the NDDataset methods
    txt_dataset_methods = ""
    for method, module in _dataset_methods.items():
        txt_dataset_methods += f'    "{method}": "{module}",\n'

    datasetfile.write_text(
        dataset_methods_template % {"txt_dataset_methods": txt_dataset_methods}
    )

    return _api_methods, _dataset_methods
