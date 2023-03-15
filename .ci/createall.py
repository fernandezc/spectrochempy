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
        if "~" in name:
            continue
        names.append(name)

    return sorted(names)


root = Path(__file__).parent.parent
apifile = root / "spectrochempy" / "_api.py"
if apifile.exists():
    apifile.unlink()  # we want a completely clean API
datasetfile = root / "spectrochempy" / "core" / "dataset" / "_dataset_methods.py"
if datasetfile.exists():
    datasetfile.unlink()  # we want a completely clean API

spectrochempy = import_item("spectrochempy")

modules = list_packages(sys.modules["spectrochempy"])
_obj_to_module = {}
_dataset_methods = {}

for module in modules:

    if module == "spectrochempy.core.readers.read_jcamp":
        pass
    x = import_item(module)

    members = []
    if hasattr(x, "__all__"):
        members = x.__all__
    # print(module)
    for member in members:
        # print("-> ", member)
        _obj_to_module[member] = module

    methods = []
    if hasattr(x, "__dataset_methods__"):
        methods = x.__dataset_methods__
    for method in methods:
        _dataset_methods[method] = module

# Now create the file
txt = """# -*- coding: utf-8 -*-
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
"""

for obj, module in _obj_to_module.items():
    txt += f'    "{obj}": "{module}",\n'

txt += """}


sys.modules[__name__] = LazyImporter(
    __name__,
    globals()["__file__"],
    _api_methods
)


# --------------------------------------------------------------------------------------
# Search for NDDataset method which can be used as API methods
# --------------------------------------------------------------------------------------
def __getattr__(name):
    from spectrochempy.core.dataset.nddataset import NDDataset
    if hasattr(NDDataset, name):
        return getattr(NDDataset, name)
    raise AttributeError
"""


apifile.write_text(txt)


# Create also a file for the NDDataset methods

txt = """# -*- coding: utf-8 -*-
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
dataset_methods = {
"""

for method, module in _dataset_methods.items():
    txt += f'    "{method}": "{module}",\n'

txt += """
}
"""

datasetfile.write_text(txt)
