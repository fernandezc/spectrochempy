# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Define a LazyImporter class to do lazy imports
"""

import os

from traitlets import import_item


class LazyImporter(object):
    def __init__(self, name, file, api):
        self._api = api
        self.__all__ = list(api.keys())
        self.__file__ = file
        self.__path__ = [os.path.dirname(file)]
        self.__name__ = name
        self.__spec__ = None

    def __dir__(self):
        return self.__all__

    def __getattr__(self, name: str):
        if name in self._api:
            value = import_item(self._api[name] + "." + name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")
        setattr(self, name, value)
        return value


if __name__ == "__main__":
    """ """
