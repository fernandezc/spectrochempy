"""Sentinel class for constants with useful reprs."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.


class Sentinel:
    def __init__(self, name, module, docstring=None):
        self.name = name
        self.module = module
        if docstring:
            self.__doc__ = docstring

    def __repr__(self):
        return str(self.module) + "." + self.name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
