# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
from pathlib import Path

import numpy as np
import traitlets as tr
from traitlets.config.configurable import Configurable
from traitlets.config.loader import LazyConfigValue

__all__ = []


class MetaConfigurable(Configurable):
    """
    A subclass of Configurable that stores configuration changes in a json file.

    Saving the configuration changes allows to retrieve them between different
    executions of the main application.
    """

    def __init__(self, section, **kwargs):  # lgtm[py/missing-call-to-init]

        super().__init__(**kwargs)

        self.cfg = self.parent.config_manager
        self.section = section

    def to_dict(self):
        """
        Return config value in a dict form.

        Returns
        -------
        dict
            A regular dictionary.
        """
        d = {}
        for k, v in self.traits(config=True).items():
            d[k] = v.default_value
        return d

    def trait_defaults(self, *names, **metadata):
        # override traitlets trait default to take into accound changes in
        # the config file
        defaults = super().trait_defaults(*names, **metadata)
        # modify with the loaded external config
        if not names:  # full dictionary
            config = self.config[self.section]
            if "shape" in config and isinstance(config["shape"], LazyConfigValue):
                del config["shape"]  # remove the lazy configurable object
            defaults.update(config)
        return defaults

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # update configuration after any change
        from matplotlib import cycler

        if not hasattr(self, "cfg"):
            # not yet initialized
            return

        if change.name in self.traits(config=True):

            value = change.new
            # replace non serializable value by an equivalent
            if isinstance(value, (type(cycler), Path)):
                value = str(value)
            if isinstance(value, np.ndarray):
                # we need to transform it to a list of elements, bUT with python built-in
                # types, which is not the case e.g., for int64
                value = value.tolist()

            self.cfg.update(
                self.section,
                {
                    self.__class__.__name__: {
                        change.name: value,
                    }
                },
            )

            self.updated = True


# ======================================================================================
if __name__ == "__main__":
    pass
