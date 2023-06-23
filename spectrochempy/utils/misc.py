# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Various methods and classes used in other part of the program.
"""

import uuid
from datetime import datetime, timezone

__all__ = [
    "make_new_object",
]


def make_new_object(objtype):
    """
    Make a new object of type obj.

    Parameters
    ----------
    objtype : the object type

    Returns
    -------
    new : the new object of same type
    """

    new = type(objtype)()

    # new id and date
    new._id = "{}_{}".format(type(objtype).__name__, str(uuid.uuid1()).split("-")[0])
    new._date = datetime.now(timezone.utc)

    return new
