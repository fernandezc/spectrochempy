# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


__all__ = ['log', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
           'consolelog', 'filelog']

logger_all = __all__

import logging
import os
from logging import DEBUG, ERROR, INFO, WARNING, CRITICAL

from logging.handlers import RotatingFileHandler
from logging import StreamHandler, Formatter

log = logging.getLogger()
log.setLevel(WARNING)

formatter = Formatter('%(message)s')
filelogformatter = Formatter(
       '%(asctime)s :: %(levelname)8s :: %(module)s.%(funcName)s - %(message)s')

# File log

from spectrochempy.utils import *

log_file = "spectrochempy.log"
if not os.path.isabs(log_file):
    path = os.path.join(get_log_dir(), log_file)
else:
    path = log_file

filelog = RotatingFileHandler(path, mode="a", maxBytes=32768,
                              backupCount=5, encoding="utf-8")
filelog.setLevel(DEBUG)
filelog.setFormatter(filelogformatter)
log.addHandler(filelog)

# Console log

consolelog = StreamHandler()
consolelog.setLevel(logging.INFO)
consolelog.setFormatter(formatter)
log.addHandler(consolelog)

