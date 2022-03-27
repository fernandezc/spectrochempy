# -*- coding: utf-8 -*-

# ======================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory
# ======================================================================================

from pathlib import Path, PosixPath, WindowsPath

from spectrochempy.utils.system import is_windows


def pathclean(paths):
    """
    Clean a path or a series of path.

    This cleaning is done in order to be compatible with windows and  unix-based system.

    Parameters
    ----------
    paths :  str or a list of str
        Path to clean. It may contain windows or conventional python separators.

    Returns
    -------
    out : a pathlib object or a list of pathlib objects
        Cleaned path(s)

    Examples
    --------
    >>> from spectrochempy.utils.pathlib import pathclean

    Using unix/mac way to write paths
    >>> filename = pathclean('irdata/nh4y-activation.spg')
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'

    or Windows
    >>> filename = pathclean("irdata\\\\nh4y-activation.spg")
    >>> filename.parent.name
    'irdata'

    Due to the escape character \\ in Unix, path string should be escaped \\\\ or the raw-string prefix `r` must be used
    as shown below
    >>> filename = pathclean(r"irdata\\nh4y-activation.spg")
    >>> filename.suffix
    '.spg'
    >>> filename.parent.name
    'irdata'
    """

    def _clean(path):
        if isinstance(path, (Path, PosixPath, WindowsPath)):
            path = path.name
        if is_windows():
            path = WindowsPath(path)  # pragma: no cover
        else:  # some replacement so we can handle window style path on unix
            path = path.strip()
            path = path.replace("\\", "/")
            path = path.replace("\n", "/n")
            path = path.replace("\t", "/t")
            path = path.replace("\b", "/b")
            path = path.replace("\a", "/a")
            path = PosixPath(path)
        return Path(path)

    if paths is not None:
        if isinstance(paths, (str, Path, PosixPath, WindowsPath)):
            path = str(paths)
            return _clean(path).expanduser()
        elif isinstance(paths, (list, tuple)):
            return [_clean(p).expanduser() if isinstance(p, str) else p for p in paths]

    return paths
