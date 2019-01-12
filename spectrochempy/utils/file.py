# -*- coding: utf-8 -*-
#
# ============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ============================================================================

import os
import sys
import io
import json
from pkgutil import walk_packages
from numpy.lib.format import read_array
from numpy.compat import asstr
from traitlets import import_item
import warnings

from spectrochempy.gui.dialogs import opendialog

__all__ = ['readfilename',
           'list_packages', 'generate_api',
           'make_zipfile', 'ScpFile',
           'unzip'  #tempo
           ]

# =============================================================================
# Utility function
# =============================================================================

def readfilename(filename=None, **kwargs):
    """
    returns a list of the filenames of existing files, filtered by extensions

    Parameters
    ----------
    filename: `str`, `list` of strings, optional.
        A filename or a list of filenames. If not provided, a dialog box is opened
        to select files.
    directory: `str`, optional.
        The directory where to look at. If not specified, read in
        default dirdata directory
    filetypes: `list`, optional, default=['all files, '.*)'].

    Returns
    --------
        list of filenames

    """

    from spectrochempy.application import general_preferences as prefs
    from spectrochempy.utils import SpectroChemPyWarning


    # if the directory is not specified we look in the prefs.datadir
    directory = kwargs.get("directory", None)

    # filters and filetype will be alias (as filters is sometimes used)
    filetypes = kwargs.get("filetypes",
                           kwargs.get("filters", ["all files (*)"]))

    if filename:
        # if a filename or a list of filename was provided
        # first look if it really a filename and not a directory
        if isinstance(filename, str) and os.path.isdir(filename):
            warnings.warn('a directory has been provided instead of a filename!\n',
                          SpectroChemPyWarning)
            # we use it instead of the eventually passed directory
            if directory:
                warnings.warn('a directory has also been provided!'
                              ' we will use it instead of this one\n',
                              SpectroChemPyWarning)
            else:
                directory = filename
            filename = None

    if directory and not os.path.exists(directory):
        # well the directory doen't exist - we cannot go further without correcting the error
        raise IOError("directory %s doesn't exists!" % directory)


    # now proceed with the filenames
    if filename:
        _filenames = []
        # make a list, even for a single file name
        filenames = filename
        if not isinstance(filenames,(list, tuple)):
            filenames = list([filenames])
        else:
            filenames = list(filenames)

        # look if all the filename exists either in the specified directory,
        # else in the current directory, and finaly in the default preference data directory
        for i, filename in enumerate(filenames):

            if directory:
                _f = os.path.expanduser(os.path.join(directory, filename))
            else:
                _f = filename
                if not os.path.exists(_f):
                    # the filename provided doesn't exists in the specified directory
                    # or the current directory
                    # let's try in the default data directory
                    _f = os.path.join(prefs.datadir, filename)
                    if not os.path.exists(_f):
                        raise IOError("Can't find  this filename %s in the specified directory "
                                      "(or the current one if it was not specified, "
                                      "nor in the default data directory %s"%(filename, prefs.datadir))
            _filenames.append(_f)

        # now we have all the filename with their correct location
        filename = _filenames

    if not filename:
        # open a file dialog
        # currently Scpy use QT (needed for next GUI features)

        if not directory:
            # if no directory was eventually specified
            directory = prefs.datadir

        caption = kwargs.get('caption', 'Select file(s)')

        filename = opendialog(  single=False,
                                directory=directory,
                                caption=caption,
                                filters = filetypes)

        if not filename:
            # if the dialog has been cancelled or return nothing
            return None


    if isinstance(filename, list):
        if not all(isinstance(elem, str) for elem in filename):
            raise IOError('one of the list elements is not a filename!')
        else:
            filenames = filename
        #    filenames = [os.path.join(directory, elem) for elem in filename]   "already the full path

    if isinstance(filename, str):
        filenames = [filename]

    # filenames passed
    files = {}
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        if extension in files.keys():
            files[extension].append(filename)
        else:
            files[extension] = [filename]
    return files




# ============================================================================
# PACKAGE and API UTILITIES
# ============================================================================

# ............................................................................
def list_packages(package):
    """Return a list of the names of a package and its subpackages.

    This only works if the package has a :attr:`__path__` attribute, which is
    not the case for some (all?) of the built-in packages.
    """
    # Based on response at
    # http://stackoverflow.com/questions/1707709

    names = [package.__name__]
    for __, name, __ in walk_packages(package.__path__,
                                      prefix=package.__name__ + '.',
                                      onerror=lambda x: None):
        names.append(name)

    return names


# ............................................................................
def generate_api(api_path):

    # name of the package
    dirname, name = os.path.split(os.path.split(api_path)[0])
    if not dirname.endswith('spectrochempy'):
        dirname, _name = os.path.split(dirname)
        name = _name+'.'+name
    pkgs = sys.modules['spectrochempy.%s' % name]
    api = sys.modules['spectrochempy.%s.api' % name]

    pkgs = list_packages(pkgs)

    __all__ = []

    for pkg in pkgs:
        if pkg.endswith('api') or "test" in pkg:
            continue
        try:
            pkg = import_item(pkg)
        except:
            raise ImportError(pkg)
        if not hasattr(pkg, '__all__'):
            continue
        a = getattr(pkg, '__all__',[])
        dmethods = getattr(pkg, '__dataset_methods__', [])
        __all__ += a
        for item in a:

            # set general method for the current package API
            setattr(api, item, getattr(pkg, item))

            # some  methods are class method of NDDatasets
            if item in dmethods:
                from spectrochempy.dataset.nddataset import NDDataset
                setattr(NDDataset, item, getattr(pkg, item))

    return __all__

# ============================================================================
# ZIP UTILITIES
# ============================================================================

# ............................................................................
def make_zipfile(file, **kwargs):
    """
    Create a ZipFile.

    Allows for Zip64 (useful if files are larger than 4 GiB, and the `file`
    argument can accept file or str.
    `kwargs` are passed to the zipfile.ZipFile
    constructor.

    (adapted from numpy)

    """
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, **kwargs)


# ............................................................................
def unzip(source_filename, dest_dir):
    """
    Unzip a zipped file in a directory

    Parameters
    ----------
    source_filename
    dest_dir

    Returns
    -------

    """
    import zipfile
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                path = os.path.join(path, word)
            zf.extract(member, path)


class ScpFile(object):
    """
    ScpFile(fid)

    (largely inspired by ``NpzFile`` object in numpy)

    `ScpFile` is used to load files stored in ``.scp`` or ``.pscp``
    format.

    It assumes that files in the archive have a ``.npy`` extension in
    the case of the dataset's ``.scp`` file format) ,  ``.scp``  extension
    in the case of project's ``.pscp`` file format and finally ``pars.json``
    files which contains other information on the structure and  attibutes of
    the saved objects. Other files are ignored.

    """

    def __init__(self, fid):
        """
        Parameters
        ----------
        fid : file or str
            The zipped archive to open. This is either a file-like object
            or a string containing the path to the archive.

        """
        _zip = make_zipfile(fid)

        self.files = _zip.namelist()
        self.zip = _zip

        if hasattr(fid, 'close'):
            self.fid = fid
        else:
            self.fid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close the file.

        """
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def __del__(self):
        self.close()

    def __getitem__(self, key):

        member = False
        base = None
        ext = None

        if key in self.files:
            member = True
            base, ext = os.path.splitext(key)

        if member and ext in [".npy"]:
            f = self.zip.open(key)
            return read_array(f, allow_pickle=True)

        elif member and ext in ['.scp']:
            from spectrochempy.dataset.nddataset import NDDataset
            f = io.BytesIO(self.zip.read(key))
            return NDDataset.load(f)

        elif member and ext in ['.json']:
            return json.loads(asstr(self.zip.read(key)))

        elif member :
            return self.zip.read(key)

        else:
            raise KeyError("%s is not a file in the archive or is not "
                           "allowed" % key)

    def __iter__(self):
        return iter(self.files)

    def items(self):
        """
        Return a list of tuples, with each tuple (filename, array in file).

        """
        return [(f, self[f]) for f in self.files]

    def iteritems(self):
        """Generator that returns tuples (filename, array in file)."""
        for f in self.files:
            yield (f, self[f])

    def keys(self):
        """Return files in the archive with a ``.npy``,``.scp`` or ``.json``
        extension."""
        return self.files

    def iterkeys(self):
        """Return an iterator over the files in the archive."""
        return self.__iter__()

    def __contains__(self, key):
        return self.files.__contains__(key)
