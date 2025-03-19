# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""File utilities."""

import importlib.util
import re
import struct
import warnings
from os import environ
from pathlib import Path
from pathlib import PosixPath
from pathlib import WindowsPath

import numpy as np

# ======================================================================================
# API utilities
# ======================================================================================
# When a function is in __all__, it is imported in the API
__all__ = ["pathclean", "download_testdata"]


# region api
def pathclean(paths):
    """
    Clean a path or a series of path.

    The aim is to be compatible with windows and unix-based system.

    Parameters
    ----------
    paths :  `str` or a `list` of `str`
        Path to clean. It may contain Windows or conventional python separators.

    Returns
    -------
    pathlib or list of pathlib
        Cleaned path(s).
    """
    import platform

    def is_windows():
        return "Windows" in platform.platform()

    def _clean(path):
        if isinstance(path, (Path, PosixPath, WindowsPath)):  # noqa: UP038  (syntax error in pyfakefs with modern union operators)
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
        if isinstance(paths, (str, Path, PosixPath, WindowsPath)):  # noqa: UP038
            path = str(paths)
            return _clean(path).expanduser()
        if isinstance(paths, (list, tuple)):  # noqa: UP038
            return [_clean(p).expanduser() if isinstance(p, str) else p for p in paths]

    return paths


def download_testdata():
    from spectrochempy.application.preferences import preferences
    from spectrochempy.core.readers.importer import read
    from spectrochempy.utils.file import pathclean

    datadir = pathclean(preferences.datadir)
    # this process is relatively long, so we do not want to do it several time:
    downloaded = datadir / "__downloaded__"
    if not downloaded.exists():
        read(datadir, download_only=True)
        downloaded.touch(exist_ok=True)


# endregion api


# ======================================================================================
# Utility functions
# ======================================================================================


# region utility
def is_editable_install(package_name):
    """
    Check if a package is installed in editable mode.

    Parameters
    ----------
    package_name : str
        The name of the package to check.

    Returns
    -------
    bool
        True if the package is installed in editable mode, False otherwise.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    print("origin", spec.origin)  # noqa: T201
    return f"{package_name}/src" in spec.origin


def get_repo_path():
    """
    Get the repository path based on the installation mode.

    Returns
    -------
    Path
        The path to the repository.
    """
    if is_editable_install("spectrochempy"):
        return Path(__file__).parent.parent.parent.parent
    return Path(__file__).parent.parent


def fromfile(fid, dtype, count):
    # to replace np.fromfile in case of io.BytesIO object instead of byte
    # object
    t = {
        "uint8": "B",
        "int8": "b",
        "uint16": "H",
        "int16": "h",
        "uint32": "I",
        "int32": "i",
        "float32": "f",
        "char8": "c",
    }
    typ = t[dtype] * count
    if dtype.endswith("16"):
        count *= 2
    elif dtype.endswith("32"):
        count *= 4

    out = struct.unpack(typ, fid.read(count))
    if len(out) == 1:
        return out[0]
    return np.array(out)


def _insensitive_case_glob(pattern):
    def either(c):
        return f"[{c.lower()}{c.upper()}]" if c.isalpha() else c

    return "".join(map(either, pattern))


def patterns(filetypes, allcase=True):
    regex = r"\*\.*\[*[0-9-]*\]*\w*\**"
    patterns = []
    if not isinstance(filetypes, (list, tuple)):  # noqa: UP038
        filetypes = [filetypes]
    for ft in filetypes:
        m = re.finditer(regex, ft)
        patterns.extend([match.group(0) for match in m])
    if not allcase:
        return patterns
    return [_insensitive_case_glob(p) for p in patterns]


def _get_file_for_protocol(f, **kwargs):
    protocol = kwargs.get("protocol")
    if protocol is not None:
        if isinstance(protocol, str):
            if protocol in ["ALL"]:
                protocol = "*"
            if protocol in ["opus"]:
                protocol = "*.0*"
            protocol = [protocol]

        lst = []
        for p in protocol:
            lst.extend(list(f.parent.glob(f"{f.stem}.{p}")))
        if not lst:
            return None
        return f.parent / lst[0]
    return None


def check_filenames(*args, **kwargs) -> list | dict:
    """
    Process and validate input filenames, returning a standardized list or dictionary.

    This function handles various input formats for specifying files, including strings,
    Path objects, URLs, byte contents, or mixed collections. If no filenames are provided,
    it can open a file dialog to allow user selection.

    Parameters
    ----------
    *args : str, Path, bytes, list, tuple, dict, optional
        Input specification for files:
        - str/Path: Path to a file
        - bytes: File content
        - list/tuple of str/Path: Multiple file paths
        - list of bytes: Multiple file contents
        - dict: Already formatted filename-to-content mapping
        - URL string (starting with "http://" or "https://"): Remote resource

    **kwargs : dict, optional
        Additional parameters to control file handling.

    Returns
    -------
    list or dict
        - A list of Path objects for the files
        - A dictionary mapping filenames to content for byte data or URLs

    Other Parameters
    ----------------
    filename : str or Path, optional
        Path to a file if not provided in args.
    content : bytes, optional
        The content of a file as bytes.
    protocol : list, optional
        Supported protocols for file handling.
    directory : str or Path, optional
        Base directory to search for files.
    processed : bool, optional
        For TopSpin data: whether to use processed data.
    expno : int, optional
        For TopSpin data: experiment number.
    procno : int, optional
        For TopSpin data: processing number.
    iterdir : bool, optional
        Whether to iterate through directory contents.
    glob : str, optional
        Pattern for matching multiple files.

    Notes
    -----
    The function searches for files in the following order:
    1. In the specified directory
    2. In the current working directory
    3. In the application's data directory

    For TopSpin NMR data, directories can be treated as files with appropriate parameters.

    See Also
    --------
    check_filename_to_open : Process filenames specifically for opening
    check_filename_to_save : Process filenames specifically for saving
    """
    from spectrochempy.application.preferences import preferences as prefs

    filenames = []
    i = 0

    if args:
        for arg in args:
            if (isinstance(arg, str)) and (
                arg.startswith("http://") or arg.startswith("https://")
            ):
                # remote resource
                filenames.append(arg)
            elif isinstance(arg, str | Path | PosixPath | WindowsPath):
                # local file: allways converted to Path object
                arg = pathclean(arg)
                filenames.append(arg)

            if isinstance(arg, list | tuple):
                # add filenames recursively
                filenames.extend(check_filenames(*arg))

            if isinstance(arg, bytes):
                # in this case, one or several byte contents has been passed instead of filenames
                # as filename where not given we passed the 'unnamed' string
                # return a dictionary
                i = i + 1
                filenames.append({f"no_name_{i}": arg})

            if isinstance(arg, dict):
                # get directly the dictionary if it is in the form: name:bytecontent
                for key, value in arg.items():
                    if isinstance(value, bytes):
                        filenames.append({key: value})
                    else:
                        raise ValueError(
                            "A dictionary passed should contain only bytes"
                        )

    # Look for content in kwargs
    content = kwargs.pop("content", None)
    if content:
        # if filename is also provided
        filename = kwargs.pop("filename", None)
        if filename is None:
            i = i + 1
            filenames.append({f"no_name_{i}": arg})
        else:
            filenames.append({filename: content})

    # look into keyword filename
    filename = kwargs.pop("filename", None)
    if filename is not None:
        filenames.append(check_filenames(filename))

    # look into keyword directory
    kw_directory = pathclean(kwargs.pop("directory", None))
    if kw_directory and not kw_directory.is_dir():
        raise ValueError(
            f"Directory {kw_directory} does not exist. Did you provide the full path?"
        )

    # process filenames and directories

    files = []
    datadir = pathclean(prefs.datadir)

    for filename in filenames:
        # if filename is a dictionary
        if isinstance(filename, dict):
            files.append(filename)  # no changes
            continue

        # in which directory ?
        directory = filename.parent

        if directory.resolve() == Path.cwd() or directory == Path():
            directory = ""

        if directory and kw_directory and directory != kw_directory:
            # conflict we do not take into account the kw.
            warnings.warn(
                "Two different directory where specified (from args and keywords arg). "
                "Keyword `directory` will be ignored!",
                stacklevel=2,
            )
        elif not directory and kw_directory:
            # kw_directory is used if provided and directory is not
            filename = pathclean(kw_directory / filename)

        # check if the file exists here
        if not directory or str(directory).startswith("."):
            # search first in the current directory
            directory = Path.cwd()

        f = pathclean(directory / filename)

        fexist = f if f.exists() else _get_file_for_protocol(f, **kwargs)
        if fexist is None:
            f = pathclean(datadir / filename)
            fexist = f if f.exists() else _get_file_for_protocol(f, **kwargs)

        if fexist:
            filename = fexist

        if filename.is_dir():
            # Particular case for topspin where filename can be provided
            # as a directory only
            if "topspin" in kwargs.get("protocol", []):
                filename = _topspin_check_filename(filename, **kwargs)
            else:
                # we list the directory if iterdir is not False
                if kwargs.get("iterdir", True):
                    pattern = kwargs.get("glob", "*.*")
                    if kwargs.get("recursive", True):
                        files.extend(
                            check_filenames(
                                list(filename.glob(f"**/{pattern}"), **kwargs)
                            )
                        )
                    else:
                        files.extend(list(filename.glob(pattern)))
                else:
                    warnings.warn(
                        f"Directory ‘{filename}’ will be ignored as parameter `iterdir=False`",
                        stacklevel=2,
                    )

        if not isinstance(filename, list):
            filename = [filename]

        files.extend(filename)

    return files


def _topspin_check_filename(filename, **kwargs):
    if kwargs.get("iterdir", False) or kwargs.get("glob") is not None:
        # when we list topspin dataset we have to read directories, not directly files
        # we can retrieve them using glob patterns
        glob = kwargs.get("glob")
        if glob:
            files_ = list(filename.glob(glob))
        elif not kwargs.get("processed", False):
            files_ = list(filename.glob("**/ser"))
            files_.extend(list(filename.glob("**/fid")))
        else:
            files_ = list(filename.glob("**/1r"))
            files_.extend(list(filename.glob("**/2rr")))
            files_.extend(list(filename.glob("**/3rrr")))
    else:
        expno = kwargs.pop("expno", None)
        procno = kwargs.pop("procno", None)

        if expno is None:
            expnos = sorted(filename.glob("[0-9]*"))
            expno = expnos[0] if expnos else expno

        # read a fid or a ser
        if procno is None:
            f = filename / str(expno)
            files_ = [f / "ser"] if (f / "ser").exists() else [f / "fid"]

        else:
            # get the adsorption spectrum
            f = filename / str(expno) / "pdata" / str(procno)
            if (f / "3rrr").exists():
                files_ = [f / "3rrr"]
            elif (f / "2rr").exists():
                files_ = [f / "2rr"]
            else:
                files_ = [f / "1r"]

    # depending on the glob patterns too many files may have been selected : restriction to the valid subset
    filename = []
    for item in files_:
        if item.name in ["fid", "ser", "1r", "2rr", "3rrr"]:
            filename.append(item)

    return filename


def get_filenames(*filenames, **kwargs):
    """
    Return a list or dictionary of the filenames of existing files, filtered by extensions.

    Parameters
    ----------
    filenames : `str` or pathlib object, `tuple` or `list` of strings of pathlib object, optional.
        A filename or a list of filenames.
        If not provided, a dialog box is opened to select files in the current
        directory if no `directory` is specified).
    **kwargs
        Other optional keyword parameters. See Other Parameters.

    Returns
    -------
    out
        List of filenames.

    Other Parameters
    ----------------
    directory : `str` or pathlib object, optional.
        The directory where to look at. If not specified, read in
        current directory, or in the datadir if unsuccessful.
    filetypes : `list` , optional, default=['all files, '.*)'].
        File type filter.
    dictionary : `bool` , optional, default=True
        Whether a dictionary or a list should be returned.
    iterdir : bool, default=False
        Read all file (possibly limited by `filetypes` in a given `directory` .
    recursive : bool, optional,  default=False.
        Read also subfolders.

    Warnings
    --------
    if several filenames are provided in the arguments,
    they must all reside in the same directory!
    """
    from spectrochempy.application.preferences import preferences as prefs

    # allowed filetypes
    # -----------------
    # alias filetypes and filters as both can be used
    filetypes = kwargs.get("filetypes", kwargs.get("filters", ["all files (*)"]))

    # filenames
    # ---------
    if len(filenames) == 1 and isinstance(filenames[0], (list, tuple)):  # noqa: UP038
        filenames = filenames[0]

    filenames = pathclean(list(filenames))

    directory = None
    if len(filenames) == 1:
        # check if it is a directory
        try:
            f = get_directory_name(filenames[0])
        except OSError:
            f = None
        if f and f.is_dir():
            # this specify a directory not a filename
            directory = f
            filenames = None
    # else:
    #    filenames = pathclean(list(filenames))

    # directory
    # ---------
    kw_dir = pathclean(kwargs.pop("directory", None))
    if directory is None:
        directory = kw_dir

    if directory is not None:
        if filenames:
            # prepend to the filename (incompatibility between filename and directory specification
            # will result to a error
            filenames = [pathclean(directory / filename) for filename in filenames]
        else:
            directory = get_directory_name(directory)

    # check the parent directory
    # all filenames must reside in the same directory
    if filenames:
        parents = set()
        for f in filenames:
            parents.add(f.parent)
        if len(parents) > 1:
            raise ValueError(
                "filenames provided have not the same parent directory. "
                "This is not accepted by the read function.",
            )

        # use get_directory_name to complete eventual missing part of the absolute path
        directory = get_directory_name(parents.pop())

        filenames = [filename.name for filename in filenames]

    # now proceed with the filenames
    if filenames:
        # look if all the filename exists either in the specified directory,
        # else in the current directory, and finally in the default preference data directory
        temp = []
        for _i, filename in enumerate(filenames):
            if not (pathclean(directory / filename)).exists():
                # the filename provided doesn't exists in the working directory
                # try in the data directory
                directory = pathclean(prefs.datadir)
                if not (pathclean(directory / filename)).exists():
                    raise OSError(f"Can't find  this filename {filename}")
            temp.append(directory / filename)

        # now we have checked all the filename with their correct location
        filenames = temp

    else:
        # no filenames:
        # open a file dialog    # TODO: revise this as we have suppressed the dialogs
        # except if a directory is specified or iterdir is True.

        getdir = kwargs.get(
            "iterdir",
            directory is not None or kwargs.get("protocol") == ["topspin"],
            # or kwargs.get("protocol", None) == ["carroucell"],
        )

        if not getdir:
            # we open a dialog to select one or several files manually
            if environ.get("TEST_FILE", None) is not None:
                # happen for testing
                filenames = [prefs.datadir / environ.get("TEST_FILE")]

        else:
            if not directory:
                directory = get_directory_name(environ.get("TEST_FOLDER"))

            elif kwargs.get("protocol") == ["topspin"]:
                directory = get_directory_name(environ.get("TEST_NMR_FOLDER"))

            if directory is None:
                return None

            filenames = []

            if kwargs.get("protocol") != ["topspin"]:
                # automatic reading of the whole directory
                fil = []
                for pat in patterns(filetypes):
                    if kwargs.get("recursive", False):
                        pat = f"**/{pat}"
                    fil.extend(list(directory.glob(pat)))
                pattern = kwargs.get("pattern", ["*"])
                pattern = pattern if isinstance(pattern, list) else [pattern]
                for kw_pat in pattern:
                    kw_pat = _insensitive_case_glob(kw_pat)
                    if kwargs.get("recursive", False):
                        kw_pat = f"**/{kw_pat}"
                    fil2 = [f for f in list(directory.glob(kw_pat)) if f in fil]
                    filenames.extend(fil2)
            else:
                # Topspin directory detection
                filenames = [directory]

            # on mac case insensitive OS this cause doubling the number of files.
            # Eliminates doublons:
            filenames = list(set(filenames))
            filenames = [
                f for f in filenames if f.name not in [".DS_Store", "__index__"]
            ]
            filenames = pathclean(filenames)

        if not filenames:
            # problem with reading?
            return None

    # now we have either a list of the selected files
    if isinstance(filenames, list) and not all(
        isinstance(elem, (Path, PosixPath, WindowsPath))  # noqa: UP038
        for elem in filenames  # noqa: UP038
    ):
        raise OSError("one of the list elements is not a filename!")

    # or a single filename
    if isinstance(filenames, (str, Path, PosixPath, WindowsPath)):  # noqa: UP038
        filenames = [filenames]

    filenames = pathclean(filenames)
    for filename in filenames[:]:
        if filename.name.endswith(".DS_Store"):
            # sometime present in the directory (MacOSX)
            filenames.remove(filename)

    dictionary = kwargs.get("dictionary", True)
    protocol = kwargs.get("protocol")
    if dictionary and protocol != ["topspin"]:
        # make and return a dictionary
        filenames_dict = {}
        for filename in filenames:
            if filename.is_dir() and protocol != ["carroucell"]:
                continue
            extension = filename.suffix.lower()
            if not extension:
                if re.match(r"^fid$|^ser$|^[1-3][ri]*$", filename.name) is not None:
                    extension = ".topspin"
            elif extension[1:].isdigit():
                # probably an opus file
                extension = ".opus"
            if extension in filenames_dict:
                filenames_dict[extension].append(filename)
            else:
                filenames_dict[extension] = [filename]
        return filenames_dict
    return filenames


def find_or_create_spectrochempy_dir():
    directory = Path.home() / ".spectrochempy"

    directory.mkdir(exist_ok=True)  # Create directory only if it does not exist

    if directory.is_file():  # pragma: no cover
        msg = "Intended SpectroChemPy directory `{0}` is actually a file."
        raise OSError(msg.format(directory))

    return directory


def get_directory_name(directory, **kwargs):
    """
    Return a valid directory name.

    Parameters
    ----------
    directory : `str` or `pathlib.Path` object, optional.
        A directory name. If not provided, a dialog box is opened to select a directory.

    Returns
    -------
    out: `pathlib.Path` object
        valid directory name.

    """
    from spectrochempy.application.application import warning_
    from spectrochempy.application.preferences import preferences as prefs

    data_dir = pathclean(prefs.datadir)
    working_dir = Path.cwd()

    directory = pathclean(directory)

    if directory:
        # Search locally
        if directory.is_dir():
            # nothing else to do
            return directory

        if (working_dir / directory).is_dir():
            # if no parent directory: look at current working dir
            return working_dir / directory

        if (data_dir / directory).is_dir():
            return data_dir / directory

        raise OSError(f'"{directory!s}" is not a valid directory')

    warning_("No directory provided!")
    return None


def check_filename_to_save(dataset, filename=None, overwrite=False, **kwargs):
    from spectrochempy.application.application import info_

    filename = pathclean(filename)

    if filename and pathclean(filename).parent.resolve() == Path.cwd():
        filename = Path.cwd() / filename

    if not filename or overwrite or filename.exists():
        # no filename provided
        if filename is None or pathclean(filename).is_dir():
            filename = dataset.name
            filename = pathclean(filename).with_suffix(kwargs.get("suffix", ".scp"))

        # existing filename provided
        if filename.exists():
            if overwrite:
                info_(f"A file {filename} is already present and will be overwritten.")
            else:
                raise FileExistsError(
                    f"A file {filename} is already present. "
                    "Please use the `overwrite=True` flag to overwrite it."
                )

    return pathclean(filename)


def check_filename_to_open(*args, **kwargs):
    # Check the args and keywords arg to determine the correct filename

    filenames = check_filenames(*args, **kwargs)

    if filenames is None:  # not args and
        return None

    # filenames returned by check_filenames are always a list or a dictionary
    if not isinstance(filenames, list):
        raise ValueError("filenames should be a list")

    files = {}

    if filenames[0] is None:
        raise FileNotFoundError("No filename provided")

    for filename in filenames:
        if isinstance(filename, dict):
            ext = "frombytes"
        elif isinstance(filename, Path | PosixPath | WindowsPath):
            ext = filename.suffix.lower()
        else:
            raise ValueError("filename should be a Path object or a dictionary")

        # deal first with special case
        if not ext and re.match(r"^fid$|^ser$|^[1-3][ri]*$", filename.name) is not None:
            # probably a TopSpin file
            ext = ".topspin"

        elif ext[1:].isdigit():
            # probably an opus file
            ext = ".opus"

        # update the files dictionary
        if ext not in files:
            files[ext] = []
        files[ext].append(filename)

    return files


# endregion utility
