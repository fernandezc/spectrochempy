# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import contextlib
import functools
import operator
import os
import warnings

import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.testing.compare import calculate_rms, ImageAssertionError
from numpy.testing import assert_approx_equal  # noqa
from numpy.testing import assert_array_almost_equal  # noqa
from numpy.testing import assert_array_compare
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal  # noqa
from numpy.testing import assert_raises  # noqa


@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    Parameters
    ----------
    environ : dict(str)
        Environment variables to set

    Examples
    --------
    >>> import os
    >>> from spectrochempy.utils.testing import set_env
    >>> with set_env(PLUGINS_DIR=u'test/plugins'):
    ...     "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    """
    # https://stackoverflow.com/questions/2059482/
    # python-temporarily-modify-the-current-processs-environment/51754362
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


# ======================================================================================
# NDDataset comparison
# ======================================================================================
def gisinf(x):
    # copied from numpy.testing._private.utils
    try:
        # Modern numpy way
        from numpy import errstate
        from numpy import isinf
    except ImportError:
        # Fallback for numpy 2.0
        errstate = np.errstate
        isinf = np.isinf

    with errstate(invalid="ignore"):
        st = isinf(x)
        if isinstance(st, type(NotImplemented)):
            raise TypeError("isinf not supported for this type")
    return st


def _compare(x, y, decimal):
    # copied from numpy.testing._private.utils
    # Fix numpy import
    try:
        # Modern numpy way
        from numpy import any as npany
        from numpy import asarray
        from numpy import float64
        from numpy import issubdtype
        from numpy import number
        from numpy import result_type
    except ImportError:
        # Fallback for numpy 2.0
        asarray = np.asarray
        float64 = np.float64
        number = np.number
        result_type = np.result_type
        npany = np.any
        issubdtype = np.issubdtype

    try:
        if npany(gisinf(x)) or npany(gisinf(y)):
            xinfid = gisinf(x)
            yinfid = gisinf(y)
            if not (xinfid == yinfid).all():
                return False
            # if one item, x and y is +- inf
            if x.size == y.size == 1:
                return x == y
            x = x[~xinfid]
            y = y[~yinfid]
    except (TypeError, NotImplementedError):
        pass

    # make sure y is an inexact type to avoid abs(MIN_INT); will cause
    # casting of x later.
    dtype = result_type(y, 1.0)
    y = asarray(y, dtype=dtype)
    z = abs(x - y)

    if not issubdtype(z.dtype, number):
        z = z.astype(float64)  # handle object arrays

    return z < 1.5 * 10.0 ** (-decimal)


def compare_ndarrays(this, other, approx=False, decimal=6, data_only=False):
    # Comparison based on attributes:
    #        data, dims, mask, labels, units, meta

    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True
    thistype = this._implements()

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = (
            "data",
            "dims",
            "mask",
            "labels",
            "units",
            "meta",
        )

    for attr in attrs:
        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    # particular case of mask
                    if attr != "mask":
                        raise AssertionError(f"{thistype}.{attr} sizes are different.")
                    assert_array_equal(
                        other.mask,
                        this.mask,
                        f"{this} and {other} masks are different.",
                    )
                if attr in ["data", "mask"]:
                    if approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} attributes are not equal",
                        )
                else:
                    eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are different.",
                    )
            else:
                return False
        else:
            # unitless and dimensionless are supposed equal units
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}",
                    )
            else:
                raise AssertionError(f"{other} has no units")

    return True


def compare_coords(this, other, approx=False, decimal=3, data_only=False):
    # TODO: compare base on signficant digit for coordinate instead of decimals
    #  (that may not work for very small coordinates numbers)
    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True
    thistype = this._implements()

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = ["data", "labels", "units", "meta"]
        if not data_only:
            attrs.append("title")

    for attr in attrs:
        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                # to avoid deprecation warning issue for unequal array
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    raise AssertionError(f"{thistype}.{attr} sizes are different.")

                if attr == "data" and not (sattr is None and oattr is None):
                    # Special case for data_only with different units
                    if data_only and this.units is not None and other.units is not None:
                        # For data_only=True, units might be different
                        # If units have different dimensionality, we can't compare directly
                        if str(this.units.dimensionality) != str(
                            other.units.dimensionality
                        ):
                            continue
                        # For comparing data with different units but same dimensionality
                        # Skip the comparison since data is handled differently with units
                        pass
                    elif approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} attributes are not equal",
                        )
                elif attr == "mask" and not data_only:
                    # Skip mask comparison here since we've already done it above
                    pass
                else:
                    eq &= np.all(sattr == oattr)

                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are different.",
                    )
            else:
                return False
        else:
            if data_only:
                # Skip units comparison if data_only is True
                continue

            # unitless and dimensionless are supposed equals
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}",
                    )
            else:
                raise AssertionError(f"{other} has no units")

    # Handle data_only with different but compatible units
    if (
        data_only
        and "data" in attrs
        and this.units is not None
        and other.units is not None
    ) and str(this.units.dimensionality) == str(other.units.dimensionality):
        try:
            # Convert this value to other's units for comparison
            this_value = this.values
            other_value = other.values
            if np.allclose(this_value, other_value, rtol=10 ** (-decimal)):
                return True
        except Exception:  # noqa: S110
            pass  # If conversion fails, we've already done the basic comparison

    return True


def compare_datasets(
    this, other, approx=False, decimal=6, data_only=False, coords_data_only=False
):
    from spectrochempy.core.units import ur

    def compare(x, y):
        return _compare(x, y, decimal)

    eq = True

    # if not isinstance(other, NDArray):
    #     # try to make some assumption to make useful comparison.
    #     if isinstance(other, Quantity):
    #         otherdata = other.magnitude
    #         otherunits = other.units
    #     elif isinstance(other, (float, int, np.ndarray)):
    #         otherdata = other
    #         otherunits = False
    #     else:
    #         raise AssertionError(
    #             f"{this} and {other} objects are too different to be " f"compared."
    #         )
    #
    #     if not this.has_units and not otherunits:
    #         eq = np.all(this._data == otherdata)
    #     elif this.has_units and otherunits:
    #         eq = np.all(this._data * this._units == otherdata * otherunits)
    #     else:
    #         raise AssertionError(
    #         f"units of {this} and {other} objects does not match")
    #     return eq

    thistype = this._implements()

    if other.data is None and this.data is None and data_only:
        attrs = ["labels"]
    elif data_only:
        attrs = ["data"]
    else:
        attrs = this._attributes_()
        exclude = (
            "filename",
            "preferences",
            "description",
            "history",
            "created",
            "modified",
            "origin",
            "roi",
            "size",
            "name",
            "modeldata",
            "processeddata",
            "baselinedata",
            "referencedata",
            "state",
        )
        for attr in exclude:
            # these attributes are not used for comparison (comparison based on
            # data and units!)
            if attr in attrs and attr in attrs:
                attrs.remove(attr)

        # if 'title' in attrs:  #    attrs.remove('title')
        # #TODO: should we use title for comparison?

    for attr in attrs:
        if attr != "units":
            sattr = getattr(this, f"_{attr}")
            if hasattr(other, f"_{attr}"):
                oattr = getattr(other, f"_{attr}")
                if sattr is None and oattr is not None:
                    raise AssertionError(f"`{attr}` of {this} is None.")
                if oattr is None and sattr is not None:
                    raise AssertionError(f"{attr} of {other} is None.")
                if (
                    hasattr(oattr, "size")
                    and hasattr(sattr, "size")
                    and oattr.size != sattr.size
                ):
                    # particular case of mask
                    if attr != "mask":
                        raise AssertionError(f"{thistype}.{attr} sizes are different.")
                    assert_array_equal(
                        other.mask,
                        this.mask,
                        f"{this} and {other} masks are different.",
                    )
                if attr in ["data"]:
                    # we must compare masked array
                    sattr = this.masked_data
                    oattr = other.masked_data
                    if approx:
                        assert_array_compare(
                            compare,
                            sattr,
                            oattr,
                            header=(
                                f"{thistype}.{attr} attributes ar"
                                f"e not almost equal to %d decimals" % decimal
                            ),
                            precision=decimal,
                        )
                    else:
                        assert_array_compare(
                            operator.__eq__,
                            sattr,
                            oattr,
                            header=f"{thistype}.{attr} attributes are not equal",
                        )

                elif attr in ["coordset"]:
                    if (sattr is None and oattr is not None) or (
                        oattr is None and sattr is not None
                    ):
                        raise AssertionError("One of the coordset is None")
                    if sattr is None and oattr is None:
                        pass
                    else:
                        for item in zip(sattr, oattr, strict=False):
                            # Pass coords_data_only as data_only parameter to compare_coords
                            res = compare_coords(
                                *item,
                                approx=True,
                                decimal=3,
                                data_only=coords_data_only,
                            )
                            if not res:
                                raise AssertionError(f"coords differs: \n{res}")
                else:
                    eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"The {attr} attributes of {this} and {other} are different.",
                    )
            else:
                return False
        else:
            # unitlesss and dimensionless are supposed equals
            sattr = this._units
            if sattr is None:
                sattr = ur.dimensionless
            if hasattr(other, "_units"):
                oattr = other._units
                if oattr is None:
                    oattr = ur.dimensionless

                eq &= np.all(sattr == oattr)
                if not eq:
                    raise AssertionError(
                        f"attributes `{attr}` are not equals or one is "
                        f"missing: \n{sattr} != {oattr}",
                    )
            else:
                raise AssertionError(f"{other} has no units")

    return True


def assert_dataset_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_dataset_almost_equal(nd1, nd2, **kwargs)
    return True


def assert_dataset_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 3)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if dataset is label only!.
    data_only = kwargs.get("data_only", False)
    coords_data_only = kwargs.get("coords_data_only", False)

    compare_datasets(
        nd1,
        nd2,
        approx=approx,
        decimal=decimal,
        data_only=data_only,
        coords_data_only=coords_data_only,
    )
    return True


def assert_coord_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_coord_almost_equal(nd1, nd2, **kwargs)
    return True


def assert_coord_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 6)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if coord is label only!.
    data_only = kwargs.get("data_only", False)
    compare_coords(nd1, nd2, approx=approx, decimal=decimal, data_only=data_only)
    return True


def assert_ndarray_equal(nd1, nd2, **kwargs):
    kwargs["approx"] = False
    assert_ndarray_almost_equal(nd1, nd2, **kwargs)
    return True


def assert_ndarray_almost_equal(nd1, nd2, **kwargs):
    decimal = kwargs.get("decimal", 6)
    approx = kwargs.get("approx", True)
    # if data_only is True, compare only based on data (not labels and so on)
    # except if ndarray is label only!.
    data_only = kwargs.get("data_only", False)
    compare_ndarrays(nd1, nd2, approx=approx, decimal=decimal, data_only=data_only)
    return True


def assert_project_equal(proj1, proj2, **kwargs):
    kwargs["approx"] = False

    # Check for project differences
    data_only = kwargs.get("dataset_data_only", False)

    if not data_only:
        # Recursively check the project hierarchy for differences
        # Collect all datasets from both projects, including those in nested projects
        def collect_all_datasets(project, path=""):
            datasets = []
            for i, ds in enumerate(project.datasets):
                datasets.append((f"{path}dataset[{i}]", ds))

            for i, subproj in enumerate(project.projects):
                subpath = f"{path}subproject[{i}]."
                datasets.extend(collect_all_datasets(subproj, subpath))

            return datasets

        # Collect all datasets from both projects
        proj1_datasets = collect_all_datasets(proj1)
        proj2_datasets = collect_all_datasets(proj2)

        # Check if number of datasets differs
        if len(proj1_datasets) != len(proj2_datasets):
            raise AssertionError(
                f"Projects have different number of total datasets: {len(proj1_datasets)} != {len(proj2_datasets)}"
            )

        # Check titles of all datasets
        for (path1, ds1), (path2, ds2) in zip(
            proj1_datasets, proj2_datasets, strict=False
        ):
            if (
                hasattr(ds1, "title")
                and hasattr(ds2, "title")
                and ds1.title != ds2.title
            ):
                raise AssertionError(
                    f"Dataset titles differ: {path1}.title = '{ds1.title}' != {path2}.title = '{ds2.title}'"
                )

        # Check scripts at any level
        def collect_all_scripts(project, path=""):
            scripts = []
            for i, sc in enumerate(project.scripts):
                scripts.append((f"{path}script[{i}]", sc))

            for i, subproj in enumerate(project.projects):
                subpath = f"{path}subproject[{i}]."
                scripts.extend(collect_all_scripts(subproj, subpath))

            return scripts

        # Collect all scripts from both projects
        proj1_scripts = collect_all_scripts(proj1)
        proj2_scripts = collect_all_scripts(proj2)

        # Check if number of scripts differs
        if len(proj1_scripts) != len(proj2_scripts):
            raise AssertionError(
                f"Projects have different number of total scripts: {len(proj1_scripts)} != {len(proj2_scripts)}"
            )

        # Compare scripts
        for (path1, sc1), (path2, sc2) in zip(
            proj1_scripts, proj2_scripts, strict=False
        ):
            if sc1 != sc2:
                raise AssertionError(f"Scripts differ: {path1} != {path2}")

    # Now continue with the more detailed comparison
    assert_project_almost_equal(proj1, proj2, **kwargs)
    return True


def assert_project_almost_equal(proj1, proj2, **kwargs):
    data_only = kwargs.get("dataset_data_only", False)

    # Handle different numbers of datasets, scripts, subprojects if data_only is True
    if not data_only:
        assert len(proj1.datasets) == len(proj2.datasets)  # noqa: S101
        assert len(proj1.projects) == len(proj2.projects)  # noqa: S101
        assert len(proj1.scripts) == len(proj2.scripts)  # noqa: S101

    # Pass dataset_data_only as data_only parameter to compare_datasets
    dataset_kwargs = kwargs.copy()  # Create a copy to avoid modifying the original

    # Compare datasets if they exist
    if len(proj1.datasets) > 0 and len(proj2.datasets) > 0:
        for nd1, nd2 in zip(proj1.datasets, proj2.datasets, strict=False):
            if data_only:
                dataset_kwargs["data_only"] = True
            compare_datasets(nd1, nd2, **dataset_kwargs)

    # Compare subprojects if they exist
    if len(proj1.projects) > 0 and len(proj2.projects) > 0:
        for pr1, pr2 in zip(proj1.projects, proj2.projects, strict=False):
            assert_project_almost_equal(pr1, pr2, **kwargs)

    # Compare scripts if they exist
    if len(proj1.scripts) > 0 and len(proj2.scripts) > 0:
        for sc1, sc2 in zip(proj1.scripts, proj2.scripts, strict=False):
            assert_script_equal(sc1, sc2, **kwargs)

    return True


def assert_script_equal(sc1, sc2, **kwargs):
    if sc1 != sc2:
        raise AssertionError(f"Scripts are different: {sc1.content} != {sc2.content}")
    return True


# ======================================================================================
# RandomSeedContext
# ======================================================================================
class RandomSeedContext:
    """
    Context manager to seed and restore the numpy RNG state.

    A context manager (for use with the `with` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    (Copied from Astropy, licence BSD-3)

    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with RandomSeedContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        from numpy import random

        random.set_state(self.startstate)


# ======================================================================================
# raises and assertions (mostly copied from astropy)
# ======================================================================================
def assert_units_equal(unit1, unit2, strict=False):
    """
    Compare units.

    Parameters
    ----------
    unit1 : units
        Units to be compared.
    unit2 : units
        Other units to be compared
    strict :  bool, optional, default: False
        If True, units should be exactly the same: `km` != `mm` .

    """
    from pint import DimensionalityError

    try:
        x = (1.0 * unit1).to_base_units() / (1.0 * unit2).to_base_units()
    except DimensionalityError:
        raise AssertionError from None

    if x.dimensionless and (x == 1.0 or not strict):
        _check_absorbance_related_units(unit1, unit2)
        return True
    raise AssertionError


def _check_absorbance_related_units(unit1, unit2):
    # particular case of absorbance, transmittance and absolute_transmitttance units
    lunit = ["absorbance", "transmittance", "absolute_transmittance"]
    if (
        f"{unit1: P}" in lunit
        and f"{unit2: P}" in lunit
        and f"{unit1: P}" != f"{unit2: P}"
    ):
        raise AssertionError
    return True


class raises:
    """
    Decorator to mark that a test should raise a given exception.

    Use as follows::

        @raises(ZeroDivisionError)
        def test_foo():
            x = 1/0

    This can also be used a context manager, in which case it is just
    an alias for the `pytest.raises` context manager (because the
    two have the same name this help avoid confusion by being
    flexible).

    (Copied from Astropy, licence BSD-3)

    """

    # pep-8 naming exception -- this is a decorator class
    def __init__(self, exc):
        self._exc = exc
        self._ctx = None

    def __call__(self, func):
        @functools.wraps(func)
        def run_raises_test(*args, **kwargs):
            import pytest

            pytest.raises(self._exc, func, *args, **kwargs)

        return run_raises_test

    def __enter__(self):
        import pytest

        self._ctx = pytest.raises(self._exc)
        return self._ctx.__enter__()

    def __exit__(self, *exc_info):
        return self._ctx.__exit__(*exc_info)


class catch_warnings(warnings.catch_warnings):
    """
    A high-powered version of warnings.catch_warnings.

    To use for testing and to make sure that there is no dependence on the order in which the tests are run.

    This completely blitzes any memory of any warnings that have appeared before so that all warnings will be caught and displayed.

    `*args` is a set of warning classes to collect.  If no arguments are provided, all warnings are collected.

    Use as follows::

        with catch_warnings(MyCustomWarning) as w :
            do.something.bad()
        assert len(w) > 0

    (Copied from Astropy, licence BSD-3)
    """

    def __init__(self, *classes):
        super().__init__(record=True)
        self.classes = classes

    def __enter__(self):
        warning_list = super().__enter__()
        if len(self.classes) == 0:
            warnings.simplefilter("always")
        else:
            warnings.simplefilter("ignore")
            for cls in self.classes:
                warnings.simplefilter("always", cls)
        return warning_list

    def __exit__(self, type, value, traceback):
        pass


# --------------------------------------------------------------------------------------
# Matplotlib testing utilities
# --------------------------------------------------------------------------------------
#
# figures_dir = os.path.join(os.path.expanduser("~"), ".spectrochempy",
# "figures")
# os.makedirs(figures_dir, exist_ok=True)
#
#
# def _compute_rms(x, y):
#     return calculate_rms(x, y)
#
#
# def _image_compare(imgpath1, imgpath2, REDO_ON_TYPEERROR):
#     # compare two images saved in files imgpath1 and imgpath2
#
#     from matplotlib.pyplot import imread
#     from skimage.measure import compare_ssim as ssim
#
#     # read image
#     try:
#         img1 = imread(imgpath1)
#     except IOError:
#         img1 = imread(imgpath1 + '.png')
#     try:
#         img2 = imread(imgpath2)
#     except IOError:
#         img2 = imread(imgpath2 + '.png')
#
#     try:
#         sim = ssim(img1, img2,
#                    data_range=img1.max() - img2.min(),
#                    multichannel=True) * 100.
#         rms = _compute_rms(img1, img2)
#     except ValueError:
#         rms = sim = -1
#
#     except TypeError as e:
#         # this happen sometimes and erratically during testing using
#         # pytest-xdist (parallele testing). This is work-around the problem
#         if e.args[0] == "unsupported operand type(s) " \
#                         "for - : 'PngImageFile' and 'int'" and not
#                         REDO_ON_TYPEERROR:
#             REDO_ON_TYPEERROR = True
#             rms = sim = -1
#         else:
#             raise
#
#     return (sim, rms, REDO_ON_TYPEERROR)
#
#
# def compare_images(imgpath1, imgpath2,
#                    max_rms=None,
#                    min_similarity=None, ):
#     sim, rms, _ = _image_compare(imgpath1, imgpath2, False)
#
#     EPSILON = np.finfo(float).eps
#     CHECKSIM = (min_similarity is not None)
#     SIM = min_similarity if CHECKSIM else 100. - EPSILON
#     MESSSIM = "(similarity : {:.2f}%)".format(sim)
#     CHECKRMS = (max_rms is not None and not CHECKSIM)
#     RMS = max_rms if CHECKRMS else EPSILON
#     MESSRMS = "(rms : {:.2f})".format(rms)
#
#     if sim < 0 or rms < 0:
#         message = "Sizes of the images are different"
#     elif CHECKRMS and rms <= RMS:
#         message = "identical images {}".format(MESSRMS)
#     elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
#         message = "identical/similar images {}".format(MESSSIM)
#     else:
#         message = "different images {}".format(MESSSIM)
#
#     message += "\n\t reference : {}".format(
#         os.path.basename(referfile))
#     message += "\n\t generated : {}\n".format(
#         tmpfile)
#     if not message.startswith("identical"):
#         errors += message
#
#     return message
#
#
# def same_images(imgpath1, imgpath2):
#     if compare_images(imgpath1, imgpath2).startswith('identical'):
#         return True
#
#     return False
#
#
# def image_comparison(reference=None,
#                      extension=None,
#                      max_rms=None,
#                      min_similarity=None,
#                      force_creation=False,
#                      savedpi=150):
#     """
#     image file comparison decorator.
#
#     Performs a comparison of the images generated by the decorated function.
#     If none of min_similarity and max_rms if set,
#     automatic similarity check is done :
#
#     Parameters
#     ----------
#     reference : list of image filename for the references
#
#         List the image filenames of the reference figures
#         (located in ` .spectrochempy/figures` ) which correspond in
#         the same order to
#         the various figures created in the decorated function.
#         If these files doesn't exist an error is generated, except if the
#         force_creation argument is True. This should allow the creation
#         of a reference figures, the first time the corresponding figures are
#         created.
#
#     extension : str, optional, default=`png`
#
#         Extension to be used to save figure, among
#         (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)
#
#     force_creation : `bool` , optional, default=`False` .
#         If set, then it will be used to decide if an image is the same (
#         or not. In this case max_rms is not used.
#         if this flag is True, the figures created in the decorated
#         function are
#         saved in the reference figures directory (
#         ` .spectrocchempy/figures` )
#
#     min_similarity : float (percent).
#
#         If set, then it will be used to decide if an image is the same
#         (more than the acceptable similarity)
#
#     max_rms : float
#
#         rms stands for `Root Mean Square` . If set, then it will
#         be used to decide if an image is the same
#         (less than the acceptable rms). Not used if min_similarity also set.
#
#     savedpi : int, optional, default=150
#
#         dot per inch of the generated figures
#     """
#     from spectrochempy.utils import is_sequence
#
#     if not reference:
#         raise ValueError('no reference image provided. Stopped')
#
#     if not extension:
#         extension = 'png'
#
#     if not is_sequence(reference):
#         reference = list(reference)
#
#     def make_image_comparison(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             # check the existence of the file if force creation is False
#             for ref in reference:
#                 filename = os.path.join(figures_dir,
#                                         '{}.{}'.format(ref, extension))
#                 if not os.path.exists(filename) and not force_creation:
#                     raise ValueError(
#                             'One or more reference file do not exist.\n'
#                             'Creation can be forced from the generated '
#                             'figure, by setting force_creation flag to True')
#
#             # get the nums of the already existing figures
#             # that, obviously,should not considered in
#             # this comparison
#             fignums = plt.get_fignums()
#
#             # execute the function generating the figures
#             # rest style to basic 'lcs' style
#             _ = func(*args, **kwargs)
#
#             # get the new fignums if any
#             curfignums = plt.get_fignums()
#             for x in fignums:
#                 # remove not newly created
#                 curfignums.remove(x)
#
#             if not curfignums:
#                 # no figure where generated
#                 raise RuntimeError(f'No figure was generated by the "{
#                 func.__name__}" function. Stopped')
#
#             if len(reference) != len(curfignums):
#                 raise ValueError("number of reference figures provided
#                 doesn't match the number of generated
#                 figures.")
#
#             # Comparison
#             REDO_ON_TYPEERROR = False
#
#             while True:
#                 errors = ""
#                 for fignum, ref in zip(curfignums, reference):
#                     referfile = os.path.join(figures_dir,
#                                              '{}.{}'.format(ref, extension))
#
#                     fig = plt.figure(fignum)
#                     # work around to set
#                     # the correct style: we
#                     # we have saved the rcParams
#                     # in the figure attributes
#                     plt.rcParams.update(fig.rcParams)
#
#                     if force_creation:
#                         # make the figure for reference and bypass
#                         # the rest of the test
#                         tmpfile = referfile
#                     else:
#                         # else we create a temporary file to save the figure
#                         fd, tmpfile = tempfile.mkstemp(
#                                 prefix='temp{}-'.format(fignum),
#                                 suffix='.{}'.format(extension), text=True)
#                         os.close(fd)
#
#                     fig.savefig(tmpfile, dpi=savedpi)
#                     sim, rms = 100.0, 0.0
#                     if not force_creation:
#                         # we do not need to loose time
#                         # if we have just created the figure
#                         sim, rms, REDO_ON_TYPEERROR = _image_compare(
#                             referfile,
#                             tmpfile,
#                             REDO_ON_TYPEERROR)
#
#                     EPSILON = np.finfo(float).eps
#                     CHECKSIM = (min_similarity is not None)
#                     SIM = min_similarity if CHECKSIM else 100. - EPSILON
#                     MESSSIM = "(similarity : {:.2f}%)".format(sim)
#                     CHECKRMS = (max_rms is not None and not CHECKSIM)
#                     RMS = max_rms if CHECKRMS else EPSILON
#                     MESSRMS = "(rms : {:.2f})".format(rms)
#
#                     if sim < 0 or rms < 0:
#                         message = "Sizes of the images are different"
#                     elif CHECKRMS and rms <= RMS:
#                         message = "identical images {}".format(MESSRMS)
#                     elif (CHECKSIM or not CHECKRMS) and sim >= SIM:
#                         message = "identical/similar images {}".format(
#                             MESSSIM)
#                     else:
#                         message = "different images {}".format(MESSSIM)
#
#                     message += "\n\t reference : {}".format(
#                         os.path.basename(referfile))
#                     message += "\n\t generated : {}\n".format(
#                         tmpfile)
#                     if not message.startswith("identical"):
#                         errors += message
#
#                 if errors and not REDO_ON_TYPEERROR:
#                     # raise an error if one of the image is different from
#                     # reference image
#                     raise ImageAssertionError("\n" + errors)
#
#                 if not REDO_ON_TYPEERROR:
#                     break
#
#             return
#
#         return wrapper
#
#     return make_image_comparison
