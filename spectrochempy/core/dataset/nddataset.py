# -*- coding: utf-8 -*-
#
# ======================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT.
#  See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the |NDDataset| class.
"""


__all__ = ["NDDataset"]

import sys
import textwrap
from datetime import datetime, tzinfo

import numpy as np
import pytz
import traitlets as tr
from traittypes import Array

from spectrochempy.core import error_, warning_
from spectrochempy.core.common.constants import DEFAULT_DIM_NAME, MaskedConstant
from spectrochempy.core.common.docstrings import _docstring
from spectrochempy.core.common.exceptions import (
    MissingCoordinatesError,
    UnknownTimeZoneError,
)
from spectrochempy.core.common.compare import is_datetime64
from spectrochempy.core.common.print import colored_output
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndmaskedcomplexarray import (
    NDMaskedComplexArray,
)
from spectrochempy.core.dataset.mixins.iomixin import NDIO

# from spectrochempy.core.dataset.ndmath import (
#     NDManipulation,
#     NDMath,
#     _set_operators,
#     _set_ufuncs,
# )
from spectrochempy.core.dataset.mixins.plotmixin import NDPlot
from spectrochempy.core.project.baseproject import AbstractProject
from spectrochempy.utils.optional import import_optional_dependency
from spectrochempy.utils.system import get_user_and_node
from spectrochempy.core.units import Unit, encode_quantity

# ======================================================================================
# NDDataset class definition
# ======================================================================================
class NDDataset(NDMaskedComplexArray):  # NDIO, NDPlot, NDManipulation, NDMath,
    __doc__ = _docstring.dedent(
        """
    The main N-dimensional dataset class used by |scpy|.

    The NDDataset is the main object used by SpectroChemPy. Like numpy
    ndarrays, NDDataset have the capability to be
    sliced, sorted and subject to mathematical operations. But, in addition,
    NDDataset may have units,
    can be masked
    and each dimensions can have coordinates also with units. This make
    NDDataset aware of unit compatibility,
    e.g.,
    for binary operation such as additions or subtraction or during the
    application of mathematical operations.
    In addition or in replacement of numerical data for coordinates,
    NDDataset can also have labeled coordinates
    where labels can be different kind of objects (strings, datetime,
    numpy nd.ndarray or other NDDatasets, etc…).

    Parameters
    ----------
    data : array of floats
        Data array contained in the object. The data can be a list, a tuple,
        a |ndarray|, a ndarray-like,
        a |NDArray| or any subclass of |NDArray|. Any size or shape of data
        is accepted. If not given, an empty
        |NDArray| will be inited.
        At the initialisation the provided data will be eventually casted to
        a numpy-ndarray.
        If a subclass of |NDArray| is passed which already contains some
        mask, labels, or units, these elements
        will
        be used to accordingly set those of the created object. If possible,
        the provided data will not be copied
        for `data` input, but will be passed by reference, so you should
        make a copy of the `data` before passing
        them if that's the desired behavior or set the `copy` argument to True.
    %(kwargs)s

    Other Parameters
    ----------------
    coordset : An instance of |CoordSet|, optional
        `coords` contains the coordinates for the different dimensions of
        the `data`. if `coords` is provided,
        it must be specified the `coord` and `labels` for all dimensions of the
        `data`.
        Multiple `coord`'s can be specified in an |CoordSet| instance for
        each dimension.
    coordunits : list, optional
        A list of units corresponding to the dimensions in the order of the
        coordset.
    coordtitles : list, optional
        A list of titles corresponding of the dimensions in the order of the
        coordset.
    author : str, optional
        Name(s) of the author(s) of this dataset. BNy default, name of the
        computer note where this dataset is
        created.
    description : str, optional
        An optional description of the nd-dataset.
    origin : str, optional
        Origin of the data: Name of organization, address, telephone number,
        name of individual contributor, etc., as appropriate.
    history : str, optional
        A string to eventually add to the object history.
    roi : list
        Region of interest (ROI) limits.
    timezone : datetime.tzinfo, optional
        The timezone where the data were created. If not specified, the local timezone
        is assumed.
    meta : dict-like object, optional
        Additional metadata for this object. Must be dict-like but no
        further restriction is placed on meta.
    %(NDMaskedComplexArray.other_parameters)s

    See Also
    --------
    Coord : Explicit coordinates object.
    CoordSet : Set of coordinates.

    Notes
    -----
    The underlying array in a |NDDataset| object can be accessed through the
    `data` attribute, which will return a conventional |ndarray|.
    """
    )

    # Examples
    # --------
    # Usage by an end-user
    #
    # >>> x = scp.NDDataset([1, 2, 3])
    # >>> print(x.data)  # doctest: +NORMALIZE_WHITESPACE
    # [       1        2        3.]
    # """

    # Dates
    _acquisition_date = tr.Instance(datetime, allow_none=True)
    _created = tr.Instance(datetime)
    _modified = tr.Instance(datetime)
    _timezone = tr.Instance(tzinfo, allow_none=True)

    # Metadata
    _author = tr.Unicode()
    _description = tr.Unicode()
    _origin = tr.Unicode()
    _history = tr.List(tr.Tuple(), allow_none=True)
    _meta = tr.Instance(Meta, allow_none=True)

    # coordinates
    _coordset = tr.Instance(CoordSet, allow_none=True)

    # model data (e.g., for fit)
    _modeldata = Array(tr.Float(), allow_none=True)

    # some setting for NDDataset
    _copy = tr.Bool(False)

    # dataset can be members of a project.
    # we use the abstract class to avoid circular imports.
    _parent = tr.Instance(AbstractProject, allow_none=True)

    # For the GUI interface

    # parameters state
    _state = tr.Dict()

    # processed data (for GUI)
    _processeddata = Array(tr.Float(), allow_none=True)

    # processed mask (for GUI)
    _processedmask = tr.Union(
        (tr.Bool(), Array(tr.Bool()), tr.Instance(MaskedConstant))
    )

    # baseline data (for GUI)
    _baselinedata = Array(tr.Float(), allow_none=True)

    # reference data (for GUI)
    _referencedata = Array(tr.Float(), allow_none=True)

    # region ranges
    _ranges = tr.Instance(Meta)

    # ------------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------------
    def __init__(self, data=None, **kwargs):

        super().__init__(data, **kwargs)

        self._created = datetime.utcnow()
        self.description = kwargs.pop("description", "")
        self.author = kwargs.pop("author", get_user_and_node())

        history = kwargs.pop("history", None)
        if history is not None:
            self.history = history

        self._parent = None

        # eventually set the coordinates with optional units and title
        coordset = kwargs.pop("coordset", None)

        if isinstance(coordset, CoordSet):
            self.set_coordset(**coordset)

        else:
            if coordset is None:
                coordset = [None] * self.ndim

            coordunits = kwargs.pop("coordunits", None)
            if coordunits is None:
                coordunits = [None] * self.ndim

            coordtitles = kwargs.pop("coordtitles", None)
            if coordtitles is None:
                coordtitles = [None] * self.ndim

            _coordset = []
            for c, u, t in zip(coordset, coordunits, coordtitles):
                if not isinstance(c, CoordSet):
                    coord = Coord(c)
                    if u is not None:
                        coord.units = u
                    if t is not None:
                        coord.title = t
                else:
                    if u:  # pragma: no cover
                        warning_(
                            "units have been set for a CoordSet, but this will be ignored "
                            "(units are only defined at the coordinate level"
                        )
                    if t:  # pragma: no cover
                        warning_(
                            "title will be ignored as they are only defined at the coordinates level"
                        )
                    coord = c

                _coordset.append(coord)

            if _coordset and set(_coordset) != {
                Coord()
            }:  # if they are no coordinates do nothing
                self.set_coordset(*_coordset)

        self._modified = self._created

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __eq__(self, other, attrs=None):
        attrs = self._attributes()
        for attr in (
            "filename",
            "preferences",
            "name",
            "description",
            "history",
            "created",
            "modified",
            "origin",
            "show_datapoints",
            "roi",
            "modeldata",
            "processeddata",
            "baselinedata",
            "referencedata",
            "state",
        ):
            # these attributes are not used for comparison (comparison based on data
            # and units!)
            try:
                attrs.remove(attr)
            except ValueError:
                pass

        return super().__eq__(other, attrs)

    def __getattr__(self, item):
        # when the attribute was not found
        if (
            item
            in [
                "__numpy_ufunc__",
                "interface",
                "_pytestfixturefunction",
                "__dataclass_fields__",
                "_ipython_canary_method_should_not_exist_",
                "_baseclass",
                "_fill_value",
                "_ax_lines",
                "_axcb",
                "clevels",
                "__wrapped__",
                "coords",
                "__await__",
                "__aiter__",
            ]
            or "_validate" in item
            or "_changed" in item
        ):
            # raise an error so that traits, ipython operation and more ... will be handled correctly
            raise AttributeError

        # syntax such as ds.x, ds.y, etc...

        if item[0] in self.dims or self._coordset:

            # look also properties
            attribute = None
            index = 0

            if len(item) > 2 and item[1] == "_":
                attribute = item[1:]
                item = item[0]
                index = self.dims.index(item)

            if self._coordset:
                try:
                    c = self._coordset[item]
                    if isinstance(c, str) and c in self.dims:
                        # probably a reference to another coordinate name
                        c = self._coordset[c]

                    if c.name in self.dims or c._parent_dim in self.dims:
                        if attribute is not None:
                            # get the attribute
                            return getattr(c, attribute)
                        return c
                    else:
                        raise AttributeError

                except Exception as err:
                    if item in self.dims:
                        return None

                    if item in self.meta.keys():  # try to find a metadata
                        return self.meta[item]
                    raise err

            elif attribute is not None:
                if attribute == "size":
                    # we want the size but there is no coords, get it from the data shape
                    return self.shape[index]
                else:
                    raise AttributeError(
                        f"Can not find `{attribute}` when no coordinate is defined"
                    )

            return None

        raise AttributeError

    def __getitem__(self, items, **kwargs):

        saveditems = items

        # coordinate selection to test first
        if isinstance(items, str):
            try:
                return self._coordset[items]
            except Exception:
                pass

        # slicing
        new, items = super().__getitem__(items, return_index=True)

        if new is None:
            return None

        if self._coordset is not None:
            names = self._coordset.names  # all names of the current coordinates
            new_coords = [None] * len(names)
            for i, item in enumerate(items):
                # get the corresponding dimension name in the dims list
                name = self.dims[i]
                # get the corresponding index in the coordinate's names list
                idx = names.index(name)
                if self._coordset[idx].is_empty:
                    new_coords[idx] = Coord(None, name=name)
                elif isinstance(item, slice):
                    # add the slice on the corresponding coordinates on the dim
                    # to the new list of coordinates
                    if not isinstance(self._coordset[idx], CoordSet):
                        new_coords[idx] = self._coordset[idx][item]
                    else:
                        # we must slice all internal coordinates
                        newc = []
                        for c in self._coordset[idx]:
                            newc.append(c[item])
                        new_coords[idx] = CoordSet(*newc[::-1], name=name)
                        # we reverse to be sure
                        # the order will be  kept for internal coordinates
                        new_coords[idx]._default = self._coordset[
                            idx
                        ]._default  # set the same default coord
                        new_coords[idx]._is_same_dim = self._coordset[idx]._is_same_dim

                elif isinstance(item, (np.ndarray, list)):
                    new_coords[idx] = self._coordset[idx][item]

            new.set_coordset(*new_coords, keepnames=True)

        new.history = f"Slice extracted: ({saveditems})"
        return new

    def __hash__(self):
        # all instance of this class has same hash, so they can be compared
        return super().__hash__ + hash(self._coordset)

    def __setattr__(self, key, value):

        if key in DEFAULT_DIM_NAME:  # syntax such as ds.x, ds.y, etc...
            # Note the above test is important to avoid errors with traitlets
            # even if it looks redundant with the following
            if key in self.dims:
                if self._coordset is None:
                    # we need to create a coordset first
                    self.set_coordset(
                        dict((self.dims[i], None) for i in range(self.ndim))
                    )
                idx = self._coordset.names.index(key)
                _coordset = self._coordset
                listcoord = False
                if isinstance(value, list):
                    listcoord = all([isinstance(item, Coord) for item in value])
                if listcoord:
                    _coordset[idx] = list(CoordSet(value).to_dict().values())[0]
                    _coordset[idx].name = key
                    _coordset[idx]._is_same_dim = True
                elif isinstance(value, CoordSet):
                    if len(value) > 1:
                        value = CoordSet(value)
                    _coordset[idx] = list(value.to_dict().values())[0]
                    _coordset[idx].name = key
                    _coordset[idx]._is_same_dim = True
                elif isinstance(value, Coord):
                    value.name = key
                    _coordset[idx] = value
                else:
                    _coordset[idx] = Coord(value, name=key)
                _coordset = self._valid_coordset(_coordset)
                self._coordset.set(_coordset)
            else:
                raise AttributeError(f"Coordinate `{key}` is not used.")
        else:
            super().__setattr__(key, value)

    # ----------------------------------------------------------------------------------
    # Private methods and properties
    # ----------------------------------------------------------------------------------

    @tr.default("_baselinedata")
    def __baselinedata_default(self):
        return None

    @tr.default("_coordset")
    def __coordset_default(self):
        return None

    @tr.validate("_coordset")
    def __coordset_validate(self, proposal):
        coords = proposal["value"]
        return self._valid_coordset(coords)

    @tr.validate("_created")
    def __created_validate(self, proposal):
        date = proposal["value"]
        if date.tzinfo is not None:
            # make the date utc naive
            date = date.replace(tzinfo=None)
        return date

    @tr.validate("_history")
    def __history_validate(self, proposal):
        history = proposal["value"]
        if isinstance(history, list) or history is None:
            # reset
            self._history = None
        return history

    @tr.default("_meta")
    def __meta_default(self):
        return Meta()

    @tr.default("_modeldata")
    def __modeldata_default(self):
        return None

    @tr.default("_processeddata")
    def __processeddata_default(self):
        return None

    @tr.default("_ranges")
    def __ranges_default(self):
        ranges = Meta()
        for dim in self.dims:
            ranges[dim] = dict(masks={}, baselines={}, integrals={}, others={})
        return ranges

    @tr.default("_referencedata")
    def __referencedata_default(self):
        return None

    @tr.default("_timezone")
    def __timezone_default(self):
        # Return the default timezone (UTC)
        return datetime.utcnow().astimezone().tzinfo

    @tr.validate("_modified")
    def __modified_validate(self, proposal):
        date = proposal["value"]
        if date.tzinfo is not None:
            # make the date utc naive
            date = date.replace(tzinfo=None)
        return date

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):

        # ex: change {
        #   'owner': object, # The HasTraits instance
        #   'new': 6, # The new value
        #   'old': 5, # The old value
        #   'name': "foo", # The name of the changed trait
        #   'type': 'change', # The event type of the notification, usually 'change'
        # }

        if change["name"] in ["_created", "_modified", "trait_added"]:
            return

        # all the time -> update modified date
        self._modified = datetime.utcnow()
        return

    def _attributes(self, removed=None, added=None):
        # Only these attributes are used for saving dataset
        # WARNING: be careful to keep the present order of the three first elements!
        # Needed for save/load operations
        return [
            # Keep the following order
            "dims",
            "coordset",
            "data",
            # From here it is free
            "name",
            "title",
            "mask",
            "units",
            "meta",
            "author",
            "description",
            "history",
            "created",
            "modified",
            "acquisition_date",
            "origin",
            "roi",
            "ranges",
            "modeldata",
            "referencedata",
            "state",
            "ranges",
        ]  # + NDIO()._attributes() + NDPLOT._attributes()

    def _cstr(self):
        # Display the metadata of the object and partially the data
        out = ""
        out += "         name: {}\n".format(self.name)
        out += "       author: {}\n".format(self.author)
        out += "      created: {}\n".format(self.created)

        wrapper1 = textwrap.TextWrapper(
            initial_indent="",
            subsequent_indent=" " * 15,
            replace_whitespace=True,
            width=self._text_width,
        )

        pars = self.description.strip().splitlines()
        if pars:
            out += "  description: "
            desc = ""
            if pars:
                ppp = pars[0]
                desc += "{}\n".format(wrapper1.fill(ppp))
            for par in pars[1:]:
                desc += "{}\n".format(textwrap.indent(par, " " * 15))
            # the three escaped null characters are here to facilitate
            # the generation of html outputs
            desc = "\0\0\0{}\0\0\0\n".format(desc.rstrip())
            out += desc

        if self._history:
            pars = self.history
            out += "      history: "
            hist = ""
            if pars:
                ppp = pars[0]
                if len(ppp) > self._text_width:
                    ppp = ppp[: min(self._text_width - 4, len(pars[0]))] + " ..."
                hist += "{}\n".format(wrapper1.fill(ppp))
            for par in pars[1:]:
                hist += "{}\n".format(textwrap.indent(par, " " * 15))
            # the three escaped null characters are here to facilitate
            # the generation of html outputs
            hist = "\0\0\0{}\0\0\0\n".format(hist.rstrip())
            out += hist

        out += "{}\n".format(self._str_value().rstrip())
        out += "{}\n".format(self._str_shape().rstrip()) if self._str_shape() else ""
        out += "{}\n".format(self._str_dims().rstrip())

        if not out.endswith("\n"):
            out += "\n"
        out += "\n"

        if not self._html_output:
            return colored_output(out.rstrip())
        else:
            return out.rstrip()

    @property
    def _dict_dims(self):
        _dict = {}
        for index, dim in enumerate(self.dims):
            if dim not in _dict:
                _dict[dim] = {"size": self.shape[index], "coord": getattr(self, dim)}
        return _dict

    def _dims_update(self, change=None):
        # when notified that a coords names have been updated
        _ = self.dims  # fire an update

    def _loc2index(self, loc, dim=-1, *, units=None):
        # Return the index of a location (label or coordinates) along the dim
        # This can work only if `coords` exists.

        if self._coordset is None:
            raise MissingCoordinatesError(
                "No coords have been defined. Slicing or selection"
                " by location ({}) needs coords definition.".format(loc)
            )

        coord = self.coord(dim)

        return coord.loc2index(loc, units=units)

    def _str_dims(self):
        if self.is_empty:
            return ""
        if len(self.dims) < 1 or not hasattr(self, "_coordset"):
            return ""
        if not self._coordset or len(self._coordset) < 1:
            return ""

        self._coordset._html_output = (
            self._html_output
        )  # transfer the html flag if necessary: false by default

        txt = self._coordset._cstr()
        txt = txt.rstrip()  # remove the trailing '\n'
        return txt

    _repr_dims = _str_dims

    def _valid_coordset(self, coordset):
        # uses in coordset_validate and setattr
        if coordset is None:
            return

        for k, coord in enumerate(coordset):

            if (
                coord is not None
                and not isinstance(coord, CoordSet)
                and coord.data is None
            ):
                continue

            # For coord to be acceptable, we require at least a NDArray,
            # a NDArray subclass or a CoordSet
            if not isinstance(coord, (Coord, CoordSet)):
                if isinstance(coord, NDArray):
                    coord = coordset[k] = Coord(coord)
                else:
                    raise TypeError(
                        "Coordinates must be an instance or a subclass of Coord class "
                        "or NDArray, or of CoordSet class, but an instance of "
                        "{type(coord)} has been passed"
                    )

            if self.dims and coord.name in self.dims:
                # check the validity of the given coordinates in terms of size (if it correspond to one of the dims)
                size = coord.size

                if self._implements("NDDataset"):
                    idx = self._get_dims_index(coord.name)[0]  # idx in self.dims
                    if size != self._data.shape[idx]:
                        raise ValueError(
                            f"the size of a coordinates array must be None or be equal"
                            f" to that of the respective `{coord.name}`"
                            f" data dimension but coordinate size={size} != data shape[{idx}]="
                            f"{self._data.shape[idx]}"
                        )
                else:
                    pass  # bypass this checking for any other derived type (should be
                    # done in the subclass)

        coordset._parent = self
        return coordset

    # ------------------------------------------------------------------------
    # DASH GUI options  (Work in Progress - not used for now)
    # ------------------------------------------------------------------------
    #
    # @property
    # def state(self):
    #     """
    #     State of the controller window for this dataset.
    #     """
    #     return self._state
    #
    # @state.setter
    # def state(self, val):
    #     self._state = val
    #
    # @property
    # def processeddata(self):
    #     """
    #     Data after processing (optionaly used).
    #     """
    #     return self._processeddata
    #
    # @processeddata.setter
    # def processeddata(self, val):
    #     self._processeddata = val
    #
    # @property
    # def processedmask(self):
    #     """
    #     Mask for the optional processed data.
    #     """
    #     return self._processedmask
    #
    # @processedmask.setter
    # def processedmask(self, val):
    #     self._processedmask = val
    #
    # @property
    # def baselinedata(self):
    #     """
    #     Data for an optional baseline.
    #     """
    #     return self._baselinedata
    #
    # @baselinedata.setter
    # def baselinedata(self, val):
    #     self._baselinedata = val
    #
    # @property
    # def referencedata(self):
    #     """
    #     Data for an optional reference spectra.
    #     """
    #     return self._referencedata
    #
    # @referencedata.setter
    # def referencedata(self, val):
    #     self._referencedata = val
    #
    # @property
    # def ranges(self):
    #     return self._ranges
    #
    # @ranges.setter
    # def ranges(self, value):
    #     self._ranges = value

    # ------------------------------------------------------------------------
    # Public methods and property
    # ------------------------------------------------------------------------

    @property
    def acquisition_date(self):
        """
        Acquisition date (Datetime).

        The acquisition date can be assigned by the user. In this case this date
        is returned.
        But if it is not the case, and if there is one datetime axis in the dataset
        coordinate, this method return the first datetime, which is then considered
        as the acquisition date. This assume that there is only one datetime axis in
        the dataset coordinates. If there is more than one, the first found in the
        coordset is used.
        """

        def get_acq(cs):
            for c in cs:
                if isinstance(c, Coord) and is_datetime64(c):
                    return c._acquisition_date
                if isinstance(c, CoordSet):
                    return get_acq(c)

        if self._acquisition_date is not None:
            # take the one which has been previously set fotr this dataset
            acq = self._acquisition_date
        else:
            # try to get one datetpme axis to determine it
            acq = get_acq(self.coordset)
        if acq is not None:
            if is_datetime64(acq):
                acq = datetime.fromisoformat(str(acq).split(".")[0])
            acq = pytz.utc.localize(acq)
            return acq.astimezone(self.timezone).isoformat(sep=" ", timespec="seconds")

    @property
    def acquisition_date(self):
        """
        Acquisition date (Datetime).

        If there is one datetime axis in the dataset coordinate, this method return
        the first datetimme, which is then considered as the acquisition date. Tjhis
        assume that there is only one datetime axis in the dataset coordinates.
        """

        def get_acq(cs):
            for c in cs:
                if isinstance(c, Coord) and is_datetime64(c):
                    return c.acquisition_date
                if isinstance(c, CoordSet):
                    return get_acq(c)

        return get_acq(self.coordset)

    def add_coordset(self, *coords, dims=None, **kwargs):
        """
        Add one or a set of coordinates from a dataset.

        Parameters
        ----------
        *coords : iterable
            Coordinates object(s).
        dims : list
            Name of the coordinates.
        **kwargs
            Optional keyword parameters passed to the coordset.
        """
        if not coords and not kwargs:
            # reset coordinates
            self._coordset = None
            return

        if self._coordset is None:
            # make the whole coordset at once
            self._coordset = CoordSet(*coords, dims=dims, **kwargs)
        else:
            # add one coordinate
            self._coordset._append(*coords, **kwargs)

        if self._coordset:
            # set a notifier to the updated traits of the CoordSet instance
            tr.HasTraits.observe(self._coordset, self._dims_update, "_updated")
            # force it one time after this initialization
            self._coordset._updated = True

    @property
    def author(self):
        """
        Creator of the dataset (str).
        """
        return self._author

    @author.setter
    def author(self, value):
        self._author = value

    def coord(self, dim="x"):
        """
        Return the coordinates along the given dimension.

        Parameters
        ----------
        dim : int or str
            A dimension index or name, default index = `x`.
            If an integer is provided, it is equivalent to the `axis` parameter for numpy array.

        Returns
        -------
        |Coord|
            Coordinates along the given axis.
        """
        idx = self._get_dims_index(dim)[0]  # should generate an error if the
        # dimension name is not recognized
        if idx is None:
            return None

        if self._coordset is None:
            return None

        # idx is not necessarily the position of the coordinates in the CoordSet
        # indeed, transposition may have taken place. So we need to retrieve the coordinates by its name
        name = self.dims[idx]
        if name in self._coordset.names:
            idx = self._coordset.names.index(name)
            return self._coordset[idx]
        else:
            error_(f"could not find this dimenson name: `{name}`")
            return None

    @property
    def coordnames(self):
        """
        tr.List of the |Coord| names.

        Read only property.
        """
        if self._coordset is not None:
            return self._coordset.names

    @property
    def coordset(self):
        """
        |CoordSet| instance.

        Contains the coordinates of the various dimensions of the dataset.
        It's a readonly property. Use set_coords to change one or more coordinates at once.
        """
        if self._coordset and all(c.is_empty for c in self._coordset):
            # all coordinates are empty, this is equivalent to None for the coordset
            return None
        return self._coordset

    @coordset.setter
    def coordset(self, coords):
        if isinstance(coords, CoordSet):
            self.set_coordset(**coords)
        else:
            self.set_coordset(coords)

    @property
    def coordtitles(self):
        """
        List of the |Coord| titles.

        Read only property.
        """
        if self._coordset is not None:
            return self._coordset.titles

    @property
    def coordunits(self):
        """
        List of the |Coord| units.

        Read only property. Use set_coordunits to eventually set units.
        """
        if self._coordset is not None:
            return self._coordset.units

    @property
    def created(self):
        """
        Creation date object (Datetime).
        """
        created = pytz.utc.localize(self._created)
        return created.astimezone(self.timezone).isoformat(sep=" ", timespec="seconds")

    @property
    def data(self):
        """
        The ``data`` array.

        If there is no data but labels, then the labels are returned instead of data.
        """
        return super().data

    @data.setter
    def data(self, data):
        # as we can't write super().data = data, we call _set_data
        # see comment in the data.setter of NDArray
        super()._set_data(data)

    def delete_coordset(self):
        """
        Delete all coordinate settings.
        """
        self._coordset = None

    @property
    def description(self):
        """Comment or description of the current object"""
        return self._description

    comment = description
    comment.__doc__ = "Alias for description"
    notes = description
    notes.__doc__ = "Alias for description"

    @description.setter
    def description(self, value):
        self._description = value

    @classmethod
    def from_xarray(cls, xarr):

        exclude = [
            "data",
            "coordset",
            "mask",
            "labels",
            "meta",
            "preferences",
            "transposed",
            "referencedata",
            "state",
            "ranges",
            "modeldata",
            "modified",
            "linear",
        ]

        def _from_xarray(klass, obj):

            if not (obj.attrs.get("linear", 0) == 1):  # bool are stored as int.
                new = klass()
                new.data = obj.data
            else:
                if klass == Coord:
                    new = Coord()
                else:
                    new = klass()

            # set attributes
            for item in new._attributes():
                if item in exclude:
                    continue
                try:
                    if item == "units":
                        if (
                            hasattr(obj, "pint_units")
                            and getattr(obj, "pint_units") != "None"
                        ):
                            setattr(new, "_units", Unit(getattr(obj, "pint_units")))
                    elif hasattr(obj, item):
                        setattr(new, item, getattr(obj, item))
                    elif obj.attrs.get(item, None) is not None:
                        setattr(new, item, obj.attrs.get(item))
                    else:
                        pass
                except Exception as e:
                    print(item)
                    error_(e)

            return new

        new = _from_xarray(cls, xarr)

        # dimensions and coord
        new.dims = xarr.coords.dims
        coordset = {}
        for dim in new.dims:
            coord = xarr.coords[dim]
            coordset[dim] = _from_xarray(Coord, coord)

        new.set_coordset(coordset)

        return new

    @property
    def history(self):
        """
        Describes the history of actions made on this array (tr.List of strings).
        """

        history = []
        for date, value in self._history:
            date = pytz.utc.localize(date)
            date = date.astimezone(self.timezone).isoformat(sep=" ", timespec="seconds")
            value = value[0].capitalize() + value[1:]
            history.append(f"{date}> {value}")
        return history

    @history.setter
    def history(self, value):
        if value is None:
            return
        if isinstance(value, list):
            # history will be replaced
            self._history = []
            if len(value) == 0:
                return
            value = value[0]
        date = datetime.utcnow()
        self._history.append((date, value))

    @property
    def meta(self):
        """
        Return an additional metadata dictionary.
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        if meta is not None:
            self._meta.update(meta)

    @property
    def modeldata(self):
        """
        |ndarray| - models data.

        Data eventually generated by modelling of the data.
        """
        return self._modeldata

    @modeldata.setter
    def modeldata(self, data):
        self._modeldata = data

    @property
    def modified(self):
        """
        Date of modification (readonly property).
        """
        modified = pytz.utc.localize(self._modified)
        return modified.astimezone(self.timezone).isoformat(sep=" ", timespec="seconds")

    @property
    def parent(self):
        """
        |Project| instance.

        The parent project of the dataset.
        """
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this dataset but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will first remove it
            # from the current project.
            self._parent.remove_dataset(self.name)
        self._parent = value

    def set_coordset(self, *args, **kwargs):
        """
        Set one or more coordinates at once.

        Warnings
        --------
        This method replace all existing coordinates.

        See Also
        --------
        add_coordset : Add one or a set of coordinates from a dataset.
        set_coordtitles : Set titles of the one or more coordinates.
        set_coordunits : Set units of the one or more coordinates.
        """
        self._coordset = None
        self.add_coordset(*args, dims=self.dims, **kwargs)

    def set_coordtitles(self, *args, **kwargs):
        """
        DEPRECATED. Use set_coordtitle
        """
        self.set_coordtitles(*args, **kwargs)

    def set_coordunits(self, *args, **kwargs):
        """
        Set units of the one or more coordinates.
        """
        self._coordset.set_units(*args, **kwargs)

    def sort(self, **kwargs):
        """
        Return the dataset sorted along a given dimension.

        By default, it is the last dimension [axis=-1]) using the numeric or label
        values.

        Parameters
        ----------
        dim : str or int, optional, default=-1
            Dimension index or name along which to sort.
        pos : int , optional
            If labels are multidimensional  - allow to sort on a define
            row of labels : labels[pos]. Experimental : Not yet checked.
        by : str among ['value', 'label'], optional, default=``value``
            Indicate if the sorting is following the order of labels or
            numeric coord values.
        descend : `bool`, optional, default=`False`
            If true the dataset is sorted in a descending direction. Default is False
            except if coordinates
            are reversed.
        inplace : bool, optional, default=`False`
            Flag to say that the method return a new object (default)
            or not (inplace=True).

        Returns
        -------
        |NDDataset|
            Sorted dataset.
        """

        inplace = kwargs.get("inplace", False)
        if not inplace:
            new = self.copy()
        else:
            new = self

        # parameter for selecting the level of labels (default None or 0)
        pos = kwargs.pop("pos", None)

        # parameter to say if selection is done by values or by labels
        by = kwargs.pop("by", "value")

        # determine which axis is sorted (dims or axis can be passed in kwargs)
        # it will return a tuple with axis and dim
        axis, dim = self._get_axis(**kwargs)
        if axis is None:
            axis, dim = self._get_axis(axis=0)

        # get the corresponding coordinates (remember the their order
        # can be different form the order
        # of dimension  in dims. S we cannot just take the coord from the indice.
        coord = getattr(self, dim)  # get the coordinate using the syntax such as self.x

        descend = kwargs.pop("descend", None)
        if descend is None:
            # when non specified, default is False (except for reversed coordinates
            descend = coord.reversed

        # import warnings
        # warnings.simplefilter("error")

        indexes = []
        for i in range(self.ndim):
            if i == axis:
                if not coord.has_data:
                    # sometimes we have only label for Coord objects.
                    # in this case, we sort labels if they exist!
                    if coord.is_labeled:
                        by = "label"
                    else:
                        # nothing to do for sorting
                        # return self itself
                        return self

                args = coord._argsort(by=by, pos=pos, descend=descend)
                setattr(new, dim, coord[args])
                indexes.append(args)
            else:
                indexes.append(slice(None))

        new._data = new._data[tuple(indexes)]
        if new.is_masked:
            new._mask = new._mask[tuple(indexes)]

        return new

    @property
    def origin(self):
        """
        Origin of the data.

        e.g. spectrometer or software
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value

    def take(self, indices, **kwargs):
        """
        Take elements from an array.

        Returns
        -------
        |NDDataset|
            A sub dataset defined by the input indices.
        """

        # handle the various syntax to pass the axis
        dims = self._get_dims_from_args(**kwargs)
        axis = self._get_dims_index(dims)
        axis = axis[0] if axis else None

        # indices = indices.tolist()
        if axis is None:
            # just do a fancy indexing
            return self[indices]

        if axis < 0:
            axis = self.ndim + axis

        index = tuple(
            [...] + [indices] + [slice(None) for i in range(self.ndim - 1 - axis)]
        )
        new = self[index]
        return new

    @property
    def timezone(self):
        """
        Return the timezone information.

        A timezone's offset refers to how many hours the timezone
        is from Coordinated Universal Time (UTC).

        A `naive` datetime object contains no timezone information. The
        easiest way to tell if a datetime object is naive is by checking
        tzinfo.  will be set to None of the object is naive.

        A naive datetime object is limited in that it cannot locate itself
        in relation to offset-aware datetime objects.

        In spectrochempy, all datetimes are stored in UTC, so that conversion
        must be done during the display of these datetimes using tzinfo.
        """
        return self._timezone

    @timezone.setter
    def timezone(self, val):
        try:
            self._timezone = pytz.timezone(val)
        except pytz.UnknownTimeZoneError:
            raise UnknownTimeZoneError(
                "You can get a list of valid timezones in "
                "https://en.wikipedia.org/wiki/tr.List_of_tz_database_time_zones ",
            )

    def to_array(self):
        """
        Return a numpy masked array.

        Other NDDataset attributes are lost.

        Returns
        -------
        |ndarray|
            The numpy masked array from the NDDataset data.

        Examples
        ========

        >>> dataset = scp.read('wodger.spg')
        >>> a = scp.to_array(dataset)

        equivalent to:

        >>> a = np.ma.array(dataset)

        or

        >>> a = dataset.masked_data
        """
        return np.ma.array(self)

    def to_xarray(self):
        """
        Convert a NDDataset instance to an `~xarray.DataArray` object.

        Warning: the xarray library must be available.

        Returns
        -------
        object
            A axrray.DataArray object.
        """

        xr = import_optional_dependency("xarray")
        coords = {}
        for index, name in enumerate(self.dims):
            coord = self.coordset[name]
            coords.update(coord._to_xarray())

        if self.is_masked:
            coords["mask"] = xr.Variable(dims=self.dims, data=self.mask)

        da = xr.DataArray(
            np.array(self.data, dtype=np.float64),
            dims=self.dims,
            coords=coords,
            name=self.name,
        )

        # 'modeldata', TODO:

        # 'referencedata', 'state', 'ranges'] # TODO: for GUI

        da.attrs["writer"] = "SpectroChemPy"
        da.attrs["name"] = self.name
        da.attrs["pint_units"] = str(self.units)
        # we cannot use units as it is
        # reserved by xarray
        da.attrs["title"] = self.title
        da.attrs["author"] = self.author
        da.attrs["description"] = self.description
        da.attrs["history"] = "\n".join(self.history)
        da.attrs["roi"] = self.roi
        da.attrs["created"] = self.created
        da.attrs["modified"] = self.modified
        da.attrs["origin"] = self.origin
        da.attrs["filename"] = self.filename
        for k, v in self.preferences.items():
            da.attrs[f"prefs_{k}"] = v
        for k, v in self.meta.items():
            da.attrs[f"meta_{k}"] = v

        da.attrs = encode_quantity(da.attrs)

        return da


# ======================================================================================
# module function
# ======================================================================================

# make some NDDataset operation accessible from the spectrochempy API
thismodule = sys.modules[__name__]

api_funcs = [
    "sort",
    "copy",
    "to_array",
    "to_xarray",
    "take",
    "set_complex",
    "set_hypercomplex",
    "component",
    "to",
    "to_base_units",
    "to_reduced_units",
    "ito",
    "ito_base_units",
    "ito_reduced_units",
    "is_units_compatible",
    "remove_masks",
]

# todo: check the fact that some function are defined also in ndmath
for funcname in api_funcs:
    setattr(thismodule, funcname, getattr(NDDataset, funcname))

    thismodule.__all__.append(funcname)

# load one method from NDIO
# load = NDDataset.load   # TODO: add
# __all__ += ["load"]

# ======================================================================================
# Set the operators
# ======================================================================================

# _set_operators(NDDataset, priority=100000)
# _set_ufuncs(NDDataset)
