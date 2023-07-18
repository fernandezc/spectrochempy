# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Peak finding module.

Contains wrappers of `scipy.signal` peak finding functions.
"""

__all__ = ["PeakFinder", "find_peaks"]
__configurables__ = ["PeakFinder"]
__dataset_methods__ = ["find_peaks"]

import numpy as np
import scipy

import traitlets as tr

from spectrochempy.application import warning_
from spectrochempy.core.units import Quantity
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils import exceptions
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.utils.objects import Adict
from spectrochempy.processing.transformation.concatenate import stack, concatenate

# Todo:
# find_peaks_cwt(vector, widths[, wavelet, ...]) Attempt to find the peaks in a 1-D array.
# argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
# argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
# argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.


class PeakFinder(BaseConfigurable):
    __doc__ = _docstring.dedent(
        r"""
    Peak finder class.

    The PeakFinder class is a wrapper of `scipy.signal` peak finding functions.
    It can be used to find peaks inside a `NDDataset` based on peak properties.

    If the NDDataSet has more than one dimension, the peak finding is performed
    along the last dimensions, except if the `dim` argument
    is given. 

    Parameters
    ----------
    %(BaseConfigurable.parameters)s

    See Also
    --------
    find_peaks: Find peaks inside a 1D `NDDataset` based on peak properties.
    """
    )

    use_coord = tr.Bool(
        default_value=True,
        help="Set whether the dimension coordinates (when it exists) should be used instead of indices for the positions and width. If True, the units of the other parameters are interpreted according to the coordinates.",
    ).tag(config=True)

    height = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Required height of peaks. Either a number, None, a list or array matching the x-coordinate size or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required height.",
    ).tag(config=True)

    threshold = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Required threshold of peaks, the vertical distance to its neighbouring samples. Either a number, None, an array matching the x-coordinate size or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required threshold.",
    ).tag(config=True)

    distance = tr.Union(
        (tr.Float(), tr.Instance(Quantity)),
        default_value=None,
        allow_none=True,
        help="Required minimal horizontal distance in samples between neighbouring peaks. Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.",
    ).tag(config=True)

    prominence = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Required prominence of peaks. Either a number, None, an array matching the x-coordinate size or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required prominence.",
    ).tag(config=True)

    width = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Required width of peaks in samples. Either a number, None, an array matching the x-coordinate size or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied, as the maximal required width. Floats are interpreted as width measured along the 'x' Coord; ints are interpreted as a number of points.",
    ).tag(config=True)

    wlen = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Used for calculation of the peaks prominences, thus it is only used if one of the arguments prominence or width is given. Floats are interpreted as measured along the 'x' Coord; ints are interpreted as a number of points. See argument len in peak_prominences of the scipy documentation for a full description of its effects.",
    ).tag(config=True)

    rel_height = tr.Float(
        default_value=0.5,
        allow_none=True,
        help="Used for calculation of the peaks width, thus it is only used if width is given. See argument rel_height in peak_widths of the scipy documentation for a full description of its effects.",
    ).tag(config=True)

    plateau_size = tr.Union(
        (tr.Float(), tr.Instance(Quantity), tr.List(), Array()),
        default_value=None,
        allow_none=True,
        help="Required size of the flat top of peaks in samples. Either a number, None, an array matching the x-coordinate size or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second, if supplied as the maximal required plateau size. Floats are interpreted as measured along the 'x' Coord; ints are interpreted as a number of points.",
    ).tag(config=True)

    window_length = tr.Integer(
        default_value=5,
        help="The length of the filter window used to interpolate the maximum. window_length must be a positive odd integer. If set to one, the actual maximum is returned.",
    ).tag(config=True)

    # =================================================================================
    # Runtime parameters
    # =================================================================================
    _dim = tr.Union((tr.Integer(), tr.Unicode()), allow_none=True, default_value=None)

    _coord = tr.Instance(Coord)
    _use_coord = tr.Bool()

    _distance = tr.Integer(allow_none=True)
    _width = tr.Integer(allow_none=True)
    _wlen = tr.Integer(allow_none=True)
    _plateau_size = tr.Integer(allow_none=True)

    _outsearch = tr.List(tr.Instance(Adict))

    # ==================================================================================
    # Initialisation
    # ==================================================================================
    def __init__(
        self,
        log_level="WARNING",
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            **kwargs,
        )

    def __call__(self, X, dim=None):
        return self.search(X, dim)

    # =================================================================================
    # Private methods
    # =================================================================================
    def _check_unit_compatibility(self, X, par, axis):
        # check units compatibility, convert to units of data and return magnitude
        units = X.coord(axis).units if X.coordset is not None else None
        parunits = par.units if hasattr(par, "units") else None
        if parunits is None:
            return par  # we assume that the parameters has the units of the data
        elif units is None and parunits is not None:
            raise exceptions.UnitsCompatibilityError(  # pragma: no cover
                f"Units of the data are None. The parameter {par.name} should have no units"
            )
        elif units is not None and parunits is not None:
            if units.dimensionality != parunits.dimensionality:
                raise exceptions.UnitsCompatibilityError(
                    f"Units of the data ({units}) and parameter {par.name}"
                    f" ({parunits}) are not compatible."
                )
            # should be compatible, so convert
            par.ito(units)
            return par.magnitude

    @tr.validate(
        "height", "threshold", "distance", "prominence", "width", "wlen", "plateau_size"
    )
    def _par_validate(self, proposal):
        if proposal.value is None or self._X_is_missing:
            return proposal.value

        if isinstance(proposal.value, (list, tuple)):
            for i, val in enumerate(proposal.value):
                proposal.value[i] = self._check_unit_compatibility(
                    self._X, val, self._dim
                )
        else:
            proposal.value = self._check_unit_compatibility(
                self._X, proposal.value, self._dim
            )
        return proposal.value

    @tr.observe("_X_coordset")
    def _X_coordset_changed(self, change):
        coordset = change.new
        self._coord = coordset[self._dim].default if coordset is not None else None

        # Check if we can work with the coordinates
        self._use_coord = self.use_coord and self._coord is not None

        # init variable step in case we do not use coordinates
        step = 1

        if self._use_coord:
            # assume linear x coordinates
            if not self._coord.linear:
                warning_(
                    f"The {self._dim} coordinates are not linear. "
                    "The peak finding might be erroneous."
                )
                spacing = np.mean(self._coord.spacing)
            else:
                spacing = self._coord.spacing
            if isinstance(spacing, Quantity):
                spacing = spacing.magnitude
            step = np.abs(spacing)

        self._distance = (
            int(round(self.distance / step)) if self.distance is not None else None
        )
        self._width = int(round(self.width / step)) if self.width is not None else None
        self._wlen = int(round(self.wlen / step)) if self.wlen is not None else None
        self._plateau_size = (
            int(round(self.plateau_size / step))
            if self.plateau_size is not None
            else None
        )

    def _quadratic_interpolation(self, peaks, data, mode):
        # quadratic interpolation to find the maximum

        window_length = (
            self.window_length
            if self.window_length % 2 == 0
            else self.window_length - 1
        )

        pos = []
        heights = []
        for peak in peaks:
            if window_length > 1:
                start = peak - window_length // 2
                end = peak + window_length // 2 + 1
                sle = slice(start, end)
                y = data[sle]
                x = range(start, end) if not self._use_coord else self._coord[sle]
                coef = np.polyfit(x, y, 2)
                x_at_max = -coef[1] / (2 * coef[0])
                if mode == "1D":  # the input was a 1D dataset
                    # thus take maximum of the interpolated peak
                    y_at_max = np.poly1d(coef)(x_at_max)
                else:
                    # the input was a 2D dataset
                    # thus take the x coordinate of the maximum of the interpolated peak
                    # but the intensity at this position for each individal spectra
                    y_at_max = self._X[..., x_at_max]
                pos.append(x_at_max)
            else:
                pos.append(peak if not self._use_coord else self._coord[peak])
                y_at_max = self._X[..., peak]

            heights.append(y_at_max)

        return pos, heights

    # ===============================================================================
    # Public methods
    # ===============================================================================
    def search(self, X, dim=None):
        """
        Search for peak's position.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ )
            The data to find peaks in.

        Returns
        -------
        `NDDataset`
            Peak's properties.
        """
        # get the name of the last axis
        self._dim = X.get_axis(-1)[1]

        # fire the validation process
        self._X = X

        # if X is a 2D dataset, we take the projection along the last dimension
        if self._X_preprocessed.squeeze().ndim > 1:
            mode = "2D"
            data = self._X_preprocessed.sum(axis=0)
        else:
            mode = "1D"
            data = self._X_preprocessed.squeeze()

        # now the distance, width ... parameters are given in data points
        # and the dataset is reduced to a 1D dataset

        # find peaks and properties
        peaks, properties = scipy.signal.find_peaks(
            data,
            height=self.height,
            threshold=self.threshold,
            distance=self._distance,
            prominence=self.prominence,
            width=self._width,
            wlen=self._wlen,
            rel_height=self.rel_height,
            plateau_size=self._plateau_size,
        )

        pos, heights = self._quadratic_interpolation(peaks, data, mode)

        coords = Coord(
            pos, name="x", units=self._coord.units if self._use_coord else None
        )
        heights = concatenate(heights, newcoord=coords)

        outsearch = NDDataset()

        self._outsearch.append(outsearch)

        if len(self._outsearch) > 1:
            return self._outsearch[0]
        else:
            return self._outsearch


"""
        `~numpy.ndarray`
            Indices of peaks in `dataset` that satisfy all given conditions.
            The shape of the results depends on the dimensionality of `X` and
            the `dim` argument of the PeakFinder object.

            * If `X` is 1D and `dim` is None, the result is a 1D array of indices.
            * If `X` is 2D and `dim` is None, the result is a 2D array of indices of
                shape (2, n_peaks) where the first row contains the indices along the
                first dimension and the second row contains the indices along the second
                dimension.
            * If `X` is 2D and `dim` is 0 or `y`, the result is a 1D array of indices
                along the first dimension.
            * If `X` is 2D and `dim` is 1 or `x`, the result is a 1D array of indices
                along the second dimension.
"""


def find_peaks(
    dataset,
    height=None,
    window_length=5,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
    use_coord=True,
):
    """
    Wrapper and extension of `scpy.signal.find_peaks`\ .

    Find peaks inside a 1D `NDDataset` based on peak properties.
    This function finds all local maxima by simple comparison of neighbouring values.
    Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    .. warning::

        This function may return unexpected results for data containing NaNs.
        To avoid this, NaNs should either be removed or replaced.

    Parameters
    ----------
    dataset : `NDDataset`
        A 1D NDDataset or a 2D NDdataset with `len(X.y) == 1` .
    height : `float` or :term:`array-like`\ , optional, default: `None`
        Required height of peaks. Either a number, `None` , an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required height.
    window_length : `int`, default: 5
        The length of the filter window used to interpolate the maximum. window_length
        must be a positive odd integer.
        If set to one, the actual maximum is returned.
    threshold : `float` or :term:`array-like`\ , optional
        Required threshold of peaks, the vertical distance to its neighbouring
        samples. Either a number, `None` , an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : `float`\ , optional
        Required minimal horizontal distance in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : `float` or :term:`array-like`\ , optional
        Required prominence of peaks. Either a number, `None` , an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : `float` or :term:`array-like`\ , optional
        Required width of peaks in samples. Either a number, `None` , an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width. Floats are interpreted as width
        measured along the 'x' Coord; ints are interpreted as a number of points.
    wlen : `int` or `float`, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
        See argument len` in `peak_prominences` of the scipy documentation for a full
        description of its effects.
    rel_height : `float`, optional,
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` of the scipy documentation
        for a full description of its effects.
    plateau_size : `float` or :term:`array-like`\ , optional
        Required size of the flat top of peaks in samples. Either a number,
        `None` , an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
    use_coord : `bool`\ , optional
        Set whether the x Coord (when it exists) should be used instead of indices
        for the positions and width. If True, the units of the other parameters
        are interpreted according to the coordinates.

    Returns
    -------
    peaks : `~numpy.ndarray`
        Indices of peaks in `dataset` that satisfy all given conditions.

    properties : `dict`
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * ``peak_heights``
            If `height` is given, the height of each peak in `dataset`\  .
        * ``left_thresholds``\ , ``right_thresholds``
            If `threshold` is given, these keys contain a peaks vertical
            distance to its neighbouring samples.
        * ``prominences``\ , ``right_bases``\ , ``left_bases``
            If `prominence` is given, these keys are accessible. See
            `scipy.signal.peak_prominences` for a
            full description of their content.
        * ``width_heights``\ , ``left_ips``\ , ``right_ips``
            If `width` is given, these keys are accessible. See
            `scipy.signal.peak_widths` for a full description of their content.
        * plateau_sizes, left_edges', 'right_edges'
            If `plateau_size` is given, these keys are accessible and contain
            the indices of a peak's edges (edges are still part of the
            plateau) and the calculated plateau sizes.

        To calculate and return properties without excluding peaks, provide the
        open interval `(None, None)` as a value to the appropriate argument
        (excluding `distance`\ ).

    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths` ).

    See Also
    --------
    find_peaks_cwt:
        In `scipy.signal`: Find peaks using the wavelet transformation.
    peak_prominences:
        In `scipy.signal`: Directly calculate the prominence of peaks.
    peak_widths:
        In `scipy.signal`: Directly calculate the width of peaks.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `scipy.signal.find_peaks_cwt` ).

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`\ ) can be given as half-open or
      closed intervals, e.g `1` or `(1, None)` defines the half-open
      interval :math:`[1, \\infty]` while `(None, 1)` defines the interval
      :math:`[-\\infty, 1]`\ . The open interval `(None, None)` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `dataset` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size` ,
      `height` , `threshold` , `distance` , `prominence` , `width` . In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance` .
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `dataset` is large or has many local maxima
      (see `scipy.signal.peak_prominences`\ ).

    Examples
    --------

    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> X = dataset[0, 1800.0:1300.0]
    >>> peaks, properties = X.find_peaks(height=1.5, distance=50.0, width=0.0)
    >>> len(peaks.x)
    2
    >>> peaks.x.values
    <Quantity([    1644     1455], 'centimeter^-1')>
    >>> properties["peak_heights"][0]
    <Quantity(2.26663446, 'absorbance')>
    >>> properties["widths"][0]
    <Quantity(38.729003, 'centimeter^-1')>
    """

    # get the dataset
    X = dataset.squeeze()
    if X.ndim > 1:
        raise ValueError(
            "Works only for 1D NDDataset or a 2D NDdataset with `len(X.y) <= 1`"
        )
    # TODO: implement for 2D datasets (would be useful e.g., for NMR)
    # be sure that data are real (NMR case for instance)
    if X.is_complex or X.is_quaternion:
        X = X.real

    # Check if we can work with the coordinates
    use_coord = use_coord and X.coordset is not None

    # init variable in case we do not use coordinates
    lastcoord = None
    xunits = 1
    dunits = 1
    step = 1

    if use_coord:
        # We will use the last coordinates (but if the data were transposed or sliced,
        # the name can be something else than 'x')
        lastcoord = X.coordset[X.dims[-1]]

        # units
        xunits = lastcoord.units if lastcoord.units is not None else 1
        dunits = X.units if X.units is not None else 1

        # assume linear x coordinates
        # TODO: what if the coordinates are not linear?
        if not lastcoord.linear:
            warning_(
                "The x coordinates are not linear. " "The peak finding might be wrong."
            )
            spacing = np.mean(lastcoord.spacing)
        else:
            spacing = lastcoord.spacing
        if isinstance(spacing, Quantity):
            spacing = spacing.magnitude
        step = np.abs(spacing)

    # transform coord (if exists) to index
    # TODO: allow units for distance, width, wlen, plateau_size
    distance = int(round(distance / step)) if distance is not None else None
    width = int(round(width / step)) if width is not None else None
    wlen = int(round(wlen / step)) if wlen is not None else None
    plateau_size = int(round(plateau_size / step)) if plateau_size is not None else None

    # now the distance, width ... parameters are given in data points
    peaks, properties = scipy.signal.find_peaks(
        X.data,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )

    out = X[peaks]

    if not use_coord:
        out.coordset = None  # remove the coordinates

    # quadratic interpolation to find the maximum
    window_length = window_length if window_length % 2 == 0 else window_length - 1
    x_pos = []
    if window_length > 1:
        for i, peak in enumerate(peaks):
            start = peak - window_length // 2
            end = peak + window_length // 2 + 1
            sle = slice(start, end)

            y = X.data[sle]
            x = lastcoord.data[sle] if use_coord else range(start, end)

            coef = np.polyfit(x, y, 2)

            x_at_max = -coef[1] / (2 * coef[0])
            y_at_max = np.poly1d(coef)(x_at_max)

            out[i] = y_at_max
            if not use_coord:
                x_pos.append(x_at_max)
            else:
                out.coordset(out.dims[-1])[i] = x_at_max
    if x_pos and not use_coord:
        from spectrochempy.core.dataset.coord import Coord

        out.coordset = Coord(x_pos)

    # transform back index to coord
    if use_coord:
        for key in ["peak_heights", "width_heights", "prominences"]:
            if key in properties:
                properties[key] = [height * dunits for height in properties[key]]

        for key in (
            "left_bases",
            "right_bases",
            "left_edges",
            "right_edges",
        ):  # values are initially of int type
            if key in properties:
                properties[key] = [
                    lastcoord.values[int(index)]
                    for index in properties[key].astype("float64")
                ]

        def _prop(ips):
            # interpolate coord
            floor = int(np.floor(ips))
            return lastcoord.values[floor] + (ips - floor) * (
                lastcoord.values[floor + 1] - lastcoord.values[floor]
            )

        for key in ("left_ips", "right_ips"):  # values are float type
            if key in properties:
                properties[key] = [_prop(ips) for ips in properties[key]]

        if "widths" in properties:
            properties["widths"] = [
                np.abs(width * step) * xunits for width in properties["widths"]
            ]

        if "plateau_sizes" in properties:
            properties["plateau_sizes"] = [
                np.abs(sizes * step) * xunits for sizes in properties["plateau_sizes"]
            ]

    out.name = "peaks of " + X.name
    out.history = f"find_peaks(): {len(peaks)} peak(s) found"

    return out, properties
