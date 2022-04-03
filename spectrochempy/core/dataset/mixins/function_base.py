import numpy as np


class NDArrayFunctionBaseMixin:
    # creation functions suitable for coordinates

    @classmethod
    def arange(cls, start=0, stop=None, step=None, **kwargs):
        """
        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval [start, stop).

        Parameters
        ----------
        start : number, optional
            Start of interval. The interval includes this value. The default start value is 0.
        stop : number
            End of interval. The interval does not include this value, except in some cases
            where step is not an integer and floating point round-off affects the length of out.
            It might be prefereble to use inspace in such case.
        step : number, optional
            Spacing between values. For any output out, this is the distance between two adjacent values,
            out[i+1] - out[i]. The default step size is 1. If step is specified as a position argument,
            start must also be given.
        **kwargs
            Keywords argument used when creating the returned object, such as units, name, title, ...

        Returns
        -------
        object
            Array of evenly spaced values.

        See Also
        --------
        linspace : Evenly spaced numbers with careful handling of endpoints.

        Examples
        --------

        >>> scp.arange(1, 20.0001, 1, units='s', name='mycoord')
        NDDataset: [float64] s (size: 20)
        """
        new = cls(np.arange(start, stop, step), **kwargs)
        new.history = f"Object created using `arange` function."
        return new

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True, **kwargs):
        """
        Return evenly spaced numbers over a specified interval.

        Returns num evenly spaced samples, calculated over the interval [start,
        stop]. The endpoint of the interval can optionally be excluded.

        Parameters
        ----------
        start : array_like
            The starting value of the sequence.
        stop : array_like
            The end value of the sequence, unless endpoint is set to False.
            In that case, the sequence consists of all but the last of num + 1 evenly
            spaced samples, so that stop is excluded. Note that the step size changes
            when endpoint is False.
        num : int, optional
            Number of samples to generate. Default is 50. Must be non-negative.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        **kwargs
            Keywords argument used when creating the returned object, such as units,
            name, title, ...

        Returns
        -------
        object
            There are num equally spaced samples in the closed interval [start, stop] or
            the half-open interval [start, stop) (depending on whether endpoint is
            True or False).
        """
        new = cls(np.linspace(start, stop, num, endpoint), **kwargs)
        new.history = f"Object created using `linspace` function."
        return new

    @classmethod
    def logspace(cls, start, stop, num=50, endpoint=True, base=10.0, **kwargs):
        """
        Return numbers spaced evenly on a log scale.

        In linear space, the sequence starts at ``base ** start``
        (`base` to the power of `start`) and ends with ``base ** stop``
        (see `endpoint` below).

        Parameters
        ----------
        start : array_like
            ``base ** start`` is the starting value of the sequence.
        stop : array_like
            ``base ** stop`` is the final value of the sequence, unless `endpoint`
            is False.  In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        num : int, optional
            Number of samples to generate.  Default is 50.
        endpoint : bool, optional
            If true, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        base : float, optional
            The base of the log space. The step size between the elements in
            ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
            Default is 10.0.
        **kwargs
            Keywords argument used when creating the returned object,
            such as units, name, title, ...

        Returns
        -------
        object
            `num` samples, equally spaced on a log scale.

        See Also
        --------
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples. Note that, when used with a float endpoint, the
                 endpoint may or may not be included.
        linspace : Similar to logspace, but with the samples uniformly distributed
                   in linear space, instead of log space.
        geomspace : Similar to logspace, but with endpoints specified directly.
        """
        new = cls(np.logspace(start, stop, num, endpoint, base), **kwargs)
        new.history = f"Obj created `logspace` function."
        return new

    @classmethod
    def geomspace(cls, start, stop, num=50, endpoint=True, **kwargs):
        """
        Return numbers spaced evenly on a log scale (a geometric progression).

        This is similar to `logspace`, but with endpoints specified directly.
        Each output sample is a constant multiple of the previous.

        Parameters
        ----------
        start : number
            The starting value of the sequence.
        stop : number
            The final value of the sequence, unless `endpoint` is False.
            In that case, ``num + 1`` values are spaced over the
            interval in log-space, of which all but the last (a sequence of
            length `num`) are returned.
        num : int, optional
            Number of samples to generate.  Default is 50.
        endpoint : bool, optional
            If true, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        **kwargs
            Keywords argument used when creating the returned object,
            such as units, name, title, ...

        Returns
        -------
        object
            `num` samples, equally spaced on a log scale.

        See Also
        --------
        logspace : Similar to geomspace, but with endpoints specified using log
                   and base.
        linspace : Similar to geomspace, but with arithmetic instead of geometric
                   progression.
        arange : Similar to linspace, with the step size specified instead of the
                 number of samples.
        """
        new = cls(np.geomspace(start, stop, num, endpoint), **kwargs)
        new.history = f"Obj created `geomspace` function."
        return new


def _set_kwargs_from(kwargs, array):
    if hasattr(array, "coordset"):
        coordset = array.coordset
        kwargs["coordset"] = kwargs.get("coordset", coordset)
    if hasattr(array, "units"):
        units = array.units
        kwargs["units"] = kwargs.get("units", units)
    if hasattr(array, "mask"):
        mask = array.mask
        kwargs["mask"] = kwargs.get("mask", mask)
    return kwargs


class NDDatasetFunctionCreationMixin:
    @classmethod
    def empty(cls, shape, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, without initializing entries.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the empty array.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        empty
            Array of uninitialized (arbitrary) data of the given shape, dtype, and
            order.  Object arrays will be initialized to None.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object.
        coordset : list or Coordset object
            Coordinates for the returned object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to 1.
        full : Fill a new array.

        Notes
        -----
        `empty`, unlike `zeros`, does not set the array values to zero,
        and may therefore be marginally faster.  On the other hand, it requires
        the user to manually set all the values in the array, and should be
        used with caution.

        Examples
        --------

        >>> scp.empty([2, 2], dtype=int, units='s')
        NDDataset: [int64] s (shape: (y:2, x:2))
        """
        new = cls(np.empty(shape), **kwargs)
        new.history = f"Object created using `empty` function."
        return new

    @classmethod
    def empty_like(cls, array, **kwargs):
        """
        Return a new uninitialized |NDDataset|.

        The returned |NDDataset| have the same shape and type as a given array.
        Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        array : |NDDataset| or array-like
            Object from which to copy the array structure.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        object
            Array with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input
            object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from
            the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of
        the input.
        ones_like : Return an array of ones with shape and type of input.
        zeros_like : Return an array of zeros with shape and type of input.
        empty : Return a new uninitialized array.
        ones : Return a new array setting values to one.
        zeros : Return a new array setting values to zero.
        full : Fill a new array.

        Notes
        -----
        This function does *not* initialize the returned array; to do that use
        for instance `zeros_like`, `ones_like` or `full_like` instead.  It may be
        marginally faster than the functions that do set the array values.
        """
        kwargs = _set_kwargs_from(array)
        new = cls(np.empty_like(array), **kwargs)
        new.history = f"Object created using `empty_like` function."
        return new

    @classmethod
    def eye(cls, N, M=None, k=0, dtype=float, **kwargs):
        """
        Return a 2-D array with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        M : int, optional
            Number of columns in the output. If None, defaults to `N`.
        k : int, optional
            Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal.
        dtype : data-type, optional
            Data-type of the returned array.
        **kwargs
            Other parameters to be passed to the object constructor (units, coordset,
            mask ...).

        Returns
        -------
        eye
            NDDataset of shape (N,M)
            An array where all elements are equal to zero, except for the `k`-th
            diagonal, whose values are equal to one.

        See Also
        --------
        identity : Equivalent function with k=0.
        diag : Diagonal 2-D NDDataset from a 1-D array specified by the user.

        Examples
        --------

        >>> scp.eye(2, dtype=int)
        NDDataset: [float64] unitless (shape: (y:2, x:2))
        >>> scp.eye(3, k=1, units='km').values
        <Quantity([[       0        1        0]
         [       0        0        1]
         [       0        0        0]], 'kilometer')>
        """

        return cls(np.eye(N, M, k, dtype), **kwargs)

    @classmethod
    def fromfunction(
        cls, function, shape=None, dtype=float, units=None, coordset=None, **kwargs
    ):
        """
        Construct a nddataset by executing a function over each coordinate.

        The resulting array therefore has a value ``fn(x, y, z)`` at coordinate ``(x,
        y, z)`` .

        Parameters
        ----------
        function : callable
            The function is called with N parameters, where N is the rank of
            `shape` or from the provided `coordset`.
        shape : (N,) tuple of ints, optional
            Shape of the output array, which also determines the shape of
            the coordinate arrays passed to `function`. It is optional only if
            `coordset` is None.
        dtype : data-type, optional
            Data-type of the coordinate arrays passed to `function`.
            By default, `dtype` is float.
        units : str, optional
            Dataset units.
            When None, units will be determined from the function results.
        coordset : |Coordset| instance, optional
            If provided, this determine the shape and coordinates of each dimension of
            the returned |NDDataset|. If shape is also passed it will be ignored.
        **kwargs
            Other kwargs are passed to the final object constructor.

        Returns
        -------
        fromfunction
            The result of the call to `function` is passed back directly.
            Therefore the shape of `fromfunction` is completely determined by
            `function`.

        See Also
        --------
        fromiter : Make a dataset from an iterable.

        Examples
        --------
        Create a 1D NDDataset from a function

        >>> func1 = lambda t, v: v * t
        >>> time = scp.Coord.arange(0, 60, 10, units='min')
        >>> d = scp.fromfunction(func1, v=scp.Quantity(134, 'km/hour'),
        coordset=scp.CoordSet(t=time))
        >>> d.dims
        ['t']
        >>> d
        NDDataset: [float64] km (size: 6)
        """

        from spectrochempy.core.dataset.coordset import CoordSet

        if coordset is not None:
            if not isinstance(coordset, CoordSet):
                coordset = CoordSet(*coordset)
            shape = coordset.sizes

        idx = np.indices(shape)

        args = [0] * len(shape)
        if coordset is not None:
            for i, co in enumerate(coordset):
                args[i] = co.data[idx[i]]
                if units is None and co.has_units:
                    args[i] = Quantity(args[i], co.units)

        kw = kwargs.pop("kw", {})
        data = function(*args, **kw)

        data = data.T
        dims = coordset.names[::-1]
        new = cls(data, coordset=coordset, dims=dims, units=units, **kwargs)
        new.ito_reduced_units()
        return new

    @classmethod
    def fromiter(cls, iterable, dtype=np.float64, count=-1, **kwargs):
        """
        Create a new 1-dimensional array from an iterable object.

        Parameters
        ----------
        iterable : iterable object
            An iterable object providing data for the array.
        dtype : data-type
            The data-type of the returned array.
        count : int, optional
            The number of items to read from iterable. The default is -1, which means
            all data is read.
        **kwargs
            Other kwargs are passed to the final object constructor.

        Returns
        -------
        fromiter
            The output nddataset.

        See Also
        --------
        fromfunction : Construct a nddataset by executing a function over each
        coordinate.

        Notes
        -----
            Specify count to improve performance. It allows fromiter to pre-allocate
            the output array,
            instead of resizing it on demand.

        Examples
        --------

        >>> iterable = (x * x for x in range(5))
        >>> d = scp.fromiter(iterable, float, units='km')
        >>> d
        NDDataset: [float64] km (size: 5)
        >>> d.data
        array([       0,        1,        4,        9,       16])
        """

        return cls(np.fromiter(iterable, dtype=dtype, count=count), **kwargs)

    @classmethod
    def full(cls, shape, fill_value=0.0, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with `fill_value`.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `np.int8`.  Default is
            fill_value.dtype.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        full
            Array of `fill_value`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input
            object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from
            the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.

        Examples
        --------

        >>> scp.full((2, ), np.inf)
        NDDataset: [float64] unitless (size: 2)
        >>> scp.NDDataset.full((2, 2), 10, dtype=np.int)
        NDDataset: [int64] unitless (shape: (y:2, x:2))
        """

        return cls(np.full(shape, fill_value, dtype), dtype=dtype, **kwargs)

    @classmethod
    def full_like(cls, dataset, fill_value=0.0, dtype=None, **kwargs):
        """
        Return a |NDDataset| of fill_value.

        The returned |NDDataset| have the same shape and type as a given array.
        Units, coordset, ... can be added in
        kwargs

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        fulllike
            Array of `fill_value` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input
            object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from
            the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------
        3 possible ways to call this method

        1) from the API

        >>> x = np.arange(6, dtype=int)
        >>> scp.full_like(x, 1)
        NDDataset: [float64] unitless (size: 6)

        2) as a classmethod

        >>> x = np.arange(6, dtype=int)
        >>> scp.NDDataset.full_like(x, 1)
        NDDataset: [float64] unitless (size: 6)

        3) as an instance method

        >>> scp.NDDataset(x).full_like(1, units='km')
        NDDataset: [float64] km (size: 6)
        """

        cls._data = np.full_like(dataset, fill_value, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    @classmethod
    def identity(cls, n, dtype=None, **kwargs):
        """
        Return the identity |NDDataset| of a given shape.

        The identity array is a square array with ones on
        the main diagonal.

        Parameters
        ----------
        n : int
            Number of rows (and columns) in `n` x `n` output.
        dtype : data-type, optional
            Data-type of the output.  Defaults to ``float``.
        **kwargs
            Other parameters to be passed to the object constructor (units, coordset,
            mask ...).

        Returns
        -------
        identity
            `n` x `n` array with its main diagonal set to one,
            and all other elements 0.

        See Also
        --------
        eye : Almost equivalent function.
        diag : Diagonal 2-D array from a 1-D array specified by the user.

        Examples
        --------

        >>> scp.identity(3).data
        array([[       1,        0,        0],
               [       0,        1,        0],
               [       0,        0,        1]])
        """

        return cls(np.identity(n, dtype), **kwargs)

    @classmethod
    def ones(cls, shape, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        ones
            Array of `ones`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        zeros : Return a new array setting values to zero.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> nd = scp.ones(5, units='km')
        >>> nd
        NDDataset: [float64] km (size: 5)
        >>> nd.values
        <Quantity([       1        1        1        1        1], 'kilometer')>
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True])
        >>> nd
        NDDataset: [int64] unitless (size: 5)
        >>> nd.values
        masked_array(data=[  --,        1,        1,        1,   --],
                     mask=[  True,   False,   False,   False,   True],
               fill_value=999999)
        >>> nd = scp.ones((5,), dtype=np.int, mask=[True, False, False, False, True], units='joule')
        >>> nd
        NDDataset: [int64] J (size: 5)
        >>> nd.values
        <Quantity([  --        1        1        1   --], 'joule')>
        >>> scp.ones((2, 2)).values
        array([[       1,        1],
               [       1,        1]])
        """

        return cls(np.ones(shape), dtype=dtype, **kwargs)

    @classmethod
    def ones_like(cls, dataset, dtype=None, **kwargs):
        """
        Return |NDDataset| of ones.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        oneslike
            Array of `1` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of the input.
        zeros_like : Return an array of zeros with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> x = scp.NDDataset(x, units='s')
        >>> x
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> scp.ones_like(x, dtype=float, units='J')
        NDDataset: [float64] J (shape: (y:2, x:3))
        """

        cls._data = np.ones_like(dataset, dtype)
        cls._dtype = np.dtype(dtype)

        return cls

    @classmethod
    def zeros(cls, shape, dtype=None, **kwargs):
        """
        Return a new |NDDataset| of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is
            `numpy.float64`.
        **kwargs
            Optional keyword parameters (see Other Parameters).

        Returns
        -------
        zeros
            Array of zeros.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        zeros_like : Return an array of zeros with shape and type of input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        full_like : Fill an array with shape and type of input.
        ones : Return a new array setting values to 1.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> nd = scp.NDDataset.zeros(6)
        >>> nd
        NDDataset: [float64] unitless (size: 6)
        >>> nd = scp.zeros((5, ))
        >>> nd
        NDDataset: [float64] unitless (size: 5)
        >>> nd.values
        array([       0,        0,        0,        0,        0])
        >>> nd = scp.zeros((5, 10), dtype=np.int, units='absorbance')
        >>> nd
        NDDataset: [int64] a.u. (shape: (y:5, x:10))
        """

        return cls(np.zeros(shape), dtype=dtype, **kwargs)

    @classmethod
    def zeros_like(cls, dataset, dtype=None, **kwargs):
        """
        Return a |NDDataset| of zeros.

        The returned |NDDataset| have the same shape and type as a given array. Units, coordset, ... can be added in
        kwargs.

        Parameters
        ----------
        dataset : |NDDataset| or array-like
            Object from which to copy the array structure.
        dtype : data-type, optional
            Overrides the data type of the result.
        **kwargs
            Optional keyword parameters (see Other Parameters).


        Returns
        -------
        zeorslike
            Array of `fill_value` with the same shape and type as `dataset`.

        Other Parameters
        ----------------
        units : str or ur instance
            Units of the returned object. If not provided, try to copy from the input object.
        coordset : list or Coordset object
            Coordinates for the returned object. If not provided, try to copy from the input object.

        See Also
        --------
        full_like : Return an array with a given fill value with shape and type of the input.
        ones_like : Return an array of ones with shape and type of input.
        empty_like : Return an empty array with shape and type of input.
        zeros : Return a new array setting values to zero.
        ones : Return a new array setting values to one.
        empty : Return a new uninitialized array.
        full : Fill a new array.

        Examples
        --------

        >>> x = np.arange(6)
        >>> x = x.reshape((2, 3))
        >>> nd = scp.NDDataset(x, units='s')
        >>> nd
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> nd.values
         <Quantity([[       0        1        2]
         [       3        4        5]], 'second')>
        >>> nd = scp.zeros_like(nd)
        >>> nd
        NDDataset: [float64] s (shape: (y:2, x:3))
        >>> nd.values
            <Quantity([[       0        0        0]
         [       0        0        0]], 'second')>
        """

        cls._data = np.zeros_like(dataset, dtype)
        cls._dtype = np.dtype(dtype)

        return cls
