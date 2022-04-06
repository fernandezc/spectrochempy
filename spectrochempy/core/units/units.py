# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
The core interface to the Pint library.
"""

__all__ = [
    "Unit",
    "Quantity",
    "ur",
    "DimensionalityError",
    "remove_args_units",
    "get_units",
    "remove_units",
    "encode_quantity",
    "set_nmr_context",
    "set_optical_context",
]

from warnings import warn
from functools import wraps

import numpy as np
from pint import (
    set_application_registry,
    UnitRegistry,
    DimensionalityError,
    formatting,
    Context,
)


from pint.unit import UnitsContainer, Unit  # , UnitDefinition
from pint.quantity import Quantity

# from pint.formatting import siunitx_format_unit
# from pint.converters import ScaleConverter


# ======================================================================================
# Modify the pint behaviour
# ======================================================================================


del formatting._FORMATTERS["P"]


@formatting.register_unit_format("P")
def format_pretty(unit, registry, **options):
    return formatting.formatter(
        unit.items(),
        as_ratio=False,
        single_denominator=False,
        product_fmt=".",
        division_fmt="/",
        power_fmt="{}{}",
        parentheses_fmt="({})",
        exp_call=formatting._pretty_fmt_exponent,
        **options,
    )


@formatting.register_unit_format("K")
def format_spectrochempy_compact(unit, registry, **options):
    return formatting.formatter(
        unit.items(),
        as_ratio=False,
        single_denominator=False,
        product_fmt=".",
        division_fmt="/",
        power_fmt="{}^{}",
        parentheses_fmt=r"({})",
        **options,
    )


del formatting._FORMATTERS["L"]


@formatting.register_unit_format("L")
def format_latex(unit, registry, **options):
    preprocessed = {
        r"\mathrm{{{}}}".format(u.replace("_", r"\_")): p for u, p in unit.items()
    }
    formatted = formatting.formatter(
        preprocessed.items(),
        as_ratio=False,
        single_denominator=True,
        product_fmt=r" \cdot ",
        division_fmt=r"\frac[{}][{}]",
        power_fmt="{}^[{}]",
        parentheses_fmt=r"\left({}\right)",
        **options,
    )
    return formatted.replace("[", "{").replace("]", "}")


del formatting._FORMATTERS["H"]


@formatting.register_unit_format("H")
def format_html(unit, registry, **options):
    return formatting.formatter(
        unit.items(),
        as_ratio=False,
        single_denominator=True,
        product_fmt=r".",
        division_fmt=r"{}/{}",
        power_fmt=r"{}<sup>{}</sup>",
        parentheses_fmt=r"({})",
        **options,
    )


del formatting._FORMATTERS["D"]


@formatting.register_unit_format("D")
def format_default(unit, registry, **options):
    return formatting.formatter(
        unit.items(),
        as_ratio=False,
        single_denominator=False,
        product_fmt="*",
        division_fmt="/",
        power_fmt="{}^{}",
        parentheses_fmt=r"({})",
        **options,
    )


del formatting._FORMATTERS["C"]


@formatting.register_unit_format("C")
def format_compact(unit, registry, **options):
    return formatting.formatter(
        unit.items(),
        as_ratio=False,
        single_denominator=False,
        product_fmt="*",
        division_fmt="/",
        power_fmt="{}**{}",
        parentheses_fmt=r"({})",
        **options,
    )


def _repr_html_(cls):
    p = cls.__format__("~H")
    # attempt to solve a display problem in notebook (recent version of pint
    # have a strange way to handle HTML. For me, it doesn't work)
    p = p.replace(r"\[", "").replace(r"\]", "").replace(r"\ ", " ")
    return p


setattr(Quantity, "_repr_html_", _repr_html_)
setattr(Quantity, "_repr_latex_", lambda cls: "$" + cls.__format__("~L") + "$")

# TODO: work on this latex format

setattr(
    Unit,
    "scaling",
    property(lambda u: u._REGISTRY.Quantity(1.0, u._units).to_base_units().magnitude),
)


# ------------------------------------------------------------------
def __format__(self, spec):
    # modify Pint unit __format__

    spec = formatting.extract_custom_flags(spec or self.default_format)
    if "~" in spec:
        if not self._units:
            return ""

        # Spectrochempy
        if self.dimensionless and "absorbance" not in self._units:
            if self._units == "ppm":
                units = UnitsContainer({"ppm": 1})
            elif self._units in ["percent"]:
                units = UnitsContainer({"%": 1})
            elif self._units == "weight_percent":
                units = UnitsContainer({"wt.%": 1})
            elif self._units == "radian":
                units = UnitsContainer({"rad": 1})
            elif self._units == "degree":
                units = UnitsContainer({"deg": 1})
            elif abs(self.scaling - 1.0) < 1.0e-10:
                units = UnitsContainer({"": 1})
            else:
                units = UnitsContainer(
                    {"scaled-dimensionless (%.2g)" % self.scaling: 1}
                )
        else:
            units = UnitsContainer(
                dict(
                    (self._REGISTRY._get_symbol(key), value)
                    for key, value in self._units.items()
                )
            )
        spec = spec.replace("~", "")
    else:
        units = self._units

    return formatting.format_unit(units, spec, registry=self._REGISTRY)


setattr(Unit, "__format__", __format__)

if globals().get("U_", None) is None:

    # filename = resource_filename(PKG, 'spectrochempy.txt')
    U_ = UnitRegistry(on_redefinition="ignore", autoconvert_offset_to_baseunit=True)
    U_.define(
        "__wrapped__ = 1"
    )  # <- hack to avoid an error with pytest (doctest activated)
    #  U_.define("@alias point = count")
    U_.define("percent = 0.01 = %")
    U_.define("weight_percent = 0.01 = wt.%")

    # Logaritmic Unit Definition
    #  Unit = scale; logbase; logfactor
    #  x_dB = [logfactor] * log( x_lin / [scale] ) / log( [logbase] )

    U_.define("transmittance = 0.01  = %")
    U_.define("absolute_transmittance = 100 * transmittance = - ")
    U_.define("absorbance = 100 * transmittance ; logbase: 10; logfactor: -1 = a.u.")
    # A = -np.log10(T%)

    U_.define("Kubelka_Munk = 1. = K.M.")

    U_.define("ppm = [ppm] = 1. = ppm")

    U_.default_format = "~P"
    Q_ = U_.Quantity
    Q_.default_format = "~P"

    set_application_registry(U_)
    del UnitRegistry  # to avoid importing it

else:
    warn("Unit registry was already set up. Bypassed the new loading")


# Context for NMR
# ------------------------------------------------------------------
def set_nmr_context(larmor):
    """
    Set a NMR context relative to the given Larmor frequency.

    Parameters
    ----------
    larmor : |Quantity| or float
        The Larmor frequency of the current nucleus.
        If it is not a quantity it is assumed to be given in MHz.

    Examples
    --------

    First we set the NMR context,

    >>> from spectrochempy.core.units import ur, set_nmr_context
    >>>
    >>> set_nmr_context(104.3 * ur.MHz)

    then, we can use the context as follows

    >>> fhz = 10000 * ur.Hz
    >>> with ur.context('nmr'):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    95.877 ppm

    or in the opposite direction

    >>> with ur.context('nmr'):
    ...     fhz = fppm.to('kHz')
    >>> print("{:~.3f}".format(fhz))
    10.000 kHz

    Now we update the context :

    >>> with ur.context('nmr', larmor=100. * ur.MHz):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    100.000 ppm

    >>> set_nmr_context(75 * ur.MHz)
    >>> fhz = 10000 * ur.Hz
    >>> with ur.context('nmr'):
    ...     fppm = fhz.to('ppm')
    >>> print("{:~.3f}".format(fppm))
    133.333 ppm
    """

    if not isinstance(larmor, U_.Quantity):
        larmor = larmor * U_.MHz

    if "nmr" not in U_._contexts:
        c = Context("nmr", defaults={"larmor": larmor})

        c.add_transformation(
            "[]",
            "[frequency]",
            lambda U_, x, **kwargs: x * kwargs.get("larmor") / 1.0e6,
        )
        c.add_transformation(
            "[frequency]",
            "[]",
            lambda U_, x, **kwargs: x * 1.0e6 / kwargs.get("larmor"),
        )
        U_.add_context(c)

    else:

        c = U_._contexts["nmr"]
        c.defaults["larmor"] = larmor


# Context for optical spectroscopy (IR)
# ------------------------------------------------------------------
def set_optical_context():
    """
    Set a IR context for transformation between absorbance and transmittance units.
    """

    if "optical" not in U_._contexts:
        c = Context("optical")

        c.add_transformation(
            "[transmittance]",
            "[absorbance]",
            lambda U_, x: -np.log10(x),
        )
        c.add_transformation(
            "[absorbance]", "[transmittance]", lambda U_, x: 10.0 ** (-x)
        )
        U_.add_context(c)

    else:
        c = U_._contexts["optical"]

    # if self.has_units:
    #     oldunits = self._units
    #     try:
    #         # particular case of dimensionless units: absorbance and
    #         # transmittance
    #
    #         if f"{oldunits:P}" in ["transmittance", "absolute_transmittance"]:
    #             if f"{units:P}" == "absorbance":
    #                 udata = (new.data * new.units).to(units)
    #                 new._data = -np.log10(udata.m)
    #                 new._units = units
    #                 new._title = "absorbance"
    #
    #             elif f"{units:P}" in ["transmittance", "absolute_transmittance", ]:
    #                 new._data = (new.data * new.units).to(units)
    #                 new._units = units
    #                 new._title = "transmittance"
    #
    #         elif f"{oldunits:P}" == "absorbance":
    #             if f"{units:P}" in ["transmittance", "absolute_transmittance"]:
    #                 scale = Quantity(1.0, self._units).to(units).magnitude
    #                 new._data = 10.0 ** -new.data * scale
    #                 new._units = units
    #                 new._title = "transmittance"
    #         else:
    #             new = self._unittransform(new, units)
    #             # change the title for spectroscopic units change
    #             if (oldunits.dimensionality in ["1/[length]", "[length]",
    #                 "[length] ** 2 * [mass] / [time] ** 2",
    #                                             ] and new._units.dimensionality
    #                 == "1/[time]"):
    #                 new._title = "frequency"
    #             elif (oldunits.dimensionality in ["1/[time]",
    #                                               "[length] ** 2 * [mass] / ["
    #                                               "time] ** 2"] and
    #                   new._units.dimensionality == "1/[length]"):
    #                 new._title = "wavenumber"
    #             elif (oldunits.dimensionality in ["1/[time]", "1/[length]",
    #                 "[length] ** 2 * [mass] / [time] ** 2",
    #                                               ] and new._units.dimensionality
    #                   == "[length]"):
    #                 new._title = "wavelength"
    #             elif (oldunits.dimensionality in ["1/[time]", "1/[length]",
    #                                               "[length]"] and
    #                   new._units.dimensionality == "[length] ** 2 * [mass] / ["
    #                                                "time] ** 2"):
    #                 new._title = "energy"
    #
    #     except pint.DimensionalityError as exc:
    #         if force:
    #             new._units = units
    #             info_("units forced to change")
    #         else:
    #             raise DimensionalityError(exc.dim1, exc.dim2, exc.units1,
    #                 exc.units2, extra_msg=exc.extra_msg, )
    #
    #
    #
    #


# enabled useful contexts
# --------------------------------------------------------------------------------------
U_.enable_contexts("spectroscopy", "boltzmann", "chemistry")
# we enable NMR context when needed as it depends on the parameter larmmor


# set alias for units
# --------------------------------------------------------------------------------------
ur = U_
Quantity = Q_

# utilities
# --------------------------------------------------------------------------------------


def remove_units(items, return_units=True):
    # recursive function
    # assume homogeneous units for list, tuple or slices
    units = None
    if isinstance(
        items,
        (
            list,
            tuple,
        ),
    ):
        _, units = remove_units(items[0])
        items = type(items)(remove_units(item, return_units=False) for item in items)
    elif isinstance(items, slice):
        start, units = remove_units(items.start, return_units=False)
        end = remove_units(items.stop, return_units=False)
        step = remove_units(items.step, return_units=False)
        items = slice(start, end, step)
    elif isinstance(items, Quantity):
        units = items.u
        items = float(items.m)
    else:
        units = None

    if return_units:
        return items, units
    return items


def remove_args_units(func):
    """
    Decorator which remove units of arguments of a function
    """

    @wraps(func)
    def new_func(*args, **kwargs):

        args = tuple([remove_units(arg, return_units=False) for arg in args])
        kwargs = {
            key: remove_units(val, return_units=False) for key, val in kwargs.items()
        }
        return func(*args, **kwargs)

    return new_func


def get_units(other):
    if other is None:
        return None
    if isinstance(other, str):
        units = ur.Unit(other)
    elif hasattr(other, "units"):
        units = other.units
    else:
        units = ur.Unit(other)
    return units


# utilities to encode quantity for export
def encode_quantity(val):
    # val is a dictionary containing quantity values
    for k, v in val.copy().items():
        if isinstance(v, Quantity):
            val[f"{k}"] = v.m
            val[f"pint_units_{k}"] = str(v.u)
    return val


if __name__ == "__main__":
    pass
