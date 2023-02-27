# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement the EFA (Evolving Factor Analysis) class.
"""
import numpy as np
import traitlets as tr

from spectrochempy.analysis._analysisutils import _wrap_ndarray_output_to_nddataset
from spectrochempy.analysis.abstractanalysis import DecompositionAnalysisConfigurable

__all__ = ["EFA"]
__configurables__ = ["EFA"]


class EFA(DecompositionAnalysisConfigurable):
    """
    Evolving Factor Analysis.

    Perform an Evolving Factor Analysis (forward and reverse) of the input |NDDataset|.

    Parameters
    ----------
    log_level : ["INFO", "DEBUG", "WARNING", "ERROR"], optional, default:"WARNING"
        The log level at startup
    config : Config object, optional
        By default the configuration is determined by the MCRALS.py
        file in the configuration directory. A traitlets.config.Config() object can
        eventually be used here.
    warm_start : bool, optional, default: false
        When fitting with SIMPLISMA repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        it may be possible to reuse previous model learned from the previous parameter
        value, saving time.
        When warm_start is true, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to fit.
    **kwargs
        Optional configuration  parameters.

    See Also
    --------
    PCA : Principal Component Analysis.
    NNMF : Perform a Non-Negative Matrix Factorization of a |NDDataset|.
    MCRALS : Perform MCR-ALS of a dataset knowing the initial C or St matrix.
    SVD :
    SIMPLISMA :
    """

    name = tr.Unicode("EFA")
    description = tr.Unicode("Evolving factor analysis model")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PCA, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions

    # _X = NDDatasetType()
    # _f_ev = NDDatasetType()
    # _b_ev = NDDatasetType()

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    cutoff = tr.Float(default_value=None, allow_none=True, help="Cut-off value").tag(
        config=True
    )

    used_components = tr.Int(
        allow_none=True, default_value=None, help="Number of components to keep."
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        config=None,
        warm_start=False,
        copy=True,
        **kwargs,
    ):
        # we have changed the name n_components use in sklearn by
        # used_components (in order  to avoid conflict with the rest of the progrma)
        # warn th user:
        if "n_components" in kwargs:
            raise KeyError(
                "`n_components` is not a valid parameter. Did-you mean "
                "`used_components`?"
            )

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            config=config,
            copy=copy,
            **kwargs,
        )

    def _fit(self, X, Y=None):
        # Y is ignored but necessary to corresponds to the signature in abstractanalysis
        # X has already been validated and eventually
        # preprocessed. X is now a nd-array with masked elements removed.
        # and this method shoud return _outfit

        # max number of components
        M, N = X.shape
        K = min(M, N)

        # ------------------------------------------------------------------------------
        # forward analysis
        # ------------------------------------------------------------------------------

        # f = NDDataset(
        #     np.zeros((M, K)),
        #     coordset=[X.y, Coord(range(K))],
        #     title="EigenValues",
        #     description="Forward EFA of " + X.name,
        #     history=str(datetime.now(timezone.utc)) + ": created by spectrochempy ",
        # )

        f = np.zeros((M, K))
        for i in range(M):
            s = np.linalg.svd(X[: i + 1], compute_uv=False)
            k = s.size
            f[i, :k] = s**2
            print(f"Evolving Factor Analysis: {int(i / (2 * M) * 100)}% \r", end="")

        # ------------------------------------------------------------------------------
        # backward analysis
        # ------------------------------------------------------------------------------
        # b = NDDataset(
        #     np.zeros((M, K)),
        #     coordset=[X.y, Coord(range(K))],
        #     title="EigenValues",
        #     name="Backward EFA of " + X.name,
        #     history=str(datetime.now(timezone.utc)) + ": created by spectrochempy ",
        # )

        b = np.zeros((M, K))
        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            s = np.linalg.svd(X[i:M], compute_uv=False)
            k = s.size
            b[i, :k] = s**2
            print(
                f"Evolving Factor Analysis: {int(100 - i / (2 * M) * 100)} % \r", end=""
            )

        # store the components number (real or desired)
        self._n_components = K

        # return results
        _outfit = f, b
        return _outfit

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _transform(self, X=None):
        # X is ignored for EFA
        # Return concentration profile
        return self._get_conc()

    def _get_conc(self):
        f, b = self._outfit
        M = f.shape[0]
        K = self._n_components
        if self.used_components is not None:
            K = min(K, self.used_components)
        c = np.zeros((M, K))
        for i in range(M):
            c[i] = np.min((f[i, :K], b[i, :K][::-1]), axis=0)
        return c

    # ----------------------------------------------------------------------------------
    # Public methods/properties
    # ----------------------------------------------------------------------------------
    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typex="components")
    def f_ev(self):
        """
        Eigenvalues for the forward analysis (|NDDataset|).
        """
        f = self._outfit[0]
        if self.cutoff is not None:
            f = np.max((f, np.ones_like(f) * self.cutoff), axis=0)
        return f

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typex="components")
    def b_ev(self):
        """
        Eigenvalues for the backward analysis (|NDDataset|).
        """
        b = self._outfit[1]
        if self.cutoff is not None:
            b = np.max((b, np.ones_like(b) * self.cutoff), axis=0)
        return b


# ======================================================================================
if __name__ == "__main__":
    pass
