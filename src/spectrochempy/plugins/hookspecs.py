# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import pluggy

from spectrochempy.core.dataset.nddataset import NDDataset

hookspec = pluggy.HookspecMarker("spectrochempy")


class ReaderSpec:
    """Hook specifications for file readers."""

    @hookspec
    def get_filetype_info(self) -> dict:
        """
        Return file type information.

        Returns
        -------
        dict
            Dictionary with keys:
            - identifier: str - Unique identifier for this file type
            - description: str - Human-readable description
            - extensions: list - List of allowed file extensions
            - reader_method: str - Name of the reader method
        """

    @hookspec
    def can_read(self, files: dict) -> bool:
        """Check if this reader can handle the given file."""

    @hookspec
    def read_file(self, files: dict, protocol: str | list, **kwargs) -> "NDDataset":
        """Read the file and return a NDDataset object."""
