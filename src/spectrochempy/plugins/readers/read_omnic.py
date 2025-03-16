import pluggy

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.plugins.pluginmanager import Plugin

hookimpl = pluggy.HookimplMarker("spectrochempy")


# generic reader for Omnic files
class OmnicReader(Plugin):
    """Reader for Omnic files."""

    # Hooks implementation
    # --------------------
    @hookimpl
    def get_filetype_info(self):
        return {
            "identifier": "omnic",
            "description": "Nicolet OMNIC files and series (*.spa *.spg *.srs)",
            "extensions": ["spa", "spg", "srs"],
            "reader_method": "read_omnic",
        }

    @hookimpl
    def can_read(self, files: dict) -> bool:
        return all(f in self.reader_extensions for f in files)

    @hookimpl
    def read_file(self, files: dict, **kwargs) -> "NDDataset":
        print(files)
        return files

    # Readeer properties
    # ------------------
    @property
    def reader_extensions(self) -> list:
        """Return list of file extensions this reader can handle."""
        finfo = self.get_filetype_info()
        return ["." + ext for ext in finfo.get("extensions", [])]
