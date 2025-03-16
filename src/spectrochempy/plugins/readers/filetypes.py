# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import warnings


class FileTypeRegistry:
    """Registry for file types and their handlers."""

    def __init__(self):
        self._filetypes: list[tuple[str, str]] = []
        self._aliases: dict[str:str] = {}
        self._exporttypes: list[tuple[str, str]] = []
        self._reader_methods: dict[str, str] = {}

    def register_filetype(
        self,
        identifier: str,
        description: str,
        extensions: list[str] = None,
        reader_method: str = None,
    ):
        """
        Register a new file type.

        Parameters
        ----------
        identifier : str
            Unique identifier for the file type
        description : str
            Human-readable description of the file type
        extensions : list[str], optional
            List of file extensions for this file type
        reader_method : str, optional
            Name of the reader method to use for this file type
        """
        # Don't add duplicates
        for idx, (id_, _) in enumerate(self._filetypes):
            if id_ == identifier:
                # Update existing entry
                self._filetypes[idx] = (identifier, description)
                break
        else:
            # Add new entry if not found
            self._filetypes.append((identifier, description))

        if extensions and reader_method:
            for ext in extensions:
                # Check if alias already exists
                alias = f"read_{ext}"
                for a, meth in self._aliases.items():
                    if a == alias:
                        # Update existing alias
                        self._aliases[alias] = reader_method
                        warnings.warn(
                            f"Alias '{alias}:{meth}' already exists and was updated to '{reader_method}'",
                            stacklevel=2,
                        )
                        break
                # Add new alias if not found
                self._aliases[alias] = reader_method

        if reader_method:
            self._reader_methods[identifier] = reader_method
            # Also register method names for each alias
            if extensions:
                for ext in extensions:
                    if ext not in self._reader_methods:
                        self._reader_methods[ext] = reader_method

    def register_exporttype(self, identifier: str, description: str):
        """Register a new export type."""
        for idx, (id_, _) in enumerate(self._exporttypes):
            if id_ == identifier:
                # Update existing entry
                self._exporttypes[idx] = (identifier, description)
                break
        else:
            # Add new entry if not found
            self._exporttypes.append((identifier, description))

    @property
    def filetypes(self) -> list[tuple[str, str]]:
        """Get all registered file types."""
        return self._filetypes.copy()

    @property
    def aliases(self) -> dict[str, str]:
        """Get all registered aliases."""
        return self._aliases.copy()

    @property
    def exporttypes(self) -> list[tuple[str, str]]:
        """Get all registered export types."""
        return self._exporttypes.copy()

    @property
    def reader_methods(self) -> dict[str, str]:
        """Get mapping of file types/extensions to reader methods."""
        return self._reader_methods.copy()

    def get_reader_method(self, identifier_or_alias: str) -> str:
        """Get the reader method for a file type or alias."""
        # First check if it's a direct identifier
        if identifier_or_alias in self._reader_methods:
            return self._reader_methods[identifier_or_alias]

        # Then check if it's an alias
        for alias, identifier in self._aliases:
            if alias == identifier_or_alias:
                return self._reader_methods.get(identifier)

        return None


# Global registry instance
registry = FileTypeRegistry()
