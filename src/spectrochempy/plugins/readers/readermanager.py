# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================


from spectrochempy.plugins.pluginmanager import PluginManager
from spectrochempy.plugins.readers.filetypes import registry
from spectrochempy.utils.file import check_filename_to_open


class ReaderManager(PluginManager):
    """Manager for reader plugins."""

    plugin_type = "readers"
    plugin_prefix = "read_"

    # Singleton pattern - store the single instance
    _instance = None
    _initialized = False

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the reader manager and register file types."""
        if not self._initialized:
            super().__init__()
            self._register_filetypes_from_plugins()
            self._setup_reader_methods()
            self.__class__._initialized = True

    def _register_filetypes_from_plugins(self):
        """Register file types from available plugins."""
        # Get plugin implementations
        hook_impls = self.pm.hook.get_filetype_info.get_hookimpls()
        if not hook_impls:
            return

        for impl in hook_impls:
            plugin = impl.plugin
            plugin_instance = self._get_plugin_instance(plugin)

            # Check if the plugin provides file type information
            if hasattr(plugin_instance, "get_filetype_info"):
                filetype_info = plugin_instance.get_filetype_info()
                if filetype_info:
                    identifier = filetype_info.get("identifier")
                    description = filetype_info.get("description")
                    extensions = filetype_info.get("extensions", [])
                    reader_method = filetype_info.get("reader_method")

                    if identifier and description:
                        registry.register_filetype(
                            identifier,
                            description,
                            extensions=extensions,
                            reader_method=reader_method,
                        )

    def _setup_reader_methods(self):
        """Set up reader methods based on registered file types."""
        for _identifier, method_name in registry.reader_methods.items():
            if not hasattr(self, method_name):
                # Create a reader method on this instance
                setattr(self, method_name, self._create_reader_method(method_name))

    def _create_reader_method(self, method_name):
        """Create a reader method for a specific file type."""

        def reader_method(filename, **kwargs):
            return self.read(filename, **kwargs)

        # Set the name and docstring
        reader_method.__name__ = method_name
        reader_method.__doc__ = f"Read files using the {method_name} format."

        return reader_method

    def read(self, *args, **kwargs):
        """Read file using appropriate plugin."""

        # Get plugin implementations
        hook_impls = self.pm.hook.can_read.get_hookimpls()
        if not hook_impls:
            raise ValueError("No reader plugins found.")

        # Check which plugin can read the file(s)
        for impl in hook_impls:
            plugin = impl.plugin
            # Get or create plugin instance
            plugin_instance = self._get_plugin_instance(plugin)

            files = check_filename_to_open(*args, **kwargs)

            if hasattr(plugin_instance, "can_read") and plugin_instance.can_read(
                files=files
            ):
                # Call read_file on the specific plugin that can handle the file
                return plugin_instance.read_file(files, **kwargs)

        # If no reader found, show available extensions
        try:
            extensions = []
            exts_impls = self.pm.hook.get_reader_extensions.get_hookimpls()
            for impl in exts_impls:
                plugin = impl.plugin
                plugin_instance = self._get_plugin_instance(plugin)
                if hasattr(plugin_instance, "get_reader_extensions"):
                    exts = plugin_instance.get_reader_extensions()
                    if exts:
                        extensions.extend(exts)

            supported = (
                ", ".join(str(ext) for ext in extensions) if extensions else "none"
            )
        except Exception:
            supported = "unknown"

        raise ValueError(
            f"No reader found for {filename}. Supported extensions: {supported}"
        )
