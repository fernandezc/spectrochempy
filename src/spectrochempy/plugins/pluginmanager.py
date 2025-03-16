# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

import importlib.util
import inspect
import sys
from pathlib import Path

import pluggy


class PluginManager:
    """Base plugin manager."""

    namespace = "spectrochempy"
    plugin_type = None  # To be overridden by subclasses (e.g., "readers")
    plugin_prefix = ""  # To be overridden by subclasses (e.g., "read_")

    # Class-level cache of plugin managers to avoid duplicate registration
    _plugin_managers = {}

    def __init__(self):
        # Use the class name as a key to retrieve or create a plugin manager
        manager_key = f"{self.namespace}.{self.plugin_type}"

        # Check if we've already created a plugin manager for this type
        if manager_key in self._plugin_managers:
            self.pm = self._plugin_managers[manager_key]
        else:
            from spectrochempy.plugins.hookspecs import ReaderSpec

            self.pm = pluggy.PluginManager(self.namespace)

            # Add hook specifications first before loading plugins
            self.pm.add_hookspecs(ReaderSpec)

            # Add plugins from entry points and local directory
            self._load_local_directory_plugins()
            self._load_entry_points_plugins()

            # Store the plugin manager in the class cache
            self._plugin_managers[manager_key] = self.pm

            # Check if any plugins were loaded
            if not self.has_plugins():
                print(f"Warning: No {self.plugin_type} plugins were registered.")  # noqa: T201

    def _load_entry_points_plugins(self):
        """Load plugins from entry points."""
        if self.plugin_type:  # Only load if plugin_type is defined
            self.pm.load_setuptools_entrypoints(f"{self.namespace}.{self.plugin_type}")
        else:
            print(
                "Warning: plugin_type not defined, skipping entry point plugin loading"
            )  # noqa: T201

    def _load_local_directory_plugins(self):
        """Search for plugins in local directory."""
        # from spectrochempy import error_  # here to avoid circular imports

        if not self.plugin_type:
            print(
                "Warning: plugin_type not defined, skipping local directory plugin loading"
            )  # noqa: T201
            return

        # Get the directory path based on plugin_type
        plugin_dir = Path(__file__).parent / self.plugin_type

        if not plugin_dir.exists():
            print(f"Warning: Plugin directory not found: {plugin_dir}")  # noqa: T201
            return

        # Find all Python files in the directory
        for filepath in plugin_dir.iterdir():
            # Skip files that don't match the plugin prefix or aren't Python files
            if not filepath.name.endswith(".py") or (
                self.plugin_prefix and not filepath.name.startswith(self.plugin_prefix)
            ):
                continue

            module_path = str(filepath)
            module_name = f"{self.namespace}.plugins.{self.plugin_type}.{filepath.stem}"

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Find plugin classes and register instances
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, Plugin)
                            and attr is not Plugin
                        ):
                            try:
                                # Register with explicit name (class name)
                                plugin_instance = attr()
                                self.pm.register(plugin_instance, name=attr.__name__)
                                print(f"Successfully registered plugin: {attr_name}")  # noqa: T201
                            except Exception as e:
                                print(f"Error registering plugin {attr_name}: {e}")  # noqa: T201
            except Exception as e:
                print(f"Error loading plugin {filepath.name}: {e}")  # noqa: T201

    def _get_plugin_instance(self, plugin):
        """Get plugin instance from class or return existing instance."""
        if inspect.isclass(plugin):
            # Plugin is a class, instantiate it
            return plugin()
        # Plugin is already an instance
        return plugin

    def has_plugins(self):
        """Check if any plugins are registered."""
        return bool(self.pm.get_plugins())


class Plugin:
    """Base class for plugins."""

    namespace = "spectrochempy"  # Add namespace here

    @property
    def name(self):
        """Reader name."""
        return self.__class__.__name__.lower()
