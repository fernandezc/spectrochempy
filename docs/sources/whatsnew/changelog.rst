
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also, do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments,
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* **Plugin System**: SpectroChemPy now supports a plugin system that allows users to extend the functionality of the software.
  - **Plugin Architecture**: The plugin system is based on the `pluggy` package, which provides a simple and flexible way to create and manage plugins.
  - **Custom Plugins**: Users can create custom plugins to add new features or modify existing ones.
  - **Plugin Hooks**: SpectroChemPy provides a set of predefined hooks that plugins can use to interact with the software.
  - **Plugin Registry**: Plugins are registered in a central registry, making it easy to manage and load them.
  - **Plugin Discovery**: SpectroChemPy automatically discovers and loads plugins from the user's environment.

  This new feature opens up a wide range of possibilities for extending and customizing SpectroChemPy to suit specific needs.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

* ``pluggy`` package is required.

.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

*
.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
