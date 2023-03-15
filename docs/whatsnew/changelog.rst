
What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the `revision` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* PCA score plot labelling (issue #543).
* Improved loading time

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Masks handling.
* Multicoordinates slicing work correctly.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* The `read` and all other `read_<protocol>` (with `<protocol>` such as `omnic`, `opus` ...) functions should no longer
  be used as methods of the NDDataset class.
  This behavior will be removed in version 0.6.0.
  This means that you should use::

    from spectrochempy import read
    read(some_data)

  or::

    import spectrochempy as scp
    scp.read(some_data)

  instead of::

    from spectrochempy import NDDataset
    NDDataset.read(some_data)

  or::

    import spectrochempy as scp
    scp.NDDataset.read(some_data)

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
