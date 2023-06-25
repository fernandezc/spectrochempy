
:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* A new reader has been added: `read_wire` (alias `read_wdf``) to read data from
  the .wdf format (WDF) files produced by the ReniShaw WiRe software.
  This reader is based on the `py_wdf_reader <https://github.com/alchem0x2A/py-wdf-reader>`_ package.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Fix a bug when slicing dataset with an array or list of index: Multi-coordinates
  were not correctly handled.


.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* `preferences` are no longer an attribute of `NDDataset`. Code like this:

    .. code-block:: python

        >>> prefs = dataset.preferences

  should be replaced by:

    .. code-block:: python

        >>> prefs = scp.preferences

* The use of a reader as a NDDataset classmethod has been removed. Use instead the generic
  :func:`~spectrochempy.read` function or the specialized functions `read_<protocol>`\ , e.g.

    .. code-block:: python

         >>> from spectrochempy import read_jcamp
         >>> dataset = read_jcamp('path/to/file.jdx')

  or

     .. code-block:: python

            >>> import spectrochempy as scp
            >>> dataset = scp.read('path/to/file.jdx')


* The use of writer as NDDataset methods is deprecated. Use instead the :func:`~spectrochempy.write`
  function.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

