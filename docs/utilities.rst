Utilities
=========

Solver Discovery
----------------

.. autofunction:: src.Miscellaneous.get_solvers

.. autofunction:: src.Miscellaneous.get_norm_types

----

File Loading
------------

.. autofunction:: src.Miscellaneous.load_file

----

Active Constraint Detection
----------------------------

These functions identify the support list — the constraints whose removal changes the
optimal value — a key step for computing the scenario approach risk bounds.

.. autofunction:: src.Miscellaneous.get_support

.. autofunction:: src.Miscellaneous.test_support
