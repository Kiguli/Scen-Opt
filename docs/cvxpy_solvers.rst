CVXPY Solvers
=============

Scen-Opt delegates all optimization to `CVXPY <https://www.cvxpy.org/>`_, which
provides a unified interface to a wide range of open-source and commercial solvers.
This page lists the solvers bundled with the Scen-Opt server and the full set of
solvers supported by CVXPY.

Installed Solvers
-----------------

The following solvers are installed on the Scen-Opt server and available
out of the box. Any of these can be passed via the **solver** dropdown in the
web interface or the ``solver`` argument in the Python API.

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 60

   * - Solver
     - LP
     - QP
     - SDP
     - Notes
   * - `CLARABEL <https://clarabel.org/>`_
     - X
     - X
     - X
     - Default solver for most problem types. Open-source, pure Rust.
   * - `SCS <https://www.cvxgrp.org/scs/>`_
     - X
     - X
     - X
     - Splitting conic solver. Handles all cone types including SDP.
   * - `MOSEK <https://www.mosek.com/>`_
     - X
     - X
     - X
     - Commercial solver (free academic license). High performance and
       reliability for LP, QP, and SDP.
   * - `ECOS <https://github.com/embotech/ecos>`_
     - X
     - X
     - X
     - Lightweight interior-point solver.
   * - `CVXOPT <https://cvxopt.org/>`_
     - X
     - X
     - X
     - Python-native interior-point solver.
   * - `OSQP <https://osqp.org/>`_
     - X
     - X
     -
     - Operator-splitting QP solver. LP and QP only.
   * - `DAQP <https://github.com/darnstrom/daqp>`_
     - X
     - X
     -
     - Dual active-set solver for LP and QP.
   * - `PIQP <https://predict-epfl.github.io/piqp/>`_
     - X
     - X
     -
     - Proximal interior-point QP solver.
   * - `HiGHS <https://highs.dev/>`_
     - X
     - X
     -
     - High-performance LP/QP solver. Open-source.
   * - `SciPy <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_
     - X
     - X
     -
     - Uses SciPy's ``linprog`` / ``minimize`` backends.

.. tip::

   If no solver is specified, CVXPY automatically selects the most specialized
   solver for the problem type. For SDP problems, CLARABEL or SCS will be used.
   For QP problems, OSQP is typically preferred.

Additional Solvers
------------------

Beyond the solvers installed on the server, CVXPY supports many additional solvers
that can be installed separately. The full compatibility matrix is shown below.

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10

   * - Solver
     - LP
     - QP
     - SDP
   * - CBC
     - X
     - X
     -
   * - COPT
     - X
     - X
     - X
   * - CPLEX
     - X
     - X
     - X
   * - GLOP
     - X
     -
     -
   * - GLPK
     - X
     -
     -
   * - GLPK_MI
     - X
     -
     -
   * - GUROBI
     - X
     - X
     - X
   * - PDLP
     - X
     -
     -
   * - PROXQP
     - X
     - X
     -
   * - QPALM
     - X
     - X
     -
   * - SCIP
     - X
     - X
     - X
   * - SDPA
     - X
     - X
     - X
   * - XPRESS
     - X
     - X
     - X

.. note::

   Only LP, QP, and SDP columns are shown — these are the problem types supported
   by Scen-Opt. Solvers already listed in the Installed Solvers table above are
   omitted here.

For installation instructions and solver-specific options, see the
`CVXPY solver documentation <https://www.cvxpy.org/tutorial/solvers/index.html>`_.

----

References
----------

.. [DB2016] S. Diamond and S. Boyd, "CVXPY: A Python-embedded modeling language for convex
   optimization," *Journal of Machine Learning Research*, vol. 17, no. 83, pp. 1--5, 2016.

.. [BV2004] S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press,
   2004.
