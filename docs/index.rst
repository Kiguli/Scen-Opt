:hide-toc:

Scen-Opt
==========

**Data-driven convex optimization using the scenario approach.**

Scen-Opt solves Linear Programs (LP), Quadratic Programs (QP), and Semidefinite
Programs (SDP) using only sampled uncertainty realizations, providing rigorous
probabilistic guarantees on out-of-sample performance — no distributional
assumptions required.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Solvers
      :link: solvers
      :link-type: doc

      LP, QP, and SDP solvers with scenario constraints, slack variables, and
      regularization.

   .. grid-item-card:: Risk Bounds
      :link: risk
      :link-type: doc

      Distribution-free violation probability bounds via the Campi--Garatti
      theory.

   .. grid-item-card:: CVXPY Solvers
      :link: cvxpy_solvers
      :link-type: doc

      Installed solvers on the server and the full CVXPY compatibility matrix.

   .. grid-item-card:: Utilities
      :link: utilities
      :link-type: doc

      Solver discovery, active constraint detection, and file I/O helpers.

Getting Started
---------------

.. code-block:: bash

   pip install -r requirements.txt
   python3 app.py              # start the web interface at http://127.0.0.1:5000

Or with Docker:

.. code-block:: bash

   docker run -p 5000:5000 ghcr.io/kiguli/scen-opt:latest

How It Works
------------

1. Collect *N* random scenarios :math:`\delta_1, \ldots, \delta_N`
2. Formulate a convex program with soft scenario constraints relaxed by slack variables :math:`\zeta`
3. Solve the optimization problem using any of 27+ supported CVXPY solvers
4. Identify the *k* active (support) constraints
5. Compute distribution-free risk bounds :math:`[\varepsilon_L, \varepsilon_U]` on violation probability

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   solvers
   risk
   cvxpy_solvers
   utilities
