Solvers
=======

All solvers follow the scenario approach pattern:

1. Collect *N* random scenarios :math:`\delta_1, \ldots, \delta_N`
2. Formulate a convex program with soft scenario constraints :math:`A(\delta_i)x + b(\delta_i) \leq \zeta_i`
3. Solve with slack variables :math:`\zeta` penalized by :math:`\rho`
4. Identify active (support) constraints to determine complexity *k*
5. Quantify the risk of constraint violation via :doc:`risk`

All solvers accept optional regularization (:math:`\tau \|x - x_\text{ref}\|_p`) and hard
constraints that are not relaxed by slack variables.

----

Linear Programming
------------------

.. autofunction:: src.LP.solve_lp

----

Quadratic Programming
---------------------

.. autofunction:: src.QP.solve_qp

----

Semidefinite Programming
------------------------

.. autofunction:: src.SDP.solve_sdp
