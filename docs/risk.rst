Risk Quantification
===================

After solving a scenario program, the **complexity** *k* (number of support constraints) and the
**sample size** *N* are used to compute distribution-free bounds on the probability of constraint
violation — the **risk** — without requiring any knowledge of the underlying probability
distribution.

Scenario Approach Background
----------------------------

Consider a decision variable :math:`x \in \mathcal{X} \subset \mathbb{R}^d` and an uncertain
parameter :math:`\delta \in \Delta \subset \mathbb{R}^q` drawn i.i.d. from an unknown probability
measure :math:`\mathbb{P}`. For each scenario :math:`\delta`, the set
:math:`\mathcal{X}_\delta \subseteq \mathcal{X}` denotes the region of feasible decisions. The
**scenario program** is:

.. math::

   \min_{x \in \mathcal{X}} \quad & c(x) \\
   \text{s.t.} \quad & x \in \bigcap_{i=1}^{N} \mathcal{X}_{\delta_i}

Given the optimal solution :math:`x_N^*`, the **risk** quantifies the probability that a newly
drawn scenario renders the solution inappropriate:

.. math::

   V(x) = \mathbb{P}\!\left\{\delta \in \Delta : x \notin \mathcal{X}_\delta\right\}

Since :math:`\mathbb{P}` is unknown, the scenario approach provides distribution-free bounds on
:math:`V(x_N^*)` using only the **complexity** :math:`s_N^*` — the cardinality of the smallest
irreducible subset of scenarios that fully determines the solution.

Support Constraints and Complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **support list** is a sublist :math:`(\delta_{i_1}, \ldots, \delta_{i_k})` of the scenarios such
that:

1. Solving the program with only these scenarios yields the same solution :math:`x_N^*`.
2. The sublist is **irreducible**: removing any single scenario changes the solution.

The **complexity** :math:`s_N^*` is the minimal cardinality among all support lists. This is the
value *k* computed by :func:`~src.Miscellaneous.get_support` and passed to
:func:`~src.Risk.quantify_risk`.

Risk Bounds
-----------

Upper Bound (Always Valid)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The upper bound on risk holds under the sole assumption of **consistency** (which is guaranteed
for all convex optimization problems handled by Scen-Opt). For a prescribed confidence level
:math:`1 - \beta`, the risk certificate of Garatti and Campi [GC2025]_ states:

.. math::

   \mathbb{P}^N\!\left\{V(x_N^*) > \epsilon(s_N^*)\right\} \leq \beta

where :math:`\epsilon(k) = 1 - t(k)` and :math:`t(k) \in (0,1)` is the unique solution of

.. math::

   \frac{\beta}{N} \sum_{i=k}^{N-1} \binom{i}{k} t^{i-k} \;-\; \binom{N}{k} t^{N-k} \;=\; 0, \qquad k = 0, 1, \ldots, N-1

with the convention :math:`\epsilon(N) = 1`.

Two-Sided Bounds (Requires Non-Degeneracy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under the additional assumption of **non-degeneracy** (there exists a unique support list),
a two-sided certificate from Garatti and Campi [GC2022]_ provides both lower and upper bounds.
For :math:`k = 0, 1, \ldots, N-1`, consider the polynomial equation in :math:`t`:

.. math::

   \binom{N}{k} t^{N-k}
   \;-\; \frac{\beta}{2N} \sum_{i=k}^{N-1} \binom{i}{k} t^{i-k}
   \;-\; \frac{\beta}{6N} \sum_{i=N+1}^{4N} \binom{i}{k} t^{i-k}
   \;=\; 0

This admits exactly two solutions in :math:`[0, +\infty)`, denoted
:math:`\underline{t}(k) \leq \overline{t}(k)`. For :math:`k = N`, a separate equation yields a
single solution :math:`\overline{t}(N)`, with :math:`\underline{t}(N) = 0`. The bounds are:

.. math::

   \underline{\epsilon}(k) = \max\!\big\{0,\; 1 - \overline{t}(k)\big\}, \qquad
   \overline{\epsilon}(k) = 1 - \underline{t}(k)

and the two-sided risk certificate reads:

.. math::

   \mathbb{P}^N\!\left\{
      \underline{\epsilon}(s_N^*) \;\leq\; V(x_N^*) \;\leq\; \overline{\epsilon}(s_N^*)
   \right\} \;\geq\; 1 - \beta

Scen-Opt computes these bounds numerically via bisection using the regularized incomplete beta
function, following the procedure described in [CGC2023]_.

.. warning::

   The **lower bound** :math:`\underline{\epsilon}` is valid only when the non-degeneracy
   assumption holds. If Scen-Opt detects degeneracy during active constraint identification
   (indicated by the ``degeneracy`` flag in the solver output), the lower bound should be
   disregarded and only the **upper bound** :math:`\overline{\epsilon}` used. While Scen-Opt
   attempts to detect degeneracy automatically, such checks cannot cover out-of-sample
   scenarios — it is the user's responsibility to assess whether the lower bound remains
   applicable.

.. tip::

   A common choice is :math:`\beta = 10^{-6}`. Smaller values of :math:`\beta` give wider
   bounds but higher confidence.

----

API Reference
-------------

.. autofunction:: src.Risk.quantify_risk

----

References
----------

.. [GC2025] S. Garatti and M. C. Campi, "Non-convex scenario optimization,"
   *Mathematical Programming*, vol. 209, no. 1, pp. 557--608, 2025.

.. [GC2022] S. Garatti and M. C. Campi, "Risk and complexity in scenario optimization,"
   *Mathematical Programming*, vol. 191, no. 1, pp. 243--279, 2022.

.. [CGC2023] M. C. Campi and S. Garatti, "Compression, generalization and learning,"
   *Journal of Machine Learning Research*, vol. 24, no. 339, pp. 1--74, 2023.
