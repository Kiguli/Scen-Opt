Scen-O-Con Documentation
========================

**Scen-O-Con** (Scenario Optimization Toolbox) solves convex optimization problems using the
data-driven scenario approach of Campi and Garatti. It provides solvers for Linear Programs (LP),
Quadratic Programs (QP), and Semidefinite Programs (SDP).

The scenario approach replaces uncertain constraints with a finite number of sampled scenarios,
then quantifies the probability of constraint violation using support-based risk bounds — no
distributional assumptions required.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   solvers
   risk
   utilities
