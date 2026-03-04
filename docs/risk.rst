Risk Quantification
===================

After solving a scenario program, the **complexity** *k* (number of support constraints) and the
**sample size** *N* are used to compute distribution-free bounds on the probability of constraint
violation. These bounds come from the theory of Campi and Garatti and rely on the regularized
incomplete beta function.

Given a confidence parameter :math:`\beta`, the function returns an interval
:math:`[\varepsilon_L, \varepsilon_U]` such that the true violation probability lies within
this interval with confidence at least :math:`1 - \beta`.

.. autofunction:: src.Risk.quantify_risk
