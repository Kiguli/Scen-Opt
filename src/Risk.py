import jax
# The bisection compares two betainc terms whose difference is ~1e-8 near the
# upper root; in JAX's default float32 that difference is pure noise, which
# flips the comparison and can leave epsU understated by up to ~0.01 at
# beta=1e-6 (the high-confidence regime this tool targets). float64 makes the
# root agree with a float64 SciPy reference to ~1e-10.
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import betainc

def quantify_risk(k,N,beta):
    r"""Compute scenario approach risk bounds on constraint violation probability.

    Uses the theory of Campi and Garatti with the regularized incomplete beta
    function to compute distribution-free bounds on the probability of
    out-of-sample constraint violation via bisection.

    The true violation probability satisfies:

    .. math::

        \varepsilon \;\in\; [\varepsilon_L,\; \varepsilon_U] \quad \text{with confidence at least } 1 - \beta

    Parameters
    ----------
    k : int
        Cardinality of the support list from the scenario optimization.
    N : int
        Number of sampled scenarios.
    beta : float
        Confidence parameter (e.g. ``1e-6`` for high confidence).

    Returns
    -------
    epsL : float
        Lower bound on the constraint violation probability.
    epsU : float
        Upper bound on the constraint violation probability.

    Notes
    -----
    The incomplete beta function ``betainc`` used here follows the JAX/SciPy
    argument convention, which differs from MATLAB (the first argument appears
    last in JAX).
    """
    t1 = 0.0
    t2 = k/N
    threshold = 1e-10


    while (t2 - t1) > threshold:
        t = (t1+t2)/2
        left = beta/3*betainc(k+1,N-k,t)+beta/6*betainc(k+1,4*N+1-k,t)
        #print("left = ",left)
        right = (1+beta/6/N)*t*N*(betainc(k,N-k+1,t)-betainc(k+1,N-k,t))
        #print("right= ", right)
        if left > right: #added threshold
            t1 = t
        else:
            t2 = t
    epsL = t1 #set lower bound
    #print("epsL = ", epsL)

    if (k==N):
        epsU = 1 #set upper bound
    else:
        t1 = k/N
        t2 = 1
        while (t2 - t1) > threshold:
            t = (t1 + t2) / 2
            left = (beta / 2 - beta / 6) * betainc(k + 1, N - k,t) + beta / 6 * betainc(k + 1, 4 * N + 1 - k,t)
            #print(left)
            right = (1 + beta / 6 / N) * t * N * (betainc( k, N - k + 1,t) - betainc( k + 1, N - k,t))
            #print(right)
            if left > right:
                t2 = t
            else:
                t1 = t
        epsU = t2 #set upper bound
    #print("epsU = ", epsU)
    # Return results
    return epsL, epsU

def quantify_conf():
    """
        To create...
    """


    # Return results
    return 0.0