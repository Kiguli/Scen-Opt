import numpy as np
import pytest

from src.Risk import quantify_risk


def test_risk_bounds_basic():
    """Verify risk bounds are valid for a typical case."""
    epsL, epsU = quantify_risk(k=10, N=4000, beta=1e-6)
    assert 0.0 <= epsL <= epsU <= 1.0


def test_risk_bounds_k_zero():
    """When k=0, risk bounds should be near zero."""
    epsL, epsU = quantify_risk(k=0, N=4000, beta=1e-6)
    assert epsL == 0.0
    assert epsU < 0.01


def test_risk_bounds_k_equals_N():
    """When k=N, upper bound should be 1."""
    epsL, epsU = quantify_risk(k=100, N=100, beta=1e-6)
    assert epsU == 1.0


def test_risk_bounds_monotonic():
    """Risk bounds should increase with k."""
    _, epsU_small = quantify_risk(k=5, N=4000, beta=1e-6)
    _, epsU_large = quantify_risk(k=50, N=4000, beta=1e-6)
    assert epsU_small < epsU_large


def test_risk_bounds_decrease_with_N():
    """For fixed k, more scenarios should tighten bounds."""
    _, epsU_small_N = quantify_risk(k=10, N=500, beta=1e-6)
    _, epsU_large_N = quantify_risk(k=10, N=5000, beta=1e-6)
    assert epsU_large_N < epsU_small_N


def test_risk_bounds_high_beta():
    """Higher beta (less conservative) should give tighter bounds."""
    _, epsU_low_beta = quantify_risk(k=10, N=4000, beta=1e-9)
    _, epsU_high_beta = quantify_risk(k=10, N=4000, beta=1e-3)
    assert epsU_high_beta < epsU_low_beta


def test_risk_invalid_inputs_raise():
    """k > N, N = 0 and β outside (0, 1) raise ValueError."""
    for k, N, beta in [(3, 2, 1e-6), (0, 0, 1e-6), (1, 10, 0.0), (1, 10, 1.0)]:
        with pytest.raises(ValueError):
            quantify_risk(k, N, beta)


def test_risk_returns_floats_and_keeps_x64_local():
    """Bounds are plain floats, and importing Risk does not switch JAX to 64-bit globally."""
    import jax
    epsL, epsU = quantify_risk(5, 5, 1e-6)
    assert isinstance(epsL, float) and isinstance(epsU, float) and epsU == 1.0
    assert not jax.config.jax_enable_x64
    np.testing.assert_allclose(quantify_risk(2, 100, 1e-6)[1], 0.20852406, atol=1e-8)
