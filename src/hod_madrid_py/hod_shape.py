# hod_shape.py

"""
Halo Occupation Distribution shape functions and occupation samplers.

Overview
--------
This module implements the analytical HOD prescriptions used to assign
central and satellite galaxies to dark-matter haloes.

It contains three main layers of functionality:

1. Central-galaxy occupation models
   These describe the probability that a halo hosts a central galaxy.
   Three analytical families are implemented:
   - erf-like transition
   - Gaussian occupation
   - Gaussian plus power-law tail

2. Satellite-galaxy occupation models
   These describe the mean number of satellites as a power law above a
   characteristic mass threshold, together with stochastic samplers that
   convert the mean into an integer realization.

3. Vectorized and Numba-accelerated array routines
   These apply the HOD model efficiently to many haloes at once, which is the
   form used in the main mock-generation loops.

Conventions
-----------
- 'logM' always denotes base-10 halo mass.
- 'M' always denotes linear halo mass in Msun/h.
- Central occupation functions return Bernoulli realizations, not mean
  probabilities.
- Satellite occupations are sampled from a count distribution controlled by
  the parameter 'beta'.
"""
import numpy as np
from numba import jit
from numba import njit
import math
import src.hod_madrid_py.hod_pdf as pdf
import src.hod_madrid_py.hod_io as io
from typing import Tuple


@jit(nopython=True)
def erf_approx(x: float) -> float:
    """
    Approximate the error function erf(x) with an Abramowitz & Stegun formula.

    This approximation is used for compatibility with Numba nopython mode,
    avoiding a Python-level call to 'math.erf' in environments where that may
    not be supported by Numba versions in use.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Approximation to erf(x) in [-1, 1].

    Notes
    -----
    Uses A&S 7.1.26 “fast erf”:
        erf(x) ≈ sign(x) * [1 - (((((a5 t + a4) t + a3) t + a2) t + a1) t) e^(-x²)],
    with t = 1 / (1 + p |x|) and constants (a1..a5, p) chosen for small max error.
    Accuracy is sufficient for probabilistic HOD usage.
    """

    # A&S 7.1.26 constants
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    # Horner scheme for the polynomial, then multiply by exp(-x^2)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y

@jit(nopython=True)
def HOD_erf(logM: float, mu: float, sig: float, As: float, rng) -> int:
    """
    Bernoulli draw for central occupation using an erf-based sigmoid (HOD1).

    Probability model
    -----------------
    r = As * 0.5 * [1 + erf( (logM - mu) / sig )]

    Parameters
    ----------
    logM : float
        log10 halo mass.
    mu : float
        log10 characteristic mass where the sigmoid transitions.
    sig : float
        Width (scatter) of the transition. Larger sig → smoother transition.
    As : float
        Amplitude (max probability is ~As for high-mass halos). Ensure 0 ≤ As ≤ 1.

    Returns
    -------
    int
        1 if a central is assigned (with probability r), else 0.

    Notes
    -----
    - This returns a single random realization (not the expectation r).
    - For parameter values that push r outside [0,1], you should clip or
      constrain inputs upstream; this function assumes a valid range.
    """
    r = As * 0.5 * (1.0 + erf_approx((logM - mu) / sig))
    rand_val = rng.random()
    return 1 if rand_val < r else 0

@jit(nopython=True)
def HOD_gauss(logM: float, mu: float, sig: float, As: float, rng) -> int:
    """
    Bernoulli draw for central occupation using a Gaussian-shaped probability (HOD2).

    Probability model
    -----------------
    r = As * [ 1 / (sig * sqrt(2π)) ] * exp( - (logM - mu)^2 / (2 sig^2) )

    Parameters
    ----------
    logM : float
        log10 halo mass.
    mu : float
        log10 mass at the Gaussian peak.
    sig : float
        Gaussian width (standard deviation) in log10 mass.
    As : float
        Amplitude scaling. Choose As so that r ≤ 1 over the mass range of interest.

    Returns
    -------
    int
        1 if a central is assigned (with probability r), else 0.

    Notes
    -----
    - This returns a single random realization (not the expectation r).
    - Unlike the sigmoid, the Gaussian's peak height depends on (As, sig);
      you must ensure the maximum r does not exceed 1, or clamp upstream.
    """
    r = As / (sig * math.sqrt(2.0 * math.pi)) * math.exp(- (logM - mu) ** 2 / (2.0 * sig ** 2))
    rand_val = rng.random()
    return 1 if rand_val < r else 0

@jit(nopython=True)
def HOD_gaussPL(logM, mu, sig, Ac, gamma, rng):
    """
    Bernoulli draw for central occupation using a Gaussian + Power-law hybrid (HOD3).

    Model
    -----
    For logM < mu (low-mass side)      : r = Ac / (sig * sqrt(2π)) * exp(-(logM - mu)^2 / (2 sig^2))
    For logM ≥ mu (high-mass tail)     : r = Ac / (sig * sqrt(2π)) * 10^{ gamma * (logM - mu) }

    The function computes a probability r (ideally 0 ≤ r ≤ 1) and returns a
    single 0/1 realization with P(central=1) = r.

    Parameters
    ----------
    logM : float
        log10 halo mass.
    mu : float
        log10 characteristic mass where the transition occurs.
    sig : float
        Gaussian width (in log10 mass) controlling the low-mass side.
    Ac : float
        Amplitude (peak normalization near logM≈mu). Choose so that r ≤ 1.
    gamma : float
        Power-law slope for the high-mass tail (often negative).

    Returns
    -------
    int
        1 if a central is assigned (with probability r), else 0.

    Notes
    -----
    - r is capped at 1.0 (as in the original C code) to avoid invalid probabilities.
    - Upstream parameter validation should ensure r ≥ 0 as well; if you expect
      extreme inputs, clamp with r = max(0.0, min(1.0, r)).
    """
    if logM < mu:
        # Gaussian regime (low masses)
        r = Ac / (sig * math.sqrt(2.0 * math.pi)) * math.exp(- (logM - mu)**2 / (2.0 * sig**2))
    else:
        # Power-law regime (high masses)
        r = Ac / (sig * math.sqrt(2.0 * math.pi)) * (10.0 ** (gamma * (logM - mu)))

    # Cap at 1.0 to maintain a valid Bernoulli probability
    if r > 1.0:
        r = 1.0
    # (Optional safety) ensure non-negative:
    # if r < 0.0:
    #     r = 0.0

    rand_val = rng.random()
    return 1 if rand_val < r else 0

@jit(nopython=True)
def satellite_mean_powerlaw(M, M0, M1, alpha, As):
    """
    Compute the mean satellite occupation for a power-law satellite model.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    M0 : float
        Minimum mass threshold for satellite occupation.
    M1 : float
        Characteristic normalization mass.
    alpha : float
        Power-law slope.
    As : float
        Satellite amplitude.

    Returns
    -------
    float
        Mean number of satellites expected in the halo.
    """
    if M1 <= 0.0 or M < M0:
        return 0.0

    xsat = (M - M0) / M1
    if xsat <= 0.0:
        return 0.0

    return As * (xsat ** alpha)

@jit(nopython=True)
def sample_satellite_occupation(mean_sat, beta, rng):
    """
    Draw an integer satellite occupation from the chosen count distribution.

    Parameters
    ----------
    mean_sat : float
        Mean satellite occupation.
    beta : float
        Dispersion-control parameter selecting the counting distribution.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    int
        Realized number of satellites.
    """
    if mean_sat <= 0.0:
        return 0

    # Select the counting distribution based on beta
    if beta < -1.0:
        return pdf.next_integer(mean_sat, rng)
    elif beta <= 0.0 and beta >= -1.0 / 171.0:
        return pdf.poisson_sample(mean_sat, rng)
    elif beta < -1.0 / 171.0 and beta >= -1.0:
        return pdf.binomial_sample(mean_sat, beta, rng)
    elif beta > 0.0:
        return pdf.neg_binomial_sample(mean_sat, beta, rng)
    else:
        # Fallback (defensive): Poisson
        return pdf.poisson_sample(mean_sat, rng)

def HOD_shape_file(M, beta, filename, rng):
    """
     Draw a satellite occupation using a tabulated HOD shape stored in an HDF5 file.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    beta : float
        Dispersion-control parameter selecting the satellite count distribution.
    filename : str
        Path to the HDF5 file containing the tabulated occupations.
    rng : numpy.random.Generator-like
        Random-number generator.

    Returns
    -------
    int
        Realized number of satellites for the halo.
    """
    # 1) Read binned means from file
    M_min, M_max, _Ncen_mean, Nsat_mean = io.read_occupation_from_h5(filename)

    # 2) Find mass bin i such that M_min[i] <= M < M_max[i]
    i = -1
    n = len(M_min)
    for k in range(n):
        if (M >= M_min[k]) and (M < M_max[k]):
            i = k
            break
    # Edge case: include exact upper edge of the last bin
    if i == -1 and n > 0 and M == M_max[-1]:
        i = n - 1

    if i == -1:
        return 0  # mass out of range

    mean_n = float(Nsat_mean[i])
    if mean_n <= 0.0:
        return 0

    # 3) Sample count according to beta
    if beta < -1.0:
        return pdf.next_integer(mean_n, rng)
    elif (-1.0/171.0) <= beta <= 0.0:
        return pdf.poisson_sample(mean_n, rng)
    elif -1.0 <= beta < (-1.0/171.0):
        return pdf.binomial_sample(mean_n, beta, rng)
    elif beta > 0.0:
        return pdf.neg_binomial_sample(mean_n, beta, rng)
    else:
        # Fallback: Poisson
        return pdf.poisson_sample(mean_n, rng)



@jit(nopython=True)
def get_hod_derived_params(mu: float, hodshape: int) -> Tuple[float, float, float, float, float]:
    """
    Return the default analytical HOD parameters derived from 'mu'.

    Parameters
    ----------
    mu : float
        Characteristic halo mass in log10 units.
    hodshape : int
        Identifier of the analytical HOD family.

    Returns
    -------
    tuple
        '(M0, M1, alpha, sig, gamma)' corresponding to the selected HOD family.
    """
    if hodshape == 1:  # HOD1
        M0 = 10.0**mu
        M1 = 10.0**(mu + 1.3)
        alpha = 1.0
        sig = 0.15
        gamma = -1.4
    elif hodshape == 2:  # HOD2
        M0 = 10.0**(mu - 0.1)
        M1 = 10.0**(mu + 0.3)
        alpha = 0.8
        sig = 0.12
        gamma = -1.4
    else:  # HOD3 (default)
        M0 = 10.0**(mu - 0.05)
        M1 = 10.0**(mu + 0.35)
        alpha = 0.9
        sig = 0.08
        gamma = -1.4

    return M0, M1, alpha, sig, gamma

from numba import jit
import numpy as np

@jit(nopython=True)
def calculate_hod_occupation(
    M,
    mu,
    Ac,
    As,
    alpha,
    sig,
    gamma,
    M0,
    M1,
    hodshape,
    beta=0.0,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
    rng=None
):
    """
    Draw the full HOD occupation for a single halo using linear halo mass.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    mu, Ac, As, alpha, sig, gamma, M0, M1 : float
        Analytical HOD parameters.
    hodshape : int
        Identifier of the central-occupation model.
    beta : float, optional
        Dispersion-control parameter for the satellite count distribution.
    conformity : bool, optional
        Whether to rescale the satellite mean according to the central outcome.
    K1_global, K2_global : float, optional
        Conformity rescaling factors.
    rng : numpy.random.Generator-like, optional
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        '(Ncen, Nsat)' for the halo.
    """
    logM = np.log10(M)

    # ---- central first ----
    if hodshape == 1:
        Ncen = HOD_erf(logM, mu, sig, Ac, rng)
    elif hodshape == 2:
        Ncen = HOD_gauss(logM, mu, sig, Ac, rng)
    else:
        Ncen = HOD_gaussPL(logM, mu, sig, Ac, gamma, rng)

    # ---- satellite mean ----
    mean_sat = satellite_mean_powerlaw(M, M0, M1, alpha, As)

    if conformity:
        if Ncen == 1:
            mean_sat *= K1_global
        else:
            mean_sat *= K2_global

    Nsat = sample_satellite_occupation(mean_sat, beta, rng)

    return Ncen, Nsat


@jit(nopython=True)
def calculate_hod_occupation_fast(
    logM,
    mu,
    Ac,
    As,
    alpha,
    sig,
    gamma,
    M0,
    M1,
    hodshape,
    beta,
    conformity,
    K1_global,
    K2_global,
    rng
):
    
    """
    Draw the full HOD occupation for a single halo using 'logM' directly.

    Parameters
    ----------
    logM : float
        Base-10 halo mass.
    mu, Ac, As, alpha, sig, gamma, M0, M1 : float
        Analytical HOD parameters.
    hodshape : int
        Identifier of the central-occupation model.
    beta : float
        Dispersion-control parameter for the satellite count distribution.
    conformity : bool
        Whether to rescale the satellite mean according to the central outcome.
    K1_global, K2_global : float
        Conformity rescaling factors.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        '(Ncen, Nsat)' for the halo.
    """
    # ---- central first ----
    if hodshape == 1:
        Ncen = HOD_erf(logM, mu, sig, Ac, rng)
    elif hodshape == 2:
        Ncen = HOD_gauss(logM, mu, sig, Ac, rng)
    else:
        Ncen = HOD_gaussPL(logM, mu, sig, Ac, gamma, rng)

    # ---- satellite mean ----
    M = 10.0 ** logM
    mean_sat = satellite_mean_powerlaw(M, M0, M1, alpha, As)
    if conformity:
        if Ncen == 1:
            mean_sat = mean_sat * K1_global
        else:
            mean_sat = mean_sat * K2_global

    Nsat = sample_satellite_occupation(mean_sat, beta, rng)

    return Ncen, Nsat

@njit
def compute_hod_arrays(
    logM,
    mu, Ac, As, alpha, sig, gamma,
    M0, M1, hodshape, beta,
    conformity, K1_global, K2_global,
    rng
):
    """
    Apply the analytical HOD model to an array of haloes.

    Parameters
    ----------
    logM : ndarray
        Array of halo masses in log10 units.
    mu, Ac, As, alpha, sig, gamma, M0, M1 : float
        Analytical HOD parameters.
    hodshape : int
        Identifier of the central-occupation model.
    beta : float
        Dispersion-control parameter for the satellite count distribution.
    conformity : bool
        Whether to apply conformity rescaling.
    K1_global, K2_global : float
        Conformity rescaling factors.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        '(Ncen, Nsat, has_gal)', where:
        - 'Ncen' is the central-occupation array,
        - 'Nsat' is the satellite-occupation array,
        - 'has_gal' flags haloes hosting at least one galaxy.
    """
    n = logM.size
    Ncen = np.zeros(n, dtype=np.int8)
    Nsat = np.zeros(n, dtype=np.int32)
    has_gal = np.zeros(n, dtype=np.int8)

    for i in range(n):
        nc, ns = calculate_hod_occupation_fast(
            logM[i],
            mu, Ac, As, alpha, sig, gamma,
            M0, M1, hodshape, beta,
            conformity, K1_global, K2_global,
            rng
        )
        Ncen[i] = nc
        Nsat[i] = ns
        if nc + ns > 0:
            has_gal[i] = 1

    return Ncen, Nsat, has_gal