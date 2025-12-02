#hod_shape.py

"""
Average HOD and scatter
"""
import numpy as np
from numba import jit
import math
import src.hod_pdf as pdf
import src.hod_io as io
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
def HOD_erf(logM: float, mu: float, sig: float, As: float) -> int:
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
    rand_val = np.random.random()
    return 1 if rand_val < r else 0

@jit(nopython=True)
def HOD_gauss(logM: float, mu: float, sig: float, As: float) -> int:
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
    - Unlike the sigmoid, the Gaussian’s peak height depends on (As, sig);
      you must ensure the maximum r does not exceed 1, or clamp upstream.
    """
    r = As / (sig * math.sqrt(2.0 * math.pi)) * math.exp(- (logM - mu) ** 2 / (2.0 * sig ** 2))
    rand_val = np.random.random()
    return 1 if rand_val < r else 0

@jit(nopython=True)
def HOD_gaussPL(logM, mu, sig, Ac, gamma):
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

    rand_val = np.random.random()
    return 1 if rand_val < r else 0


@jit(nopython=True)
def HOD_powerlaw(M, M0, M1, alpha, As, beta):
    """
    Draw satellite occupation number from a power-law mean and a flexible count PDF.

    Mean satellite occupation
    -------------------------
        If (M1 <= 0) or (M < M0): return 0
        xsat      = (M - M0) / M1
        mean_sat  = As * xsat^alpha

    Count PDF selection by 'beta'
    -----------------------------
        beta <  -1                → nearest-integer-like draw (pdf.next_integer)
       -1 ≤ beta < -1/171         → binomial-like draw        (pdf.binomial_sample)
    -1/171 ≤ beta ≤ 0             → Poisson draw              (pdf.poisson_sample)
        beta >  0                 → negative binomial draw    (pdf.neg_binomial_sample)

    Parameters
    ----------
    M : float
        Halo mass (linear, Msun/h; NOT log10).
    M0 : float
        Minimum (cut) mass for satellites; below this, returns 0.
    M1 : float
        Characteristic satellite mass scale (normalization of xsat).
    alpha : float
        Power-law slope controlling how ⟨N_sat⟩ grows with mass.
    As : float
        Amplitude for the mean satellite number.
    beta : float
        Dispersion control selecting the counting distribution (see table above).

    Returns
    -------
    int
        Realization of satellite count for this halo (0, 1, 2, ...).

    Notes
    -----
    - This is a *draw* (random integer), not the mean; for the expected value use
      'mean_sat' directly.
    - The PDF helpers ('next_integer', 'poisson_sample', 'binomial_sample',
      'neg_binomial_sample') are provided by 'src.hod_pdf' and must be Numba-compatible.
    - Ensure M1 > 0 and As ≥ 0 upstream; otherwise mean_sat may be invalid.
    """
    if M1 <= 0.0 or M < M0:
        return 0

    xsat = (M - M0) / M1
    mean_sat = As * (xsat ** alpha)
    if mean_sat <= 0.0:
        return 0

    # Select the counting distribution based on beta
    if beta < -1.0:
        return pdf.next_integer(mean_sat)
    elif beta <= 0.0 and beta >= -1.0 / 171.0:
        return pdf.poisson_sample(mean_sat)
    elif beta < -1.0 / 171.0 and beta >= -1.0:
        return pdf.binomial_sample(mean_sat, beta)
    elif beta > 0.0:
        return pdf.neg_binomial_sample(mean_sat, beta)
    else:
        # Fallback (defensive): Poisson
        return pdf.poisson_sample(mean_sat)

def HOD_shape_file(M, beta, filename):
    """
    Draw satellite occupation using a binned HOD shape read from an HDF5 file.

    This function:
      1) Reads mass bins and mean occupations from `filename` (HDF5).
      2) Finds the bin where M_min[i] <= M < M_max[i].
      3) Uses the per-bin mean satellite occupation (Nsat_mean[i]) to draw an
         integer count according to the dispersion selector `beta`:
           - beta <  -1                → nearest-integer-like (pdf.next_integer)
           - -1 <= beta < -1/171       → binomial-like        (pdf.binomial_sample)
           - -1/171 <= beta <= 0       → Poisson              (pdf.poisson_sample)
           - beta >  0                 → negative binomial    (pdf.neg_binomial_sample)

    Parameters
    ----------
    M : float
        Halo mass (linear, NOT log10).
    beta : float
        Dispersion control that selects the counting distribution (see above).
    filename : str
        Path to the HDF5 file (expects group 'data' with M_min, M_max, Ncen, Nsat, N_halo).
        Means are computed inside `io.read_occupation_from_h5` as N/N_halo (safe-divided).

    Returns
    -------
    int
        Realization of satellite count for this halo (0, 1, 2, ...).
        Returns 0 if mass falls outside the binned range or mean ≤ 0.

    Notes
    -----
    - This function performs file I/O on every call. For performance, consider
      preloading:
          M_min, M_max, Ncen_mean, Nsat_mean = io.read_occupation_from_h5(filename)
      y pasar esos arrays a una versión que no lea el archivo en cada llamada.
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
        return pdf.next_integer(mean_n)
    elif (-1.0/171.0) <= beta <= 0.0:
        return pdf.poisson_sample(mean_n)
    elif -1.0 <= beta < (-1.0/171.0):
        return pdf.binomial_sample(mean_n, beta)
    elif beta > 0.0:
        return pdf.neg_binomial_sample(mean_n, beta)
    else:
        # Fallback: Poisson
        return pdf.poisson_sample(mean_n)



@jit(nopython=True)
def get_hod_derived_params(mu: float, hodshape: int) -> Tuple[float, float, float, float, float]:
    """
    Return C-style default derived HOD parameters for a given shape preset.

    Presets (mirror of the original C/ifdef logic)
    ----------------------------------------------
    hodshape == 1 (HOD1):
        M0 = 10^mu
        M1 = 10^(mu + 1.3)
        alpha = 1.0
        sig = 0.15
        gamma = -1.4

    hodshape == 2 (HOD2):
        M0 = 10^(mu - 0.1)
        M1 = 10^(mu + 0.3)
        alpha = 0.8
        sig = 0.12
        gamma = -1.4

    else (HOD3 default):
        M0 = 10^(mu - 0.05)
        M1 = 10^(mu + 0.35)
        alpha = 0.9
        sig = 0.08
        gamma = -1.4

    Parameters
    ----------
    mu : float
        log10 characteristic mass.
    hodshape : int
        HOD shape preset selector (1, 2, or 3).

    Returns
    -------
    (M0, M1, alpha, sig, gamma) : tuple of floats
        Defaults that seed the HOD model parameters.
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
    M: float,
    mu: float,
    Ac: float,
    As: float,
    alpha: float,
    sig: float,
    gamma: float,
    M0: float,
    M1: float,
    hodshape: int,
    beta: float = 0.0,
):
    """
    Draw central and satellite occupation numbers for a single halo mass M.

    This function mirrors the C reference logic:
      - Centrals: one Bernoulli draw (0/1) according to the selected HOD shape:
          hodshape == 1 → HOD_erf    (erf-based sigmoid)
          hodshape == 2 → HOD_gauss  (Gaussian-shaped probability)
          else          → HOD_gaussPL (Gaussian + power-law tail)
      - Satellites: integer draw from a power-law mean with dispersion selector beta
        via `HOD_powerlaw`.

    Parameters
    ----------
    M : float
        Halo mass (linear, Msun/h; NOT log10).
    mu : float
        log10 characteristic mass for the central HOD.
    Ac : float
        Central amplitude/normalization (ensure produces probabilities in [0,1]).
    As : float
        Satellite amplitude for the mean satellite occupation.
    alpha : float
        Satellite power-law slope.
    sig : float
        Central scatter/width parameter (definition depends on HOD shape).
    gamma : float
        High-mass slope for the central HOD (used by HOD_gaussPL / HOD3).
    M0 : float
        Minimum halo mass for satellites (cut below which Nsat = 0).
    M1 : float
        Characteristic satellite mass scale (normalization in the power-law).
    hodshape : int
        Central HOD shape selector:
            1 = erf-sigmoid, 2 = Gaussian, 3/other = Gaussian + power-law.
    beta : float, default 0.0
        Dispersion selector for satellite counts:
            beta <  -1               → nearest-integer-like
           -1 ≤ beta < -1/171        → binomial-like
        -1/171 ≤ beta ≤ 0            → Poisson
            beta >  0                → negative binomial

    Returns
    -------
    (Ncen, Nsat) : tuple of ints
        Ncen ∈ {0,1} is a Bernoulli draw for the central.
        Nsat ∈ {0,1,2,...} is an integer draw for satellites.

    Notes
    -----
    - Randomness: this function performs random draws; for reproducibility, set
      the NumPy seed outside (before JIT compilation / loop).
    - Validity: upstream parameters should ensure resulting central probabilities
      remain within [0,1]. The underlying HOD_* functions cap as needed.
    - Performance: `@jit(nopython=True)` accelerates loops over many halos.
    - Units: M [Msun/h], mu/logM [log10(Msun/h)].
    """
    logM = np.log10(M)

    # Satellites: one random draw from power-law mean with dispersion beta
    Nsat = HOD_powerlaw(M, M0, M1, alpha, As, beta)

    # Centrals: one Bernoulli draw according to the chosen HOD shape
    if hodshape == 1:
        # HOD1: erf-based sigmoid
        Ncen = HOD_erf(logM, mu, sig, Ac)
    elif hodshape == 2:
        # HOD2: Gaussian probability
        Ncen = HOD_gauss(logM, mu, sig, Ac)
    else:
        # HOD3 / default: Gaussian + power-law tail
        Ncen = HOD_gaussPL(logM, mu, sig, Ac, gamma)

    return Ncen, Nsat
