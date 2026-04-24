# hod_shape.py

"""
Random sampling utilities for HOD occupation models.

Overview
--------
This module contains the low-level stochastic machinery used to convert mean
halo occupations into discrete galaxy counts. It provides:

1. Basic random variate generation:
   - Gaussian deviates
   - Poisson counts
   - negative-binomial counts
   - sub-Poisson/binomial-like counts
   - nearest-integer sampling

2. Auxiliary combinatorial and probability helpers needed to evaluate the
   custom count distributions.

3. Fast Numba-compiled routines that apply the occupation model to arrays of
   halos in one pass.

Role in the pipeline
--------------------
The analytical HOD modules typically compute mean occupations such as

    <N_cen>(M),   <N_sat>(M),

but mock generation requires discrete realizations for each halo. This module
implements that stochastic step.

Interpretation of 'beta'
------------------------
The parameter 'beta' controls the counting distribution used for satellites:

- 'beta < -1'
    nearest-integer-like assignment

- '-1 <= beta < -1/171'
    sub-Poisson binomial-like sampling

- '-1/171 <= beta <= 0'
    Poisson sampling

- 'beta > 0'
    negative-binomial sampling

This convention is used consistently throughout the HOD pipeline.

Implementation
--------------
Most routines are decorated with Numba so they can be called efficiently from
the inner halo-processing loops.
"""

from numba import jit
from numba import njit
import numpy as np
import math

import src.hod_madrid_py.hod_const as c

@jit(nopython=True)
def rand_gauss(rng):
    """ 
    Draw a standard normal deviate using the Box-Muller method.

    Parameters
    ----------
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    float
        A single Gaussian random number with mean 0 and variance 1.
    """
    v1 = 2.0 * rng.random() - 1.0
    v2 = 2.0 * rng.random() - 1.0
    s = v1*v1 + v2*v2
    
    while s >= 1.0:
        v1 = 2.0 * rng.random() - 1.0
        v2 = 2.0 * rng.random() - 1.0
        s = v1*v1 + v2*v2
    
    if s == 0.0:
        return 0.0
    else:
        return v1 * math.sqrt(-2.0 * math.log(s) / s)

    
@jit(nopython=True)
def factorial_float(f):
    """
    Compute the factorial of a non-negative integer and return it as a float.

    Parameters
    ----------
    f : int or float
        Input value. In practice this is assumed to represent a non-negative
        integer.

    Returns
    -------
    float
        Factorial of 'f', returned in floating-point form.
    """
    if f == 0:
        return 1.0
    result = 1.0
    for i in range(1, int(f) + 1):
        result *= float(i)
    return result

@jit(nopython=True)
def poisson_sample(lam, rng):
    """
    Draw a Poisson-distributed random count by cumulative summation of the PMF.

    Parameters
    ----------
    lam : float
        Mean count parameter.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    int
        A Poisson-distributed random integer.
    """
    k = 0
    prob = 0.0
    r = rng.random()
    
    while prob < 0.999999999999999:
        prob += math.pow(lam, k) * math.exp(-lam) / factorial_float(k)
        if r < prob:
            return k
        k += 1
        if k > c.chunk_size:  # Safety break to prevent infinite loops
            break
    
    return k


@jit(nopython=True)
def next_integer(x, rng):
    """
    Draw the nearest-integer stochastic rounding of a real number.

    Parameters
    ----------
    x : float
        Expected count.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    int
        Either 'floor(x)' or 'floor(x) + 1', with probabilities chosen so that
        the expectation value is 'x'.
    """
    low = int(math.floor(x))
    rand01 = rng.random()
    
    if rand01 > (x - low):
        return low
    else:
        return low + 1

    
@jit(nopython=True)
def product_gamma(a, b):
    """
    Compute a finite product used in the negative-binomial PMF.

    Parameters
    ----------
    a : float
        Upper argument of the product.
    b : float
        Lower argument of the product.

    Returns
    -------
    float
        Product over the required range, used as a gamma-function surrogate.
    """    
    c = int(round(a - b))
    s = 1.0
    for j in range(c + 1):
        s *= (j + b)    
    return s    


@jit(nopython=True)
def neg_binomial_pdf(lam, k, beta):
    """
    Evaluate the negative-binomial probability mass function.

    Parameters
    ----------
    lam : float
        Mean count parameter.
    k : int
        Non-negative integer count.
    beta : float
        Over-dispersion parameter. Only positive values are meaningful in the
        intended usage of this function.

    Returns
    -------
    float
        Probability 'P(X = k)' for the corresponding negative-binomial model.
    """
    if k < 0:
        return 0.0
    
    q = 1.0 / beta
    p = q / (q + lam)
    
    # Use product function to avoid gamma function overflow
    prob = (product_gamma(k + q - 1, q) / factorial_float(k) * 
            math.pow(p, q) * math.pow(1 - p, k))
    
    return prob

@jit(nopython=True)
def neg_binomial_sample(lam, beta, rng):
    """
    Draw a negative-binomial random count by cumulative summation of the PMF.

    Parameters
    ----------
    lam : float
        Mean count parameter.
    beta : float
        Over-dispersion parameter.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    int
        A negative-binomially distributed random integer.
    """
    P = 0.0
    k = -1
    rand01 = rng.random()
    
    while P < rand01:
        k += 1
        prob_term = neg_binomial_pdf(lam, k, beta)
        P += prob_term
        
        if k > c.chunk_size:  # Safety break
            break
    return k


@jit(nopython=True)
def g0(mean, b):
    """
    Auxiliary series used in the extended binomial construction.

    Parameters
    ----------
    mean : float
        Mean occupation parameter.
    b : int
        Integer controlling the upper limit of the series.

    Returns
    -------
    float
        Value of the auxiliary sum.
    """
    result = 0.0
    if b >= 0:
        for i in range(b + 1):
            result += math.pow(-mean, i) / factorial_float(i)
    return result


@jit(nopython=True)
def betas_func(i, beta):
    """
    Auxiliary multiplicative factor used in the extended binomial model.

    Parameters
    ----------
    i : int
        Integer index.
    beta : float
        Dispersion parameter.

    Returns
    -------
    float
        Value of the product entering the extended binomial construction.
    """
    result2 = 1.0
    for j in range(1, i + 1):
        result2 *= (j * beta + 1)
    return result2


@jit(nopython=True)
def gi(i, Nsat, mean, beta):
    """
    Auxiliary term of the extended binomial expansion.

    Parameters
    ----------
    i : int
        Series index.
    Nsat : int
        Satellite count.
    mean : float
        Mean occupation parameter.
    beta : float
        Dispersion parameter.

    Returns
    -------
    float
        Value of the corresponding auxiliary term.
    """
    if i + 1 - Nsat < 0:
        return 0.0
    else:
        return (math.pow(-1, i + 1 - Nsat) / factorial_float(i + 1 - Nsat) * 
                math.pow(mean, i + 1 - Nsat) * betas_func(i, beta))

    
@jit(nopython=True)
def gn(i, Nsat, mean, beta):
    """
    Partial sum of auxiliary terms entering the extended binomial model.

    Parameters
    ----------
    i : int
        Upper index of the sum.
    Nsat : int
        Satellite count.
    mean : float
        Mean occupation parameter.
    beta : float
        Dispersion parameter.

    Returns
    -------
    float
        Partial sum of the auxiliary series.
    """
    res = 0.0
    for j in range(1, i):
        res += gi(j, Nsat, mean, beta)
    return res

@jit(nopython=True)
def f(Nsat, beta, mean):
    """
    Correction factor used in the extended binomial probability.

    Parameters
    ----------
    Nsat : int
        Satellite count.
    beta : float
        Dispersion parameter.
    mean : float
        Mean occupation parameter.

    Returns
    -------
    float
        Multiplicative correction factor entering the binomial-like distribution.
    """
    q = int(math.ceil(-1.0 / beta))
    if q > (mean + 1) and Nsat < q + 0.01 and beta < -1.0/171.0 and mean >= 1.0:
        numerator = (math.pow(q, Nsat) * factorial_float(q - Nsat) / factorial_float(q) * 
                    (g0(mean, 1 - Nsat) + gn(q, Nsat, mean, beta)))
        denominator = math.pow(1 - mean / q, q - Nsat)
        return numerator / denominator
    else:
        return 1.0

    
@jit(nopython=True)
def n_func(y, z):
    """
    Return the effective number of Bernoulli trials used in the extended
    binomial model.

    Parameters
    ----------
    y : float
        Mean occupation parameter.
    z : float
        Positive quantity related to '-beta'.

    Returns
    -------
    float
        Effective number of trials.
    """
    q = int(math.ceil(1.0 / z))
    trunc_val = int(math.trunc(y + 1.0))
    if q >= trunc_val:
        return float(q)
    else:
        return float(trunc_val)

    
@jit(nopython=True)
def p_func(y, z):
    """
    Return the effective success probability used in the extended binomial model.

    Parameters
    ----------
    y : float
        Mean occupation parameter.
    z : float
        Positive quantity related to '-beta'.

    Returns
    -------
    float
        Effective success probability.
    """
    q = int(math.ceil(1.0 / z))
    trunc_val = int(math.trunc(y + 1.0))
    if q >= trunc_val:
        return y / q
    else:
        return y / trunc_val

    
@jit(nopython=True)
def binomial_sample(x, beta, rng):
    """
    Draw a random count from the extended binomial-like distribution.

    Parameters
    ----------
    x : float
        Mean occupation parameter.
    beta : float
        Dispersion parameter in the sub-Poisson regime.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    int
        Random integer drawn from the extended binomial-like model.
    """
    a = -beta
    P = 0.0
    N = -1
    rand01 = rng.random()
    
    n_val = n_func(x, a)
    p_val = p_func(x, a)
    
    while P < rand01:
        N += 1
        f_val = f(N, beta, x)
        
        # Calculate binomial probability term
        binom_coeff = (factorial_float(int(n_val)) / 
                      (factorial_float(int(n_val) - N) * factorial_float(N)))
        prob_term = (f_val * binom_coeff * 
                    math.pow(p_val, N) * math.pow(1 - p_val, n_val - N))
        
        P += prob_term
        
        if N > c.chunk_size:  # Safety break
            break
    
    return N

@njit
def hod_occupation_numba_basic(
    halo_logM,
    M_max,
    Ncen_mean,
    Nsat_mean,
    beta,
    conformity,
    K1_global,
    K2_global,
    rng
):
    """
    Draw central and satellite occupations for a single halo using tabulated
    binned HOD inputs.

    Parameters
    ----------
    halo_logM : float
        Halo mass in log10 units.
    M_max : ndarray
        Upper edges of the tabulated mass bins.
    Ncen_mean : ndarray
        Mean central occupation per bin.
    Nsat_mean : ndarray
        Mean satellite occupation per bin.
    beta : float
        Dispersion parameter controlling the satellite counting distribution.
    conformity : bool
        Whether to apply conformity rescaling to the satellite mean.
    K1_global, K2_global : float
        Global conformity factors applied depending on whether a central is
        present.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        '(Ncen, Nsat)' for the halo.
    """
    # --------------------------------------------------
    # Find mass bin
    # --------------------------------------------------
    k = np.searchsorted(M_max, halo_logM)

    if k == len(M_max):
        return 0, 0

    mean_Ncen = Ncen_mean[k]
    mean_Nsat = Nsat_mean[k]

    # --------------------------------------------------
    # Central galaxy (Bernoulli)
    # --------------------------------------------------
    if rng.random() < mean_Ncen:
        Ncen = 1
        if conformity:
            mean_Nsat_adj = mean_Nsat * K1_global
        else:
            mean_Nsat_adj = mean_Nsat
    else:
        Ncen = 0
        if conformity:
            mean_Nsat_adj = mean_Nsat * K2_global
        else:
            mean_Nsat_adj = mean_Nsat

    # --------------------------------------------------
    # Satellite sampling 
    # --------------------------------------------------
    if mean_Nsat_adj < 0.0:
        Nsat = 0

    elif beta < -1.0:
        Nsat = next_integer(mean_Nsat_adj, rng)

    elif beta <= 0.0 and beta >= -1.0 / 171.0:
        Nsat = poisson_sample(mean_Nsat_adj, rng)

    elif beta < -1.0 / 171.0 and beta >= -1.0:
        Nsat = binomial_sample(mean_Nsat_adj, beta, rng)

    else:
        # beta > 0 → negative binomial
        Nsat = neg_binomial_sample(mean_Nsat_adj, beta, rng)

    return Ncen, Nsat

@njit
def compute_hod_arrays_binned(
    logM,
    M_max,
    Ncen_mean,
    Nsat_mean,
    beta,
    conformity,
    K1_global,
    K2_global,
    rng
):
    """
    Apply the tabulated HOD occupation model to an array of halos.

    Parameters
    ----------
    logM : ndarray
        Halo masses in log10 units.
    M_max : ndarray
        Upper edges of the tabulated mass bins.
    Ncen_mean : ndarray
        Mean central occupation per bin.
    Nsat_mean : ndarray
        Mean satellite occupation per bin.
    beta : float
        Dispersion parameter controlling the satellite counting distribution.
    conformity : bool
        Whether to apply conformity rescaling to the satellite mean.
    K1_global, K2_global : float
        Global conformity factors.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        '(Ncen, Nsat, has_gal)', where:
        - 'Ncen' is the array of central occupations,
        - 'Nsat' is the array of satellite occupations,
        - 'has_gal' flags halos hosting at least one galaxy.
    """
    n = logM.size
    Ncen = np.zeros(n, dtype=np.int8)
    Nsat = np.zeros(n, dtype=np.int32)
    has_gal = np.zeros(n, dtype=np.int8)

    for i in range(n):
        nc, ns = hod_occupation_numba_basic(
            logM[i],
            M_max,
            Ncen_mean,
            Nsat_mean,
            beta,
            conformity,
            K1_global,
            K2_global,
            rng
        )
        Ncen[i] = nc
        Nsat[i] = ns
        if nc + ns > 0:
            has_gal[i] = 1

    return Ncen, Nsat, has_gal