"""
Random sampling functions for HOD models
"""

from numba import jit
import numpy as np
import math

import src.hod_const as c

@jit(nopython=True)
def rand_gauss():
    """Generate Gaussian (Box-Muller) random number"""
    v1 = 2.0 * np.random.random() - 1.0
    v2 = 2.0 * np.random.random() - 1.0
    s = v1*v1 + v2*v2
    
    while s >= 1.0:
        v1 = 2.0 * np.random.random() - 1.0
        v2 = 2.0 * np.random.random() - 1.0
        s = v1*v1 + v2*v2
    
    if s == 0.0:
        return 0.0
    else:
        return v1 * math.sqrt(-2.0 * math.log(s) / s)

    
@jit(nopython=True)
def factorial_float(f):
    """Calculate factorial returning float (for use in probability calculations)"""
    if f == 0:
        return 1.0
    result = 1.0
    for i in range(1, int(f) + 1):
        result *= float(i)
    return result


@jit(nopython=True)
def poisson_pdf(lam, k):
    """
    Calculate the Poisson probability distribution function.
    
    Parameters
    ----------
    k : int
        Number of occurrences (must be non-negative)
    lam : float
        Mean (lambda, must be positive)
    
    Returns
    -------
    float
        Probability P(X = k) for the Poisson PDF
    """
    if k < 0 or lam <= 0:
        return 0.0
    
    return math.pow(lam, k) * math.exp(-lam) / factorial_float(k)

'''
@jit(nopython=True)
def poisson_sample(lam):
    """Sample from a Poisson PDF"""
    k = 0
    prob = 0.0
    r = np.random.random()
    
    while prob < 0.999999999999999:
        prob += poisson_pdf(lam, k)
        if r < prob:
            return k
        k += 1
        if k > c.chunk_size:  # Safety break to prevent infinite loops
            break
    
    return k
'''
@jit(nopython=True)
def poisson_sample(lam):
    """Sample from Poisson distribution"""
    k = 0
    prob = 0.0
    r = np.random.random()
    
    while prob < 0.999999999999999:
        prob += math.pow(lam, k) * math.exp(-lam) / factorial_float(k)
        if r < prob:
            return k
        k += 1
        if k > c.chunk_size:  # Safety break to prevent infinite loops
            break
    
    return k


@jit(nopython=True)
def next_integer(x):
    """Sample integer based on fractional part - direct translation from C"""
    low = int(math.floor(x))
    rand01 = np.random.random()
    
    if rand01 > (x - low):
        return low
    else:
        return low + 1

    
@jit(nopython=True)
def product_gamma(a, b):
    """Calculate product from b to a-1 (avoiding gamma function overflow)"""    
    c = int(round(a - b))
    s = 1.0
    for j in range(c + 1):
        s *= (j + b)    
    return s    


@jit(nopython=True)
def neg_binomial_pdf(lam, k, beta):
    """
    Calculate the Negative Binomial probability distribution function.
    
    Parameters
    ----------
    k : int
        Number of failures (must be non-negative)
    beta : float
        Dispersion parameter
    
    Returns
    -------
    float
        Probability P(X = k) for Negative Binomial PDF
    """
    if k < 0:
        return 0.0
    
    q = 1.0 / beta
    p = q / (q + lam)
    
    # Use product function to avoid gamma function overflow
    prob = (product_gamma(k + q - 1, q) / factorial_float(k) * 
            math.pow(p, q) * math.pow(1 - p, k))
    
    return prob

'''
@jit(nopython=True)
def neg_binomial_pdf2(lam, k, beta):
    """Negative Binomial PDF using log-gamma to avoid overflow."""
    if k < 0 or beta <= 0:
        return 0.0
    q = 1.0 / beta
    p = 1.0 / (1.0 + beta * lam)

    # log(P) = log Γ(k+r) - log Γ(r) - log k! + r log p + k log(1-p)
    logP = math.lgamma(k + q) - math.lgamma(q) - math.lgamma(k + 1) \
           + q * math.log(p) + k * math.log(1 - p)

    return math.exp(logP)
'''


@jit(nopython=True)
def neg_binomial_sample(lam, beta):
    """Sample from negative binomial distribution"""
    P = 0.0
    k = -1
    rand01 = np.random.random()
    
    while P < rand01:
        k += 1
        prob_term = neg_binomial_pdf(lam, k, beta)
        P += prob_term
        
        if k > c.chunk_size:  # Safety break
            break
    return k


@jit(nopython=True)
def g0(mean, b):
    """Helper function g0 for extended binomial"""
    result = 0.0
    if b >= 0:
        for i in range(b + 1):
            result += math.pow(-mean, i) / factorial_float(i)
    return result


@jit(nopython=True)
def betas_func(i, beta):
    """Helper function betas for extended binomial"""
    result2 = 1.0
    for j in range(1, i + 1):
        result2 *= (j * beta + 1)
    return result2


@jit(nopython=True)
def gi(i, Nsat, mean, beta):
    """Helper function gi for extended binomial"""
    if i + 1 - Nsat < 0:
        return 0.0
    else:
        return (math.pow(-1, i + 1 - Nsat) / factorial_float(i + 1 - Nsat) * 
                math.pow(mean, i + 1 - Nsat) * betas_func(i, beta))

    
@jit(nopython=True)
def gn(i, Nsat, mean, beta):
    """Helper function gn for extended binomial"""
    res = 0.0
    for j in range(1, i):
        res += gi(j, Nsat, mean, beta)
    return res

@jit(nopython=True)
def f(Nsat, beta, mean):
    """Helper function f for extended binomial"""
    q = int(math.ceil(-1.0 / beta))
    if q > (mean + 1) and Nsat < q + 0.01 and beta < -0.3333334:
        numerator = (math.pow(q, Nsat) * factorial_float(q - Nsat) / factorial_float(q) * 
                    (g0(mean, 1 - Nsat) + gn(q, Nsat, mean, beta)))
        denominator = math.pow(1 - mean / q, q - Nsat)
        return numerator / denominator
    else:
        return 1.0

    
@jit(nopython=True)
def n_func(y, z):
    """Helper function n for extended binomial"""
    q = int(math.ceil(1.0 / z))
    trunc_val = int(math.trunc(y + 1.0))
    if q >= trunc_val:
        return float(q)
    else:
        return float(trunc_val)

    
@jit(nopython=True)
def p_func(y, z):
    """Helper function p for extended binomial"""
    q = int(math.ceil(1.0 / z))
    trunc_val = int(math.trunc(y + 1.0))
    if q >= trunc_val:
        return y / q
    else:
        return y / trunc_val

    
@jit(nopython=True)
def binomial_sample(x, beta):
    """Sample from extended binomial distribution"""
    a = -beta
    P = 0.0
    N = -1
    rand01 = np.random.random()
    
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
