# hod_r_profile.py

"""
Radial-profile utilities for satellite placement in HOD mock catalogues.

Overview
--------
This module provides the machinery used to assign 3D spatial offsets to
satellite galaxies inside host haloes. It supports three conceptually distinct
routes:

1. Standard NFW sampling
   Satellites are drawn from a truncated Navarro-Frenk-White radial profile,
   using either halo-by-halo concentrations read from the input catalogue or
   concentrations estimated from a mass-redshift relation.

2. Extended analytical radial profile
   Satellites are drawn from a fitted radial law of the form

       N(r) = N0 (r / r0)^alpha [1 + (r / r0)^beta]^kappa,

   normalized over the finite interval '[0, Rmax]'.

3. Empirical radial sampling
   Satellites are drawn from a tabulated radial histogram read from an HDF5
   file or other external source.

Role in the pipeline
--------------------
The functions defined here are used when converting halo occupations into
actual satellite positions. The output of the sampling routines is always a
Cartesian offset '(Dx, Dy, Dz)' relative to the host-halo centre.

Conventions
-----------
- Distances are expressed in comoving Mpc/h inside the pipeline.
- Halo mass is expressed in Msun/h.
- If 'Rvir' and 'Rs' are read from external catalogues, they are assumed to
  have already been converted to Mpc/h by the I/O layer before being passed to
  this module.
- The returned offsets are always isotropically oriented in 3D.
"""

import numpy as np
from numba import jit
import src.hod_madrid_py.hod_const as c
from src.hod_madrid_py.hod_cosmology import Delta_vir
import h5py


@jit(nopython=True)
def f_nfw(x):
    """
    Evaluate the dimensionless cumulative-mass kernel of the NFW profile.

    Parameters
    ----------
    x : float or ndarray
        Dimensionless radius defined as 'x = r / rs'.

    Returns
    -------
    float or ndarray
        Value of

            ln(1 + x) - x / (1 + x),

        which is proportional to the enclosed mass of the NFW profile.
    """
    return np.log(1.0 + x) - x/(1.0 + x)

@jit(nopython=True)
def Rvir_from_mass(M, Delta_vir, rho_crit):
    """
    Compute the virial radius corresponding to a halo mass.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    Delta_vir : float
        Virial overdensity relative to the critical density.
    rho_crit : float
        Critical density in Msun / (Mpc/h)^3.

    Returns
    -------
    float
        Virial radius in Mpc/h.
    """
    return (3.0*M / (4.0*np.pi*Delta_vir*rho_crit))**(1.0/3.0)


@jit(nopython=True)
def concentration_klypin(M, z):
    """
    Estimate halo concentration from the Klypin-style mass-redshift relation.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    z : float
        Redshift.

    Returns
    -------
    float
        Estimated concentration parameter.
    """
    if z < 0.25:
        C0, gamma, M0 = 9.5, 0.09, 3.0e5 * 1.0e12
    elif z < 0.75:
        C0, gamma, M0 = 6.75, 0.088, 5000 * 1.0e12
    elif z < 1.22:
        C0, gamma, M0 = 5.0, 0.086, 450 * 1.0e12
    else:
        C0, gamma, M0 = 4.05, 0.085, 90.0 * 1.0e12
    
    return C0 * (M / 1.0e12)**(-gamma) * (1.0 + (M / M0)**0.4)

@jit(nopython=True)
def I_NFW(x):
    """
    Evaluate the integral kernel used for inverse sampling of the NFW profile.

    Parameters
    ----------
    x : float
        Dimensionless radius 'x = r / rs'.

    Returns
    -------
    float
        Value of the integral kernel associated with the enclosed NFW mass.
    """
    return (1.0 / (1.0 + x) + np.log(1.0 + x) - 1.0)


@jit(nopython=True)
def concentration_from_radii(Rvir, Rs):
    """
    Compute halo concentration from virial and scale radii.

    Parameters
    ----------
    Rvir : float
        Virial radius in Mpc/h.
    Rs : float
        Scale radius in Mpc/h.

    Returns
    -------
    float
        Concentration 'c = Rvir / Rs', or 'NaN' if the input is invalid.
    """
    if Rs <= 0.0 or not np.isfinite(Rs):
        return np.nan
    else:
        return Rvir / Rs


@jit(nopython=True)
def get_base_concentration(M, zsnap, Rvir, Rs, read_concentrations=True):
    """
    Return the baseline concentration used by the NFW sampler.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    zsnap : float
        Snapshot redshift.
    Rvir : float
        Virial radius in Mpc/h.
    Rs : float
        Scale radius in Mpc/h.
    read_concentrations : bool, optional
        If True, attempt to read concentration from '(Rvir, Rs)'. Otherwise use
        the analytic concentration-mass relation.

    Returns
    -------
    float
        Baseline halo concentration.
    """
    if read_concentrations:
        c_base = concentration_from_radii(Rvir, Rs)
        if np.isfinite(c_base) and c_base > 0.0:
            return c_base

    else:
        return concentration_klypin(M, zsnap)


@jit(nopython=True)
def get_effective_concentration(M, zsnap, Rvir, Rs, K, read_concentrations):
    """
    Return the effective concentration used for standard NFW sampling.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    zsnap : float
        Snapshot redshift.
    Rvir : float
        Virial radius in Mpc/h.
    Rs : float
        Scale radius in Mpc/h.
    K : float
        Multiplicative concentration-rescaling factor.
    read_concentrations : bool
        Whether to use halo-by-halo concentrations read from the catalogue.

    Returns
    -------
    float
        Effective concentration after rescaling by 'K'.
    """
    c_base = get_base_concentration(M, zsnap, Rvir, Rs, read_concentrations)
    c_eff = K * c_base
    if not np.isfinite(c_eff) or c_eff <= 0.0:
        c_eff = c_base

    return c_eff


@jit(nopython=True)
def sample_nfw_radius_from_concentration(c_eff, Rvir, rng):
    """
    Draw a radius from a standard NFW profile truncated at 'Rvir'.

    Parameters
    ----------
    c_eff : float
        Effective halo concentration.
    Rvir : float
        Virial radius in Mpc/h.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    float
        Random radius in Mpc/h.
    """
    x_max = c_eff
    I_max = I_NFW(x_max)
    y_rand = rng.random() * I_max

    tol = 1.0e-4 * I_max
    low = 0.0
    high = x_max
    mid = 0.5 * (low + high)
    y_try = I_NFW(mid)

    while abs(y_try - y_rand) >= tol:
        if y_try > y_rand:
            high = mid
        else:
            low = mid
        mid = 0.5 * (low + high)
        y_try = I_NFW(mid)
    return mid * Rvir / c_eff


@jit(nopython=True)
def generate_nfw_position_from_concentration(Rvir, c_eff, rng):
    """
    Draw an isotropic 3D satellite offset from a standard NFW profile.

    Parameters
    ----------
    Rvir : float
        Virial radius in Mpc/h.
    c_eff : float
        Effective halo concentration.
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    tuple
        Cartesian offset '(Dx, Dy, Dz)' in Mpc/h.
    """
    R = sample_nfw_radius_from_concentration(c_eff, Rvir, rng)

    phi = rng.random() * 2.0 * np.pi
    costh = rng.random() * 2.0 - 1.0
    sinth = np.sqrt(1.0 - costh * costh)

    Dx = R * sinth * np.cos(phi)
    Dy = R * sinth * np.sin(phi)
    Dz = R * costh

    return Dx, Dy, Dz

@jit(nopython=True)
def get_effective_Rvir(M, zsnap, omega_M, Rvir):
    """
    Return the virial radius to be used for satellite placement.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    zsnap : float
        Snapshot redshift.
    omega_M : float
        Present-day matter density parameter.
    Rvir : float
        Virial radius from the input catalogue, if available.

    Returns
    -------
    float
        Valid virial radius in Mpc/h. If the supplied 'Rvir' is invalid, it is
        recomputed from the halo mass.
    """
    if np.isfinite(Rvir) and Rvir > 0.0:
        return Rvir
    else:
        Delta = Delta_vir(zsnap, omega_M)
        return Rvir_from_mass(M, Delta, c.rho_crit)

@jit(nopython=True)
def extended_N_of_r(r, r0, alpha, beta, kappa, N0):
    """
    Evaluate the analytical extended radial law.

    Parameters
    ----------
    r : float or ndarray
        Radius in Mpc/h.
    r0 : float
        Characteristic scale radius in Mpc/h.
    alpha : float
        Inner-slope parameter.
    beta : float
        Transition-shape parameter.
    kappa : float
        Outer-shape parameter.
    N0 : float
        Overall normalization.

    Returns
    -------
    float or ndarray
        Value of the analytical radial law at 'r'.
    """
    x = np.clip(r / r0, 1e-300, None)  # numeric safety for alpha < 0
    return N0 * (x**alpha) * (1.0 + x**beta)**kappa

def _build_cdf(Rmax, r0, alpha, beta, kappa, N0, ngrid=2048):
    """
    Tabulate and normalize the cumulative radial distribution of the extended
    analytical profile on the interval '[0, Rmax]'.

    Parameters
    ----------
    Rmax : float
        Maximum radius in Mpc/h.
    r0, alpha, beta, kappa, N0 : float
        Parameters of the analytical radial law.
    ngrid : int, optional
        Number of tabulation points.

    Returns
    -------
    tuple
        '(r_grid, cdf)', where 'cdf' is normalized to unity at 'Rmax'.

    Raises
    ------
    ValueError
        If the integrated radial law is non-positive.
    """
    r = np.linspace(0.0, Rmax, ngrid)
    if r[0] == 0.0:
        r[0] = 1e-12 * (r0 if r0 > 0 else Rmax)

    f = extended_N_of_r(r, r0, alpha, beta, kappa, N0)
    dr = np.diff(r)

    F = np.empty_like(r)
    F[0] = 0.0
    F[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * dr)  # trapezoidal integral
    if F[-1] <= 0.0:
        raise ValueError("Non-positive integral of N(r). Check parameters.")

    cdf = F / F[-1]
    return r, cdf


def _sample_r_from_cdf(r_grid, cdf, rng):
    """
    Draw a radius from a tabulated cumulative radial distribution.

    Parameters
    ----------
    r_grid : ndarray
        Radius grid in Mpc/h.
    cdf : ndarray
        Monotonic cumulative distribution evaluated on 'r_grid'.
    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    float
        Random radius in Mpc/h.
    """
    u = rng.random()
    j = np.searchsorted(cdf, u, side="right")
    j = np.clip(j, 1, len(cdf)-1)
    c0, c1 = cdf[j-1], cdf[j]
    r0, r1 = r_grid[j-1], r_grid[j]
    t = (u - c0) / (c1 - c0)
    return r0 + t*(r1 - r0)

@jit(nopython=True)
def _random_unit_vector(rng):
    """
    Draw an isotropically distributed unit vector.

    Parameters
    ----------
    rng : numpy.random.Generator-like
        Random-number generator compatible with Numba.

    Returns
    -------
    ndarray
        Three-component unit vector.
    """
    phi = rng.uniform(0.0, 2.0*np.pi)
    mu  = rng.uniform(-1.0, 1.0)
    s   = np.sqrt(1.0 - mu*mu)
    return np.array([s*np.cos(phi), s*np.sin(phi), mu])

def generate_extended_position(
    M, zsnap, omega_M, rho_crit, Delta_vir_func, Rvir, Rmax,
    r0, alpha, beta, kappa, N0,
    rng=None, ngrid=4096
):
    """
    Draw an isotropic 3D satellite offset from the extended analytical radial
    profile.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h.
    zsnap : float
        Snapshot redshift.
    omega_M : float
        Present-day matter density parameter.
    rho_crit : float
        Critical density in Msun / (Mpc/h)^3.
    Delta_vir_func : callable
        Function returning the virial overdensity as a function of redshift and
        cosmology.
    Rvir : float
        Virial radius in Mpc/h, if available.
    Rmax : float
        Maximum radius in Mpc/h up to which the profile is normalized.
    r0, alpha, beta, kappa, N0 : float
        Parameters of the extended analytical profile.
    rng : numpy.random.Generator, optional
        Random-number generator.
    ngrid : int, optional
        Number of tabulation points used for the CDF.

    Returns
    -------
    tuple
        Cartesian offset '(Dx, Dy, Dz)' in Mpc/h.

    Raises
    ------
    ValueError
        If 'Rmax' is not a positive finite number.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Use halo Rvir if available; otherwise compute it from mass
    if Rvir is None or not np.isfinite(Rvir) or Rvir <= 0.0:
        Delta = Delta_vir_func(zsnap, omega_M)
        Rvir = Rvir_from_mass(M, Delta, rho_crit)

    # Truncation radius
    #Rmax = Rvir / max(K_trunc, 1e-6)
    #Rmax = 1.5
    if Rmax is None or not np.isfinite(Rmax) or Rmax <= 0.0:
        raise ValueError("Rmax must be a positive finite number for the extended radial profile.")

    # Build CDF and sample radius
    r_grid, cdf = _build_cdf(Rmax, r0, alpha, beta, kappa, N0, ngrid=ngrid)
    r = _sample_r_from_cdf(r_grid, cdf, rng)

    # Isotropic direction
    nhat = _random_unit_vector(rng)
    Dx, Dy, Dz = (r * nhat).tolist()
    return Dx, Dy, Dz


def load_radial_histogram_from_h5(h5file, group='data'):
    """
    Load an empirical radial histogram from an HDF5 file.

    Parameters
    ----------
    h5file : str or Path
        Path to the HDF5 file.
    group : str, optional
        Group containing the radial datasets.

    Returns
    -------
    tuple
        '(r_bin_edges, Nsat_r, probs_r)', where:
        - 'r_bin_edges' are the histogram edges,
        - 'Nsat_r' are the bin counts,
        - 'probs_r' are the normalized bin probabilities.
    """
    with h5py.File(h5file, 'r') as f:
        data = f[group]
        r_min = np.array(data['r_min'])  # length N
        r_max = np.array(data['r_max'])  # length N
        Nsat_r = np.array(data['Nsat_r'])  # length N

    # Create bin edges array (length N+1)
    r_bin_edges = np.concatenate([r_min, [r_max[-1]]])
    # Calculate PDF (normalized probabilities)
    probs_r = Nsat_r / np.sum(Nsat_r)

    return r_bin_edges, Nsat_r, probs_r

def sample_radius_from_histogram(bin_edges, probs, size=1, rng=None):
    """
    Draw one or more radii from an empirical histogram.

    Parameters
    ----------
    bin_edges : ndarray
        Histogram bin edges.
    probs : ndarray
        Normalized bin probabilities.
    size : int, optional
        Number of random draws.
    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    float or ndarray
        One sampled radius if 'size == 1', otherwise an array of sampled radii.
    """
    if rng is None:
        rng = np.random.default_rng()
    chosen_bins = rng.choice(len(probs), size=size, p=probs)
    left_edges = bin_edges[chosen_bins]
    right_edges = bin_edges[chosen_bins + 1]
    samples = rng.uniform(left_edges, right_edges)
    return samples if size > 1 else samples[0]

def random_unit_vector(size=1, rng=None):
    """
    Draw one or more isotropically distributed unit vectors.

    Parameters
    ----------
    size : int, optional
        Number of unit vectors to generate.
    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    ndarray
        Array of shape '(size, 3)' if 'size > 1', or shape '(3,)' if 'size == 1'.
    """
    if rng is None:
        rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=size)
    costheta = rng.uniform(-1, 1, size=size)
    sintheta = np.sqrt(1 - costheta**2)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = costheta
    vecs = np.stack((x, y, z), axis=-1)
    return vecs if size > 1 else vecs[0]

def sample_empirical_position(r_bin_edges, r_probs, size=1, rng=None):
    """
    Draw isotropic 3D satellite offsets from an empirical radial histogram.

    Parameters
    ----------
    r_bin_edges : ndarray
        Histogram bin edges in radius.
    r_probs : ndarray
        Normalized radial-bin probabilities.
    size : int, optional
        Number of samples to generate.
    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    ndarray
        Array of sampled Cartesian offsets. The returned shape is '(size, 3)' if
        'size > 1', or '(3,)' if 'size == 1'.
    """
    r = sample_radius_from_histogram(r_bin_edges, r_probs, size=size, rng=rng)
    directions = random_unit_vector(size=size, rng=rng)
    pos = r[:, None] * directions if size > 1 else r * directions
    return pos

