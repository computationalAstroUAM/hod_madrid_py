# hod_v_profile.py

"""
Velocity-profile utilities for satellite galaxies in HOD mock catalogues.

Overview
--------
This module provides the machinery required to assign internal velocity offsets
to satellite galaxies once their host haloes and spatial offsets have already
been determined.

Three complementary routes are implemented:

1. Virial velocity model
   A simple Gaussian model in which each Cartesian component is drawn from a
   zero-mean normal distribution with a dispersion motivated by virial scaling.

2. Empirical velocity sampling
   Radial and tangential velocities are drawn from tabulated histograms,
   typically obtained from external calibration data such as HODfit2sim or
   semi-analytic catalogues.

3. Analytical extended velocity model
   The radial component is sampled from a three-Gaussian mixture, while the
   tangential speed is sampled from an exponential-power-law form. The final
   three-dimensional velocity vector is reconstructed relative to the local
   radial direction.

Role in the pipeline
--------------------
The functions in this module are called after the spatial offset of a satellite
has been sampled. The velocity assignment therefore takes as input either:
- the host halo mass alone, for virial models, or
- the local radial direction, for empirical and analytical anisotropic models.

Conventions
-----------
- Halo masses are expressed in Msun/h.
- Velocities are expressed in km/s.
- The returned quantities '(Dvx, Dvy, Dvz)' are always velocity offsets with
  respect to the bulk velocity of the host halo.
- The radial direction is defined by the satellite position offset relative to
  the halo centre.
"""
import numpy as np
from numba import jit
import math
import src.hod_madrid_py.hod_io as io
from src.hod_madrid_py.hod_cosmology import Delta_vir, E2
from src.hod_madrid_py.hod_pdf import rand_gauss
import h5py

@jit(nopython=True)
def generate_virial_velocity(M: float, zsnap: float, omega_M: float, rng=None):
    """
    Draw a three-dimensional virial velocity offset for a satellite galaxy.

    The one-dimensional velocity dispersion is modelled as

        sigma_1D = 476 x 0.9 x [Delta_vir(z) E(z)^2]^(1/6) x (M / 10^15)^(1/3),

    with the final Cartesian components drawn independently from a Gaussian
    distribution of width 'sigma_1D'.

    Parameters
    ----------
    M : float
        Host halo mass in Msun/h.
    zsnap : float
        Snapshot redshift.
    omega_M : float
        Present-day matter density parameter.
    rng : numpy.random.Generator-like, optional
        Random-number generator compatible with the internal Gaussian sampler.

    Returns
    -------
    tuple of float
        Velocity offset '(Dvx, Dvy, Dvz)' in km/s.
    """
    Delta_vir_val = Delta_vir(zsnap, omega_M)
    E2_val = E2(zsnap, omega_M)

    sigma = (
        476.0
        * 0.9
        * math.pow(Delta_vir_val * E2_val, 1.0 / 6.0)
        * math.pow(M / 1.0e15, 1.0 / 3.0)
    )

    Dvx = sigma * rand_gauss(rng)
    Dvy = sigma * rand_gauss(rng)
    Dvz = sigma * rand_gauss(rng)
    return Dvx, Dvy, Dvz


def load_velocity_histograms_from_h5(h5file: str):
    """
    Read empirical satellite-velocity histograms from an HDF5 file.

    The file is expected to contain, inside the group 'data', the datasets

    - 'vr_min', 'vr_max', 'Nsat_vr'
    - 'vtan_min', 'vtan_max', 'Nsat_vtan'

    which define binned counts for the radial velocity and the tangential speed,
    respectively.

    Parameters
    ----------
    h5file : str
        Path to the HDF5 input file.

    Returns
    -------
    tuple
        A four-element tuple containing

        - 'vr_bin_edges' : ndarray
            Bin edges for the radial-velocity histogram.
        - 'vr_probs' : ndarray
            Normalized bin probabilities for the radial-velocity distribution.
        - 'vtan_bin_edges' : ndarray
            Bin edges for the tangential-speed histogram.
        - 'vtan_probs' : ndarray
            Normalized bin probabilities for the tangential-speed distribution.

    Raises
    ------
    ValueError
        If one of the histogram datasets is empty.
    """
    with h5py.File(h5file, 'r') as f:
        data = f['data']
        # Radial velocities
        vr_min = np.asarray(data['vr_min'])
        vr_max = np.asarray(data['vr_max'])
        Nsat_vr = np.asarray(data['Nsat_vr'], dtype=float)

        # Tangential speeds (absolute value)
        vtan_min = np.asarray(data['vtan_min'])
        vtan_max = np.asarray(data['vtan_max'])
        Nsat_vtan = np.asarray(data['Nsat_vtan'], dtype=float)

    # Construct edges: last edge is the last bin's max
    if vr_min.size == 0 or vtan_min.size == 0:
        raise ValueError("Velocity histogram datasets are empty in the HDF5 file.")

    vr_bin_edges = np.concatenate([vr_min, [vr_max[-1]]])
    vtan_bin_edges = np.concatenate([vtan_min, [vtan_max[-1]]])

    # Normalize counts → probabilities (safe handling for 0 totals)
    vr_total = Nsat_vr.sum()
    vtan_total = Nsat_vtan.sum()

    if vr_total > 0:
        vr_probs = Nsat_vr / vr_total
    else:
        vr_probs = np.zeros_like(Nsat_vr, dtype=float)

    if vtan_total > 0:
        vtan_probs = Nsat_vtan / vtan_total
    else:
        vtan_probs = np.zeros_like(Nsat_vtan, dtype=float)

    return vr_bin_edges, vr_probs, vtan_bin_edges, vtan_probs

def sample_velocity_from_histogram(bin_edges, probs, size=1, rng=None):
    """
    Draw one or more random values from a tabulated one-dimensional histogram.

    Sampling proceeds in two steps:
    1. a bin is selected according to the supplied probabilities;
    2. a value is drawn uniformly inside that bin.

    Parameters
    ----------
    bin_edges : ndarray
        Histogram bin edges.
    probs : ndarray
        Normalized probabilities associated with each bin.
    size : int, optional
        Number of samples to draw.
    rng : numpy.random.Generator, optional
        Random-number generator. If omitted, a new generator is created.

    Returns
    -------
    float or ndarray
        One sampled value if 'size == 1', otherwise an array of sampled values.
    """
    if rng is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng)

    # Choose bins according to probs
    chosen_bins = rng.choice(len(probs), size=size, p=probs)
    # Uniformly sample within each chosen bin
    left_edges = bin_edges[chosen_bins]
    right_edges = bin_edges[chosen_bins + 1]
    samples = rng.uniform(left_edges, right_edges)
    return samples if size > 1 else samples[0]

def make_orthonormal_basis(r_hat):
    """
    Construct an orthonormal basis adapted to a radial unit vector.

    Given a radial direction 'r_hat', this function returns two additional unit
    vectors orthogonal to 'r_hat' and to each other. The three vectors together
    define a local spherical frame that is used to reconstruct tangential
    velocity components.

    Parameters
    ----------
    r_hat : array-like, shape (3,)
        Radial direction vector.

    Returns
    -------
    tuple of ndarray
        Two orthonormal transverse vectors '(v1, v2)'.
    """
    r = r_hat / np.linalg.norm(r_hat)
    a = np.array([1.0,0.0,0.0]) if abs(r[0])<0.9 else np.array([0.0,1.0,0.0])
    v1 = np.cross(r, a); v1 /= np.linalg.norm(v1)
    v2 = np.cross(r, v1)
    return v1, v2

def sample_empirical_velocity(r_hat, vr_bin_edges, vr_probs, vtan_bin_edges, vtan_probs, rng=None):
    """
    Draw a three-dimensional velocity offset from empirical radial and tangential
    histograms.

    The radial component 'v_rad' and tangential speed '|v_tan|' are sampled
    independently from their respective histograms. The tangential direction is
    then chosen isotropically in the local plane orthogonal to 'r_hat'.

    Parameters
    ----------
    r_hat : array-like, shape (3,)
        Unit vector defining the radial direction of the satellite.
    vr_bin_edges : ndarray
        Bin edges of the empirical radial-velocity histogram.
    vr_probs : ndarray
        Normalized probabilities of the radial-velocity histogram.
    vtan_bin_edges : ndarray
        Bin edges of the empirical tangential-speed histogram.
    vtan_probs : ndarray
        Normalized probabilities of the tangential-speed histogram.
    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    tuple of float
        Velocity offset '(Dvx, Dvy, Dvz)' in km/s.
    """
    if rng is None:
        rng = np.random.default_rng()
    v_rad = sample_velocity_from_histogram(vr_bin_edges, vr_probs, rng=rng)
    v_tan = sample_velocity_from_histogram(vtan_bin_edges, vtan_probs, rng=rng)
    v1, v2 = make_orthonormal_basis(r_hat)
    theta = rng.uniform(0, 2*np.pi)
    tangential_vec = np.cos(theta) * v1 + np.sin(theta) * v2
    Dv = v_rad * r_hat + v_tan * tangential_vec
    return Dv[0], Dv[1], Dv[2]

# Extended velocity profiles
def _gauss_pdf(v, mu, sigma):
    """
    Evaluate a Gaussian probability density.

    Parameters
    ----------
    v : float or ndarray
        Velocity variable.
    mu : float
        Gaussian mean.
    sigma : float
        Gaussian dispersion.

    Returns
    -------
    float or ndarray
        Gaussian probability-density value.
    """
    s = np.abs(sigma)
    return np.exp(-0.5*((v-mu)/s)**2) / (np.sqrt(2*np.pi)*s)

def _safe_radial_pdf_grid(v, vr, mu, sig, floor_frac=1e-12):
    """
    Build a numerically safe tabulated radial-velocity density.

    The nominal model is a signed three-Gaussian mixture. Because fitted
    amplitudes may yield negative values in some regions, the raw density is
    clipped to remain non-negative. If the resulting profile is still not
    normalizable, a fallback using the absolute values of the amplitudes is
    applied.

    Parameters
    ----------
    v : ndarray
        Velocity grid.
    vr : ndarray, shape (3,)
        Amplitudes of the three Gaussian components.
    mu : ndarray, shape (3,)
        Means of the three Gaussian components.
    sig : ndarray, shape (3,)
        Dispersions of the three Gaussian components.
    floor_frac : float, optional
        Small numerical floor, expressed as a fraction of the peak value, added
        to avoid flat segments in the cumulative distribution.

    Returns
    -------
    ndarray
        Non-negative tabulated radial-velocity density on the input grid.
    """
    pdf_raw = (vr[0]*_gauss_pdf(v, mu[0], sig[0]) +
               vr[1]*_gauss_pdf(v, mu[1], sig[1]) +
               vr[2]*_gauss_pdf(v, mu[2], sig[2]))

    pdf = np.clip(pdf_raw, 0.0, None)
    area = np.trapz(pdf, v)

    if not np.isfinite(area) or area <= 0.0:
        pdf = (np.abs(vr[0])*_gauss_pdf(v, mu[0], sig[0]) +
               np.abs(vr[1])*_gauss_pdf(v, mu[1], sig[1]) +
               np.abs(vr[2])*_gauss_pdf(v, mu[2], sig[2]))
        pdf = np.clip(pdf, 0.0, None)
        area = np.trapz(pdf, v)

    if np.max(pdf) > 0:
        pdf += floor_frac * np.max(pdf)

    return pdf

def _sample_from_tabulated_pdf(x, pdf, rng):
    """
    Draw a random value from a tabulated one-dimensional probability density.

    Parameters
    ----------
    x : ndarray
        Grid on which the density is tabulated.
    pdf : ndarray
        Tabulated non-negative probability density.
    rng : numpy.random.Generator
        Random-number generator.

    Returns
    -------
    float
        Random draw obtained by inverse-CDF interpolation.

    Raises
    ------
    ValueError
        If the supplied density is not normalizable.
    """
    area = np.trapz(pdf, x)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Radial PDF not normalizable with given parameters/range.")
    cdf = np.cumsum(pdf) / np.sum(pdf)
    u = rng.random()
    j = np.searchsorted(cdf, u, side="right")
    if j <= 0: return x[0]
    if j >= len(x): return x[-1]
    t = (u - cdf[j-1]) / (cdf[j] - cdf[j-1])
    return x[j-1] + t*(x[j]-x[j-1])

def _vtan_pdf_grid(v, v0, eps, omega, delta):
    """
    Evaluate the analytical tangential-speed density on a grid.

    The functional form is

        p(v) proportional to v0 x v^eps x exp(omega x v^delta),  for v >= 0.

    Parameters
    ----------
    v : ndarray
        Tangential-speed grid.
    v0 : float
        Amplitude parameter.
    eps : float
        Power-law exponent.
    omega : float
        Exponential coefficient.
    delta : float
        Power of the exponential argument.

    Returns
    -------
    ndarray
        Non-negative tabulated tangential-speed density.
    """
    v = np.asarray(v, dtype=float)
    pdf = np.where(v >= 0.0, v0 * np.power(v, eps) * np.exp(omega * np.power(v, delta)), 0.0)
    # ensure non-negative due to numerical underflow
    return np.clip(pdf, 0.0, None)

def _sample_vtan(v0, eps, omega, delta, vmin, vmax, rng, ngrid=4096):
    """
    Draw a tangential speed from the analytical extended tangential model.

    Parameters
    ----------
    v0 : float
        Amplitude parameter.
    eps : float
        Power-law exponent.
    omega : float
        Exponential coefficient.
    delta : float
        Exponent appearing in the exponential argument.
    vmin : float
        Lower bound of the sampling interval.
    vmax : float
        Upper bound of the sampling interval.
    rng : numpy.random.Generator
        Random-number generator.
    ngrid : int, optional
        Number of grid points used to tabulate the density.

    Returns
    -------
    float
        Sampled tangential speed in km/s.

    Raises
    ------
    ValueError
        If the velocity range is invalid or if the density is not normalizable
        in the requested interval.
    """
    if rng is None:
        rng = np.random.default_rng()
    vmin = max(0.0, float(vmin))
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Bad tangential range [vmin, vmax].")

    v = np.linspace(vmin, vmax, ngrid)
    pdf = _vtan_pdf_grid(v, v0, eps, omega, delta)

    area = np.trapz(pdf, v)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Tangential PDF not normalizable on the chosen range; "
                         "check (v0, eps, omega, delta) or widen [vmin, vmax].")

    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * (v[1:] - v[:-1]))
    cdf = np.concatenate(([0.0], cdf / cdf[-1]))  # normalize and prepend 0

    u = rng.random()
    j = np.searchsorted(cdf, u, side="right")
    j = np.clip(j, 1, len(cdf)-1)
    # linear interpolation within bin
    c0, c1 = cdf[j-1], cdf[j]
    t = 0.0 if (c1 == c0) else (u - c0) / (c1 - c0)
    return v[j-1] + t * (v[j] - v[j-1])

def sample_velocity_analytic(
    r_hat,
    vr1, vr2, vr3,
    mu1, mu2, mu3,
    sigma1, sigma2, sigma3,
    vtan0, epsilon, omega, delta,
    vtan_min=0.0, vtan_max=2000.0,
    vr_min=None, vr_max=None,
    rng=None, ngrid=4096, pad_sigmas=10.0
):
    """
    Draw a three-dimensional velocity offset from the analytical extended model.

    The radial component is sampled from a three-Gaussian mixture, while the
    tangential speed is sampled independently from the analytical tangential
    law. The tangential direction is then chosen isotropically in the local
    plane orthogonal to 'r_hat'.

    Parameters
    ----------
    r_hat : array-like, shape (3,)
        Radial direction of the satellite.
    vr1, vr2, vr3 : float
        Amplitudes of the three Gaussian radial components.
    mu1, mu2, mu3 : float
        Means of the three Gaussian radial components, in km/s.
    sigma1, sigma2, sigma3 : float
        Dispersions of the three Gaussian radial components, in km/s.
    vtan0 : float
        Amplitude parameter of the tangential-speed law.
    epsilon : float
        Power-law exponent of the tangential-speed law.
    omega : float
        Exponential coefficient of the tangential-speed law.
    delta : float
        Exponent entering the exponential argument of the tangential-speed law.
    vtan_min, vtan_max : float, optional
        Sampling range for the tangential speed.
    vr_min, vr_max : float, optional
        Explicit sampling range for the radial velocity. If omitted, the range
        is inferred from the Gaussian means and dispersions.
    rng : numpy.random.Generator, optional
        Random-number generator.
    ngrid : int, optional
        Number of grid points used in the tabulated sampling.
    pad_sigmas : float, optional
        Width factor used when constructing the automatic radial-velocity range.

    Returns
    -------
    tuple of float
        Velocity offset '(Dvx, Dvy, Dvz)' in km/s.

    Raises
    ------
    ValueError
        If the radial or tangential sampling range is invalid, or if the
        tangential model is requested with 'delta <= 0'.
    """
    if rng is None:
        rng = np.random.default_rng()

    mus = np.array([mu1, mu2, mu3], float)
    sig = np.abs(np.array([sigma1, sigma2, sigma3], float))
    vrs = np.array([vr1, vr2, vr3], float)

    # rango automático amplio
    lo_auto = np.min(mus - pad_sigmas*sig)
    hi_auto = np.max(mus + pad_sigmas*sig)
    vmin = lo_auto if vr_min is None else vr_min
    vmax = hi_auto if vr_max is None else vr_max
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Bad radial range; provide vr_min/vr_max or check params.")

    v_grid = np.linspace(vmin, vmax, ngrid)

    pdf_r = _safe_radial_pdf_grid(v_grid, vrs, mus, sig)

    # si aún no hay área (extremadamente patológico), intenta ampliar rango 2×
    area = np.trapz(pdf_r, v_grid)
    if area <= 0 or not np.isfinite(area):
        span = vmax - vmin
        v_grid = np.linspace(vmin-0.5*span, vmax+0.5*span, ngrid)
        pdf_r = _safe_radial_pdf_grid(v_grid, vrs, mus, sig)

    v_rad = _sample_from_tabulated_pdf(v_grid, pdf_r, rng)

    if delta <= 0:
        raise ValueError("For (12.4) use delta>0; typically omega<0 for normalizable tails.")
    v_tan = _sample_vtan(vtan0, epsilon, omega, delta, vtan_min, vtan_max, rng, ngrid=ngrid)

    r_hat = np.asarray(r_hat, float); r_hat /= np.linalg.norm(r_hat)
    v1, v2 = make_orthonormal_basis(r_hat)
    theta = rng.uniform(0.0, 2.0*np.pi)
    tangential_vec = np.cos(theta)*v1 + np.sin(theta)*v2
    Dv = v_rad * r_hat + v_tan * tangential_vec
    return Dv[0], Dv[1], Dv[2]