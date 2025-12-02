import numpy as np
from numba import jit

import src.hod_const as c
from src.hod_cosmology import Delta_vir
import h5py

import numpy as np
from numba import njit
import src.hod_const as cst
from src.hod_cosmology import Delta_vir

@jit(nopython=True)
def f_nfw(x):
    return np.log(1.0 + x) - x/(1.0 + x)

def build_nfw_cdf_up_to_c(c, ngrid=2048, loggrid=True):
    """
    Mallado en x = r/rs hasta x_max = c (i.e., Rvir).
    Devuelve: x_grid (>=0), cdf(x) normalizada.
    """
    if c <= 0.0:
        raise ValueError("Concentration 'c' must be > 0.")
    # malla en x: evitar 0 exacto por estabilidad
    xmin = 1e-12
    if loggrid:
        x = np.exp(np.linspace(np.log(xmin), np.log(c), ngrid))
    else:
        x = np.linspace(xmin, c, ngrid)

    F = f_nfw(x)
    Fmax = f_nfw(c)
    if Fmax <= 0.0:
        raise ValueError("Invalid NFW normalization; check 'c'.")
    cdf = F / Fmax
    return x, cdf

def sample_from_cdf_1d(xgrid, cdf, rng):
    u = rng.random()
    j = np.searchsorted(cdf, u, side="right")
    j = np.clip(j, 1, len(cdf)-1)
    c0, c1 = cdf[j-1], cdf[j]
    x0, x1 = xgrid[j-1], xgrid[j]
    t = (u - c0) / (c1 - c0 + 1e-300)
    return x0 + t*(x1 - x0)

@jit(nopython=True)
def Rvir_from_mass(M, Delta_vir, rho_crit):
    return (3.0*M / (4.0*np.pi*Delta_vir*rho_crit))**(1.0/3.0)

def generate_nfw_position_with_concentration(
    M, zsnap, omega_M, rho_crit, Delta_vir_func,
    Rvir, Rs,
    rng=None,
    ngrid=4096,
    loggrid=True
):
    """
    NFW clásico, truncado implícitamente en Rvir (no K_trunc).
    - Construimos CDF en x=r/rs hasta x=c.
    - Muestreamos r y dirección isotrópica.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Escalas del halo
    c_halo = Rvir / Rs

    # CDF hasta x=c (i.e., r<=Rvir)
    xgrid, cdf = build_nfw_cdf_up_to_c(c_halo, ngrid=ngrid, loggrid=loggrid)
    x = sample_from_cdf_1d(xgrid, cdf, rng)
    r = x * Rs

    # Dirección isotrópica
    phi = rng.uniform(0.0, 2.0*np.pi)
    mu  = rng.uniform(-1.0, 1.0)
    s   = np.sqrt(1.0 - mu*mu)
    Dx, Dy, Dz = r*s*np.cos(phi), r*s*np.sin(phi), r*mu
    return Dx, Dy, Dz

@jit(nopython=True)
def concentration_klypin(M, z):
    """Concentration parameter calculation"""
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
    """NFW integral"""
    return (1.0 / (1.0 + x) + np.log(1.0 + x) - 1.0)


@jit(nopython=True)
def R_from_mass(Mass, OVD, rho_crit):
    """Radius from mass and overdensity"""
    return (3.0 / (4.0 * rho_crit * OVD * np.pi) * Mass)**(1.0/3.0)


@jit(nopython=True)
def generate_nfw_position(M, K, zsnap, omega_M):
    """NFW position generation"""
    x_max = concentration_klypin(M, zsnap)

    I_max = I_NFW(x_max)
    
    y_rand = np.random.random() * I_max

    tol = 0.0001 * I_NFW(x_max)
    low = 0.0
    high = x_max
    mid = 0.5 * (low + high)
    y_try = I_NFW(mid)
    
    # Keep iterating until convergence
    while abs(y_try - y_rand) >= tol: 
        if y_try > y_rand:
            high = mid
        else:
            low = mid
        mid = 0.5 * (low + high)
        y_try = I_NFW(mid)
    
    # Calculate radius
    Delta_vir_val = Delta_vir(zsnap, omega_M)
    R_mass = R_from_mass(M, Delta_vir_val, c.rho_crit)
    c_val = concentration_klypin(M, zsnap)

    R = mid * R_mass / (c_val*K)

    M_test = 1e12  # masa típica de halo
    Delta_vir_val = Delta_vir(0, 0.3)
    R_test = R_from_mass(M_test, Delta_vir_val, c.rho_crit)
    print(R_test)

    
    # Generate spherical coordinates
    phi = np.random.random() * 2 * np.pi
    costh = np.random.random() * 2 - 1.0
    
    Dx = R * np.sqrt(1.0 - costh*costh) * np.cos(phi)
    Dy = R * np.sqrt(1.0 - costh*costh) * np.sin(phi)
    Dz = R * costh
    
    return Dx, Dy, Dz

@jit(nopython=True)
def generate_nfw_position_from_c(Rvir, Rs):
    """
    NFW position generation with given concentration c_val.
    """
    x_max = Rvir / Rs
    c_val = Rvir / Rs

    I_max = I_NFW(x_max)
    y_rand = np.random.random() * I_max

    tol = 1e-4 * I_max
    low, high = 0.0, x_max
    mid = 0.5 * (low + high)
    y_try = I_NFW(mid)

    while abs(y_try - y_rand) >= tol:
        if y_try > y_rand:
            high = mid
        else:
            low = mid
        mid = 0.5 * (low + high)
        y_try = I_NFW(mid)

    R = mid * Rvir / (c_val)

    phi = np.random.random() * 2.0 * np.pi
    costh = np.random.random() * 2.0 - 1.0
    sinth = np.sqrt(1.0 - costh * costh)

    Dx = R * sinth * np.cos(phi)
    Dy = R * sinth * np.sin(phi)
    Dz = R * costh

    return Dx, Dy, Dz

@jit(nopython=True)
def extended_N_of_r(r, r0, alpha, beta, kappa, N0):
    """
    N(r) = N0 * (r/r0)^alpha * (1 + (r/r0)^beta)^kappa
    (number per unit radius; N0 cancels out when building the PDF)
    """
    x = np.clip(r / r0, 1e-300, None)  # numeric safety for alpha < 0
    return N0 * (x**alpha) * (1.0 + x**beta)**kappa

def _build_cdf(Rmax, r0, alpha, beta, kappa, N0, ngrid=2048):
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
    u = rng.random()
    j = np.searchsorted(cdf, u, side="right")
    j = np.clip(j, 1, len(cdf)-1)
    c0, c1 = cdf[j-1], cdf[j]
    r0, r1 = r_grid[j-1], r_grid[j]
    t = (u - c0) / (c1 - c0)
    return r0 + t*(r1 - r0)

@jit(nopython=True)
def _random_unit_vector(rng):
    phi = rng.uniform(0.0, 2.0*np.pi)
    mu  = rng.uniform(-1.0, 1.0)
    s   = np.sqrt(1.0 - mu*mu)
    return np.array([s*np.cos(phi), s*np.sin(phi), mu])

@jit(nopython=True)
def Rvir_from_mass(M, Delta_vir, rho_crit):
    return (3.0*M / (4.0*np.pi*Delta_vir*rho_crit))**(1.0/3.0)

def generate_extended_position(
    M, zsnap, omega_M, rho_crit, Delta_vir_func,
    K_trunc,                      # e.g. Rmax = Rvir / K_trunc
    r0, alpha, beta, kappa, N0,   
    rng=None, ngrid=4096
):
    """
    Draw a single satellite position (Dx, Dy, Dz) from the extended-NFW radial law.

    Returns
    -------
    Dx, Dy, Dz : float
        Cartesian offsets from the halo center.

    Notes
    -----
    - N0 affects the *mean count* (integral of N), but cancels in the *shape* of the
      sampling PDF p(r) ∝ N(r). It is still accepted for completeness.
    - Rmax is set to Rvir / K_trunc to mimic your NFW truncation convention.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute truncation radius
    Delta = Delta_vir_func(zsnap, omega_M)
    Rvir  = Rvir_from_mass(M, Delta, rho_crit)
    Rmax  = Rvir / max(K_trunc, 1e-6)

    # Build CDF once and sample r
    r_grid, cdf = _build_cdf(Rmax, r0, alpha, beta, kappa, N0, ngrid=ngrid)
    r = _sample_r_from_cdf(r_grid, cdf, rng)

    # Isotropic direction
    nhat = _random_unit_vector(rng)
    Dx, Dy, Dz = (r * nhat).tolist()
    return Dx, Dy, Dz


def load_radial_histogram_from_h5(h5file, group='data'):
    """
    Load the empirical radial distribution histogram from an h5 file.
    Returns:
        r_bin_edges: np.ndarray of bin edges (length N+1)
        Nsat_r: np.ndarray of counts (length N)
        probs_r: np.ndarray of probabilities (length N)
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
    Sample radial distances according to an empirical PDF (histogram).
    Returns one value if size=1, or array if size>1.
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
    Generate one or more random unit vectors (isotropically distributed).
    Returns: (N,3) array if size > 1, else (3,) array.
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
    Samples Dx, Dy, Dz according to the empirical radial profile (histogram),
    assigning an isotropic random direction.
    Returns: array (size, 3) or (3,) if size=1.
    """
    r = sample_radius_from_histogram(r_bin_edges, r_probs, size=size, rng=rng)
    directions = random_unit_vector(size=size, rng=rng)
    pos = r[:, None] * directions if size > 1 else r * directions
    return pos

