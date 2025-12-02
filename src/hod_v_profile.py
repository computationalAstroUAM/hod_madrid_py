import numpy as np
from numba import jit
import math
import src.hod_io as io
from src.hod_cosmology import Delta_vir, E2
from src.hod_pdf import rand_gauss
import h5py

@jit(nopython=True)
def generate_virial_velocity(M: float, zsnap: float, omega_M: float):
    """
    Draw a 3D random velocity (Dvx, Dvy, Dvz) consistent with a virial dispersion.

    The velocity dispersion is modeled as
        sigma_1D = 476 * 0.9 * [Δ_vir(z) * E(z)^2]^(1/6) * (M / 1e15)^(1/3)  [km/s]
    and each component is sampled as a Gaussian deviate with std = sigma_1D.

    Parameters
    ----------
    M : float
        Halo mass in Msun/h (linear, not log10).
    zsnap : float
        Snapshot redshift.
    omega_M : float
        Present-day matter density parameter Ω_M(0).

    Returns
    -------
    (Dvx, Dvy, Dvz) : tuple of floats
        Velocity components in km/s drawn from N(0, sigma_1D^2).

    Notes
    -----
    - Δ_vir(z) is the virial overdensity (Bryan & Norman 1998).
    - E(z)^2 = Ω_M(1+z)^3 + Ω_Λ (flat ΛCDM, with Ω_Λ=1-Ω_M).
    - The mass scaling ∝ M^(1/3) and the weak redshift dependence via
      [Δ_vir E^2]^(1/6) are typical of virial arguments.
    - `rand_gauss()` must return a standard normal deviate (mean 0, std 1).
    """
    Delta_vir_val = Delta_vir(zsnap, omega_M)
    E2_val = E2(zsnap, omega_M)

    sigma = (
        476.0
        * 0.9
        * math.pow(Delta_vir_val * E2_val, 1.0 / 6.0)
        * math.pow(M / 1.0e15, 1.0 / 3.0)
    )

    Dvx = sigma * rand_gauss()
    Dvy = sigma * rand_gauss()
    Dvz = sigma * rand_gauss()
    return Dvx, Dvy, Dvz


def load_velocity_histograms_from_h5(h5file: str):
    """
    Load satellite velocity histograms from an HDF5 produced by h2s.

    The file is expected to contain, under the group 'data', the per-bin min/max
    edges and counts for:
      - radial velocity v_r  (datasets: 'vr_min', 'vr_max', 'Nsat_vr')
      - tangential speed |v_tan| (datasets: 'vtan_min', 'vtan_max', 'Nsat_vtan')

    This helper reconstructs the **bin edges** and returns **normalized PDFs**.

    Parameters
    ----------
    h5file : str
        Path to the HDF5 file (e.g., 'h2s_output.h5').

    Returns
    -------
    vr_bin_edges : ndarray, shape (n_vr_bins + 1,)
        Monotonic edges for the v_r histogram.
    vr_probs : ndarray, shape (n_vr_bins,)
        Bin probabilities for v_r (sum to 1.0).
    vtan_bin_edges : ndarray, shape (n_vtan_bins + 1,)
        Monotonic edges for the |v_tan| histogram.
    vtan_probs : ndarray, shape (n_vtan_bins,)
        Bin probabilities for |v_tan| (sum to 1.0).

    Notes
    -----
    - The input is assumed to store **counts per bin** (Nsat_vr, Nsat_vtan). We
      normalize by the total count to obtain probabilities.
    - Units are those of the input file (typically km/s). This function does not
      rescale units.
    - If a histogram has zero total counts, the corresponding probability array
      is returned as zeros to avoid division-by-zero.

    Examples
    --------
    >>> vr_edges, vr_p, vt_edges, vt_p = load_velocity_histograms_from_h5("h2s_output.h5")
    >>> # Sample a radial velocity bin index:
    >>> idx = np.random.choice(len(vr_p), p=vr_p)
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

def load_velocity_histograms_from_h5_not_normalized(h5file):
    """
    Reads satellite velocity histograms from h2s_output.h5 file.

    Returns:
        - vr_bin_edges: array of bin edges for v_rad
        - vr_probs: probabilities per bin for v_rad (normalized)
        - vtan_bin_edges: array of bin edges for |v_tan|
        - vtan_probs: probabilities per bin for |v_tan| (normalized)
    """
    with h5py.File(h5file, 'r') as f:
        data = f['data']
        # Radial velocities
        vr_min = np.array(data['vr_min'])  # shape (n_bins,)
        vr_max = np.array(data['vr_max'])  # shape (n_bins,)
        Nsat_vr = np.array(data['Nsat_vr']).astype(float)
        # Tangential velocities
        vtan_min = np.array(data['vtan_min'])
        vtan_max = np.array(data['vtan_max'])
        Nsat_vtan = np.array(data['Nsat_vtan']).astype(float)

    # Construct bin edges
    vr_bin_edges = np.concatenate([vr_min, [vr_max[-1]]])
    vtan_bin_edges = np.concatenate([vtan_min, [vtan_max[-1]]])

    # Normalize to get probabilities
    vr_probs = Nsat_vr
    vtan_probs = Nsat_vtan

    return vr_bin_edges, vr_probs, vtan_bin_edges, vtan_probs

def sample_velocity_from_histogram(bin_edges, probs, size=1, random_state=None):
    """
    Samples random values from a 1D histogram (defined by bin edges and probabilities).

    Returns:
        - samples: array of sampled values (shape: size)
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_state)

    # Choose bins according to probs
    chosen_bins = rng.choice(len(probs), size=size, p=probs)
    # Uniformly sample within each chosen bin
    left_edges = bin_edges[chosen_bins]
    right_edges = bin_edges[chosen_bins + 1]
    samples = rng.uniform(left_edges, right_edges)
    return samples if size > 1 else samples[0]

def random_perpendicular_unit_vector(r_hat):
    """
    Generates a random unit vector perpendicular to r_hat.
    """
    random_vec = np.random.randn(3)
    # Ensure the random vector is not parallel to r_hat
    perp = np.cross(r_hat, random_vec)
    norm = np.linalg.norm(perp)
    if norm < 1e-10:
        # Try again if accidental parallelism
        random_vec = np.random.randn(3)
        perp = np.cross(r_hat, random_vec)
        norm = np.linalg.norm(perp)
    return perp / norm


def make_orthonormal_basis(r_hat):
    r_hat = r_hat / np.linalg.norm(r_hat)
    if np.abs(r_hat[0]) < 0.9:
        v1 = np.cross(r_hat, [1,0,0])
    else:
        v1 = np.cross(r_hat, [0,1,0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(r_hat, v1)
    return v1, v2

def sample_empirical_velocity(r_hat, vr_bin_edges, vr_probs, vtan_bin_edges, vtan_probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v_rad = sample_velocity_from_histogram(vr_bin_edges, vr_probs, random_state=rng)
    v_tan = sample_velocity_from_histogram(vtan_bin_edges, vtan_probs, random_state=rng)
    v1, v2 = make_orthonormal_basis(r_hat)
    theta = rng.uniform(0, 2*np.pi)
    tangential_vec = np.cos(theta) * v1 + np.sin(theta) * v2
    Dv = v_rad * r_hat + v_tan * tangential_vec
    return Dv[0], Dv[1], Dv[2]

# Extended velocity profiles
def _gauss_pdf(v, mu, sigma):
    s = np.abs(sigma)
    return np.exp(-0.5*((v-mu)/s)**2) / (np.sqrt(2*np.pi)*s)

def _safe_radial_pdf_grid(v, vr, mu, sig, floor_frac=1e-12):
    """Construye una 'PDF' radial robusta a signos en vr y sigma."""
    # intento 1: amplitudes con signo, como en el fit del paper
    pdf_raw = (vr[0]*_gauss_pdf(v, mu[0], sig[0]) +
               vr[1]*_gauss_pdf(v, mu[1], sig[1]) +
               vr[2]*_gauss_pdf(v, mu[2], sig[2]))

    pdf = np.clip(pdf_raw, 0.0, None)
    area = np.trapz(pdf, v)

    # si todo quedó ≤0 (o área ~0), fallback: usar |vr_i|
    if not np.isfinite(area) or area <= 0.0:
        pdf = (np.abs(vr[0])*_gauss_pdf(v, mu[0], sig[0]) +
               np.abs(vr[1])*_gauss_pdf(v, mu[1], sig[1]) +
               np.abs(vr[2])*_gauss_pdf(v, mu[2], sig[2]))
        pdf = np.clip(pdf, 0.0, None)
        area = np.trapz(pdf, v)

    # suelo numérico para evitar CDF plana en regiones
    if np.max(pdf) > 0:
        pdf += floor_frac * np.max(pdf)

    return pdf

def _sample_from_tabulated_pdf(x, pdf, rng):
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
    Strict form: pdf(v) ∝ v0 * v**eps * exp(omega * v**delta), for v >= 0.
    (The overall factor v0 cancels upon normalization, but we keep it to
    match the exact parameterization.)
    """
    v = np.asarray(v, dtype=float)
    pdf = np.where(v >= 0.0, v0 * np.power(v, eps) * np.exp(omega * np.power(v, delta)), 0.0)
    # ensure non-negative due to numerical underflow
    return np.clip(pdf, 0.0, None)

def _sample_vtan(v0, eps, omega, delta, vmin, vmax, rng, ngrid=4096):
    """
    Sample |v_tan| from the strict law on [vmin, vmax] via tabulated CDF.
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

def make_orthonormal_basis(r_hat):
    r = r_hat / np.linalg.norm(r_hat)
    a = np.array([1.0,0.0,0.0]) if abs(r[0])<0.9 else np.array([0.0,1.0,0.0])
    v1 = np.cross(r, a); v1 /= np.linalg.norm(v1)
    v2 = np.cross(r, v1)
    return v1, v2

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