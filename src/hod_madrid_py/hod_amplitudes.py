# hod_amplitudes.py

"""
Amplitude calibration, bias matching, and fast mass-compressed solvers
for analytical Halo Occupation Distribution (HOD) models.

Overview
--------
This module provides the core utilities needed to connect an analytical HOD
prescription to global target observables measured or imposed for a galaxy
sample, such as:

    - galaxy number density, n_gal
    - satellite fraction, f_sat
    - large-scale galaxy bias, b_gal

The module supports two complementary tasks:

1. Amplitude calibration
   Given an analytical HOD shape and a halo mass catalogue, compute the
   amplitudes of the central and satellite occupations, '(A_c, A_s)', such that
   the resulting mock matches a desired number density and satellite fraction.

2. Bias-based mu solving
   When the characteristic HOD mass scale 'mu' is not fixed a priori, solve for
   'mu' such that the mock reproduces a target large-scale galaxy bias 'b_gal',
   while simultaneously matching the desired 'n_gal' and 'f_sat'.

To accelerate the 'mu' solver, the full halo mass catalogue can be compressed
into a binned representation in log-mass space. The final amplitudes are then
computed again using the full halo sample for improved accuracy.

Physical conventions
--------------------
- Halo masses are assumed to be in Msun/h.
- Log-masses are assumed to be base-10 logarithms, i.e. 'log10(M)'.
- Number densities are in h^3 / Mpc^3 when 'Lbox' is given in Mpc/h.
- The halo bias relation is provided as a function of 'log10(M)'.

Analytical HOD ingredients
--------------------------
The module uses the analytical central and satellite occupation shape functions

    g_cen(logM)
    g_sat(logM)

with amplitudes 'A_c' and 'A_s' applied externally. This separation is useful
because it allows:
- a fixed parametric shape,
- a clean amplitude rescaling,
- and an efficient inverse calibration against target observables.

Conformity
----------
If enabled, the mean satellite occupation is modulated by whether a halo hosts
a central galaxy, through two global factors '(K1_global, K2_global)'. In the
mean-field treatment adopted here, this translates into an effective weight for
satellite occupation depending on the central probability.

Notes
-----
- This module assumes the mapping 'mu -> (M0, M1, alpha, sig, gamma)' is defined
  elsewhere, via 'src.hod_madrid_py.hod_shape.get_hod_derived_params'.
- The numerical solver for 'mu' is based on a bracket search followed by
  bisection when a sign change is found.
- If no sign-changing bracket exists in the scanned interval, the code returns
  the scanned value of 'mu' that minimizes '|b_gal(mu) - b_gal,target|'.

See Also
--------
src.hod_madrid_py.hod_shape
    Analytical HOD functional forms and default derived parameter mappings.
src.hod_madrid_py.hod_io
    Chunked catalogue readers used to stream halo masses from disk.
"""

import numpy as np
import math

import src.hod_madrid_py.hod_io as io
import src.hod_madrid_py.hod_const as c
import src.hod_madrid_py.hod_shape as shape

def central_shape_unit(logM, mu, sig, gamma, hodshape):
    """
    Return the unit-shape central occupation function for a given HOD family.

    This function evaluates the central occupation shape without applying the
    global amplitude A_c. The output is therefore the dimensionless shape
    factor 'g_cen(logM)' such that the full mean central occupation is

        <N_cen>(logM) = A_c * g_cen(logM)

    depending on the selected HOD family:

    - 'hodshape == 1': erf-like sigmoid
    - 'hodshape == 2': Gaussian
    - otherwise      : Gaussian + power-law tail (HOD3)

    Parameters
    ----------
    logM : array_like
        Base-10 halo mass, 'log10(M / (Msun/h))'.
    mu : float
        Characteristic HOD mass scale in log10 units.
    sig : float
        Width/scatter parameter controlling the transition or Gaussian width.
    gamma : float
        High-mass power-law slope used only for HOD3.
    hodshape : int
        Identifier of the analytical HOD family.

    Returns
    -------
    ndarray
        Unit-shape central occupation evaluated at 'logM'.

    Notes
    -----
    - No clipping to [0, 1] is performed here, because this function represents
      only the shape factor prior to amplitude rescaling.
    - For HOD3, the low-mass side is Gaussian and the high-mass side becomes
      a power law in log-mass.
    """
    logM = np.asarray(logM, dtype=float)

    if hodshape == 1:
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec((logM - mu) / sig))

    elif hodshape == 2:
        return (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((logM - mu) / sig) ** 2
        )

    else:  # HOD3
        out = np.empty_like(logM, dtype=float)
        low = logM < mu
        high = ~low

        out[low] = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((logM[low] - mu) / sig) ** 2
        )
        out[high] = (1.0 / (sig * np.sqrt(2.0 * np.pi))) * (
            10.0 ** (gamma * (logM[high] - mu))
        )
        return out
    
def satellite_shape_unit(logM, M0, M1, alpha):
    """
    Return the unit-shape satellite occupation function.

    The function evaluates the standard truncated power-law form used for
    satellites, without applying the global amplitude A_s. Thus,

        <N_sat>(M) = A_s * g_sat(M)

    with

        g_sat(M) = ((M - M0) / M1)^alpha   for M > M0
                 = 0                       otherwise

    Parameters
    ----------
    logM : array_like
        Base-10 halo mass, 'log10(M / (Msun/h))'.
    M0 : float
        Low-mass cutoff scale for satellites [Msun/h].
    M1 : float
        Characteristic satellite mass scale [Msun/h].
    alpha : float
        Power-law slope.

    Returns
    -------
    ndarray
        Unit-shape satellite occupation evaluated at 'logM'.
    """
    M = 10.0 ** np.asarray(logM, dtype=float)
    out = np.zeros_like(M, dtype=float)

    mask = M > M0
    out[mask] = ((M[mask] - M0) / M1) ** alpha
    return out

def load_logM_from_halo_catalog(infile, ftype, chunk_size=c.chunk_size):
    """
    Stream halo masses from an input catalogue and return the concatenated log-mass array.

    Parameters
    ----------
    infile : str
        Path to the halo catalogue.
    ftype : str
        Input catalogue type understood by 'io.read_halo_data_chunked'
        (e.g. 'txt', 'h5', 'npy').
    chunk_size : int, optional
        Number of rows per chunk when streaming the file.

    Returns
    -------
    ndarray
        Concatenated array of halo log-masses, taken from canonical column 6.

    Raises
    ------
    ValueError
        If no halo masses are found in the input catalogue.

    Notes
    -----
    This helper is intended for global calibration tasks, such as deriving
    amplitudes or solving for 'mu', where access to the full halo mass
    distribution is required.
    """
    all_logM = []

    for chunk in io.read_halo_data_chunked(infile, ftype, chunk_size):
        if len(chunk) == 0:
            continue
        all_logM.append(chunk[:, 6])

    if len(all_logM) == 0:
        raise ValueError("No halo masses found in input catalog.")

    return np.concatenate(all_logM)

def compute_Ac_As_from_halo_masses(
    logM_halos,
    Lbox,
    mu,
    sig,
    gamma,
    hodshape,
    M0,
    M1,
    alpha,
    ngal_target,
    fsat_target,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Derive '(A_c, A_s)' from a halo mass catalogue and target '(n_gal, f_sat)'.

    The amplitudes are obtained by matching the analytical HOD model to:
    - a target total number density 'n_gal',
    - a target satellite fraction 'f_sat'.

    In the absence of conformity, the equations are

        n_cen,target = (1 - f_sat) * n_gal
        n_sat,target = f_sat * n_gal

        A_c = n_cen,target / I_c
        A_s = n_sat,target / I_s

    where

        I_c = (1/V) sum g_cen
        I_s = (1/V) sum g_sat

    If conformity is enabled, the effective satellite integral is modified by
    the central probability through the global factors '(K1_global, K2_global)'.

    Parameters
    ----------
    logM_halos : array_like
        Halo mass distribution in log10 units.
    Lbox : float
        Simulation box size [Mpc/h].
    mu, sig, gamma, hodshape : float, float, float, int
        Parameters defining the central occupation shape.
    M0, M1, alpha : float
        Parameters defining the satellite occupation shape.
    ngal_target : float
        Target galaxy number density [h^3 / Mpc^3].
    fsat_target : float
        Target satellite fraction.
    conformity : bool, optional
        Whether to include the effective conformity weighting in the satellite term.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    tuple
        If 'conformity=False':
            (Ac, As, Ic, Is0)
        If 'conformity=True':
            (Ac, As, Ic, Is_eff)

    Raises
    ------
    ValueError
        If the central or satellite integrals are non-positive.

    Notes
    -----
    In conformity mode, the effective satellite integral is

        I_s,eff = K2 * I_s + (K1 - K2) * A_c * I_cs

    where

        I_cs = (1/V) sum [g_cen * g_sat].

    This corresponds to a mean-field treatment in which the satellite weight
    depends on the probability that a central galaxy is present.
    """
    volume = Lbox ** 3

    gcen = central_shape_unit(logM_halos, mu, sig, gamma, hodshape)
    gsat = satellite_shape_unit(logM_halos, M0, M1, alpha)

    Ic = np.sum(gcen) / volume
    Is0 = np.sum(gsat) / volume

    if Ic <= 0:
        raise ValueError("Central integral Ic is not positive.")
    if Is0 <= 0:
        raise ValueError("Satellite integral Is is not positive.")

    ncen_target = (1.0 - fsat_target) * ngal_target
    nsat_target = fsat_target * ngal_target

    Ac = ncen_target / Ic

    if conformity:
        Ics = np.sum(gcen * gsat) / volume
        Is_eff = K2_global * Is0 + (K1_global - K2_global) * Ac * Ics

        if Is_eff <= 0:
            raise ValueError("Effective satellite integral Is_eff is not positive.")

        As = nsat_target / Is_eff
        return Ac, As, Ic, Is_eff

    else:
        As = nsat_target / Is0
        return Ac, As, Ic, Is0

def predict_number_densities_from_halo_masses(
    logM_halos,
    Lbox,
    mu,
    sig,
    gamma,
    hodshape,
    M0,
    M1,
    alpha,
    Ac,
    As,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Predict '(n_cen, n_sat, n_gal, f_sat)' from a halo mass catalogue and HOD amplitudes.

    Parameters
    ----------
    logM_halos : array_like
        Halo mass distribution in log10 units.
    Lbox : float
        Simulation box size [Mpc/h].
    mu, sig, gamma, hodshape : float, float, float, int
        Parameters defining the central occupation shape.
    M0, M1, alpha : float
        Parameters defining the satellite occupation shape.
    Ac, As : float
        Central and satellite amplitudes.
    conformity : bool, optional
        Whether to include conformity in the satellite prediction.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    tuple
        '(ncen, nsat, ngal, fsat)'.

    Notes
    -----
    This routine is typically used as a consistency check after solving for
    amplitudes or for 'mu'.
    """
    volume = Lbox ** 3

    gcen = central_shape_unit(logM_halos, mu, sig, gamma, hodshape)
    gsat = satellite_shape_unit(logM_halos, M0, M1, alpha)
    
    ncen = Ac * np.sum(gcen) / volume
    
    if conformity:
        pcen = Ac * gcen
        conf_weight = K2_global + (K1_global - K2_global) * pcen
        nsat = As * np.sum(gsat * conf_weight) / volume
    else:
        nsat = As * np.sum(gsat) / volume

    ngal = ncen + nsat
    fsat = 0.0 if ngal <= 0 else nsat / ngal

    return ncen, nsat, ngal, fsat

def halo_bias_poly(logM, coeffs):
    """
    Evaluate a polynomial approximation to the halo bias relation.

    Parameters
    ----------
    logM : array_like
        Base-10 halo mass.
    coeffs : array_like
        Polynomial coefficients compatible with 'numpy.polyval'.

    Returns
    -------
    ndarray
        Halo bias evaluated at 'logM'.

    Notes
    -----
    The polynomial is assumed to approximate the large-scale halo bias as a
    function of 'log10(M)'.
    """
    x = np.asarray(logM, dtype=float)
    return np.polyval(coeffs, x)

def load_bias_table(bias_file):
    """
    Load a halo bias table from a two-column text file.

    The file must contain at least two columns:

        log10(M)   bias

    Parameters
    ----------
    bias_file : str
        Path to the text file containing the bias relation.

    Returns
    -------
    tuple
        '(logM, bias)' as one-dimensional NumPy arrays.

    Raises
    ------
    ValueError
        If no valid finite rows are found.

    Notes
    -----
    Rows with non-finite values are removed.
    """
    data = np.loadtxt(bias_file, comments="#", usecols=(0, 1))
    if data.ndim == 1:
        data = data[None, :]

    logM = np.asarray(data[:, 0], dtype=float)
    bias = np.asarray(data[:, 1], dtype=float)

    mask = np.isfinite(logM) & np.isfinite(bias)
    logM = logM[mask]
    bias = bias[mask]

    if logM.size == 0:
        raise ValueError("No valid rows found in bias file.")

    return logM, bias

def fit_bias_polynomial_from_file(bias_file, degree, drop_nonpositive=True):
    """
    Fit a polynomial approximation to the halo bias relation stored in a file.

    Parameters
    ----------
    bias_file : str
        Path to the text file with columns '(log10(M), bias)'.
    degree : int
        Polynomial degree.
    drop_nonpositive : bool, optional
        If True, discard rows with non-positive bias values before fitting.

    Returns
    -------
    tuple
        '(coeffs, logM, bias)' where:
        - 'coeffs' are the fitted polynomial coefficients,
        - 'logM', 'bias' are the filtered data actually used in the fit.

    Raises
    ------
    ValueError
        If no valid rows remain after filtering, or if the polynomial degree is
        not compatible with the number of available points.

    Notes
    -----
    The condition 'degree < N_points' is enforced explicitly to avoid
    ill-defined polynomial fits.
    """
    logM, bias = load_bias_table(bias_file)

    if drop_nonpositive:
        mask = bias > 0.0
        logM = logM[mask]
        bias = bias[mask]

    if logM.size == 0:
        raise ValueError("No valid positive-bias points found in bias file.")

    if degree >= logM.size:
        raise ValueError(
            f"Requested polynomial degree {degree}, but only {logM.size} valid points are available."
        )

    coeffs = np.polyfit(logM, bias, degree)
    return coeffs, logM, bias

def get_hod_params_from_mu(mu, hodshape):
    """
    Return the derived analytical HOD parameters associated with a given 'mu'.

    Parameters
    ----------
    mu : float
        Characteristic HOD mass scale in log10 units.
    hodshape : int
        HOD family selector.

    Returns
    -------
    tuple
        '(M0, M1, alpha, sig, gamma)' as provided by
        'shape.get_hod_derived_params'.

    Notes
    -----
    This helper provides a thin wrapper that keeps the parameter-derivation
    interface local to this module.
    """
    M0, M1, alpha, sig, gamma = shape.get_hod_derived_params(mu, hodshape)
    return M0, M1, alpha, sig, gamma

def compress_halo_masses(
    logM_halos,
    nbins=120,
    logM_min=None,
    logM_max=None,
):
    """
    Compress a halo mass catalogue into a binned representation in log-mass.

    Parameters
    ----------
    logM_halos : array_like
        Halo log-masses.
    nbins : int, optional
        Number of equally spaced bins in log-mass.
    logM_min, logM_max : float, optional
        Explicit lower and upper mass limits. If not provided, the min/max of the
        input catalogue are used.

    Returns
    -------
    tuple
        '(centers, counts)' for the non-empty bins only.

    Notes
    -----
    This representation is used to accelerate repeated evaluations of smooth
    functions of halo mass, such as bias-weighted HOD integrals in the 'mu'
    solver.
    """
    logM_halos = np.asarray(logM_halos, dtype=float)

    if logM_min is None:
        logM_min = np.min(logM_halos)
    if logM_max is None:
        logM_max = np.max(logM_halos)

    edges = np.linspace(logM_min, logM_max, nbins + 1)
    counts, _ = np.histogram(logM_halos, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = counts > 0
    return centers[mask], counts[mask]

def satellite_shape_unit_from_mass(M, M0, M1, alpha):
    """
    Return the satellite unit-shape occupation evaluated in linear mass.

    This is the linear-mass counterpart of 'satellite_shape_unit', useful when
    the mass centres of binned catalogues are already available in linear units.

    Parameters
    ----------
    M : array_like
        Halo mass [Msun/h].
    M0 : float
        Satellite cutoff scale [Msun/h].
    M1 : float
        Satellite characteristic scale [Msun/h].
    alpha : float
        Satellite power-law slope.

    Returns
    -------
    ndarray
        Satellite unit-shape occupation evaluated at 'M'.
    """
    M = np.asarray(M, dtype=float)
    out = np.zeros_like(M, dtype=float)

    mask = M > M0
    out[mask] = ((M[mask] - M0) / M1) ** alpha
    return out

def compute_bias_integrals_binned(
    logM_centers,
    M_centers,
    counts,
    bias_centers,
    Lbox,
    mu,
    hodshape,
    ngal_target,
    fsat_target,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Compute binned HOD and bias-weighted integrals for a trial value of 'mu'.

    This function evaluates the quantities needed to predict the galaxy bias:

        I_c = (1/V) sum w_i g_cen
        J_c = (1/V) sum w_i b_i g_cen
        I_s = (1/V) sum w_i g_sat [* conformity weight if enabled]
        J_s = (1/V) sum w_i b_i g_sat [* conformity weight if enabled]

    where 'w_i' are the halo counts in each mass bin.

    Parameters
    ----------
    logM_centers : array_like
        Log-mass bin centres.
    M_centers : array_like
        Linear-mass bin centres.
    counts : array_like
        Number of halos in each bin.
    bias_centers : array_like
        Halo bias evaluated at the bin centres.
    Lbox : float
        Simulation box size [Mpc/h].
    mu : float
        Trial HOD mass scale.
    hodshape : int
        HOD family selector.
    ngal_target : float
        Target galaxy number density.
    fsat_target : float
        Target satellite fraction.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    tuple
        '(Ic, Is, Jc, Js, M0, M1, alpha, sig, gamma)'.

    Notes
    -----
    In conformity mode, the central amplitude implied by the target
    '(n_gal, f_sat)' at this 'mu' is first estimated as

        A_c(mu) = n_cen,target / I_c

    and used to build the effective satellite weight.
    """
    M0, M1, alpha, sig, gamma = get_hod_params_from_mu(mu, hodshape)

    gcen = central_shape_unit(logM_centers, mu, sig, gamma, hodshape)
    gsat = satellite_shape_unit_from_mass(M_centers, M0, M1, alpha)

    volume = Lbox ** 3
    weights = counts.astype(float)

    Ic = np.sum(weights * gcen) / volume
    Jc = np.sum(weights * bias_centers * gcen) / volume

    if conformity:
        ncen_target = (1.0 - fsat_target) * ngal_target
        Ac_mu = ncen_target / Ic
        pcen = Ac_mu * gcen
        conf_weight = K2_global + (K1_global - K2_global) * pcen

        Is = np.sum(weights * gsat * conf_weight) / volume
        Js = np.sum(weights * bias_centers * gsat * conf_weight) / volume
    else:
        Is = np.sum(weights * gsat) / volume
        Js = np.sum(weights * bias_centers * gsat) / volume
    
    return Ic, Is, Jc, Js, M0, M1, alpha, sig, gamma

def predict_bgal_given_mu_binned(
    logM_centers,
    M_centers,
    counts,
    bias_centers,
    Lbox,
    mu,
    hodshape,
    fsat_target,
    ngal_target,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Predict the large-scale galaxy bias for a trial 'mu' using binned halo masses.

    The prediction is based on the decomposition

        b_gal = (1 - f_sat) * (J_c / I_c) + f_sat * (J_s / I_s)

    where the central and satellite contributions are computed from the binned
    halo mass distribution.

    Parameters
    ----------
    logM_centers, M_centers, counts, bias_centers : array_like
        Binned halo-mass representation and corresponding halo bias values.
    Lbox : float
        Simulation box size [Mpc/h].
    mu : float
        Trial HOD mass scale.
    hodshape : int
        HOD family selector.
    fsat_target : float
        Target satellite fraction.
    ngal_target : float
        Target galaxy number density.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    tuple
        '(bgal, Ic, Is, Jc, Js, M0, M1, alpha, sig, gamma)'.

    Raises
    ------
    ValueError
        If the central or satellite integrals are not positive.
    """
    Ic, Is, Jc, Js, M0, M1, alpha, sig, gamma = compute_bias_integrals_binned(
        logM_centers=logM_centers,
        M_centers=M_centers,
        counts=counts,
        bias_centers=bias_centers,
        Lbox=Lbox,
        mu=mu,
        hodshape=hodshape,
        ngal_target=ngal_target,
        fsat_target=fsat_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
    )

    if Ic <= 0:
        raise ValueError("Ic is not positive while predicting bgal.")
    if Is <= 0:
        raise ValueError("Is is not positive while predicting bgal.")

    bgal = (1.0 - fsat_target) * (Jc / Ic) + fsat_target * (Js / Is)

    return bgal, Ic, Is, Jc, Js, M0, M1, alpha, sig, gamma

def mu_objective_binned(
    mu,
    logM_centers,
    M_centers,
    counts,
    bias_centers,
    Lbox,
    hodshape,
    fsat_target,
    bgal_target,
    ngal_target,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,

):
    """
    Objective function for solving 'mu' from a target galaxy bias.

    Parameters
    ----------
    mu : float
        Trial value of the HOD characteristic mass scale.
    logM_centers, M_centers, counts, bias_centers : array_like
        Binned halo representation and halo bias.
    Lbox : float
        Simulation box size [Mpc/h].
    hodshape : int
        HOD family selector.
    fsat_target : float
        Target satellite fraction.
    bgal_target : float
        Desired large-scale galaxy bias.
    ngal_target : float
        Target galaxy number density.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    float
        Difference 'bgal(mu) - bgal_target'.
    """
    bgal, *_ = predict_bgal_given_mu_binned(
        logM_centers=logM_centers,
        M_centers=M_centers,
        counts=counts,
        bias_centers=bias_centers,
        Lbox=Lbox,
        mu=mu,
        hodshape=hodshape,
        fsat_target=fsat_target,
        ngal_target=ngal_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
    )
    return bgal - bgal_target

def find_mu_bracket_binned(
    logM_centers,
    M_centers,
    counts,
    bias_centers,
    Lbox,
    hodshape,
    fsat_target,
    bgal_target,
    mu_min,
    mu_max,
    n_scan=25,
    ngal_target=None,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Scan a range of 'mu' values and search for a sign-changing bracket.

    Parameters
    ----------
    logM_centers, M_centers, counts, bias_centers : array_like
        Binned halo representation and halo bias.
    Lbox : float
        Simulation box size [Mpc/h].
    hodshape : int
        HOD family selector.
    fsat_target : float
        Target satellite fraction.
    bgal_target : float
        Desired large-scale galaxy bias.
    mu_min, mu_max : float
        Search interval for 'mu'.
    n_scan : int, optional
        Number of trial points in the coarse scan.
    ngal_target : float, optional
        Target galaxy number density.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    tuple
        '(mu_lo, mu_hi, mu_grid, f_grid, bgal_grid)'

    Notes
    -----
    - If an exact zero is found on the scan grid, '(mu_i, mu_i)' is returned.
    - If no sign change is found, '(None, None, mu_grid, f_grid, bgal_grid)' is returned.
    """
    mu_grid = np.linspace(mu_min, mu_max, n_scan)
    f_grid = np.empty_like(mu_grid)
    bgal_grid = np.empty_like(mu_grid)

    for i, mu in enumerate(mu_grid):
        bgal, *_ = predict_bgal_given_mu_binned(
            logM_centers=logM_centers,
            M_centers=M_centers,
            counts=counts,
            bias_centers=bias_centers,
            Lbox=Lbox,
            mu=mu,
            hodshape=hodshape,
            fsat_target=fsat_target,
            ngal_target=ngal_target,
            conformity=conformity,
            K1_global=K1_global,
            K2_global=K2_global,
        )
        bgal_grid[i] = bgal
        f_grid[i] = bgal - bgal_target

    # Look for bracket
    for i in range(len(mu_grid) - 1):
        if f_grid[i] == 0.0:
            return mu_grid[i], mu_grid[i], mu_grid, f_grid, bgal_grid
        if f_grid[i] * f_grid[i + 1] < 0:
            return mu_grid[i], mu_grid[i + 1], mu_grid, f_grid, bgal_grid

    # If no bracket is found, return None but with the evaluated grid for diagnostics
    return None, None, mu_grid, f_grid, bgal_grid

def solve_mu_bisection_binned(
    logM_centers,
    M_centers,
    counts,
    bias_centers,
    Lbox,
    hodshape,
    fsat_target,
    bgal_target,
    mu_lo,
    mu_hi,
    tol=1.0e-4,
    max_iter=100,
    ngal_target=None,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Solve for 'mu' by bisection once a sign-changing bracket has been found.

    Parameters
    ----------
    logM_centers, M_centers, counts, bias_centers : array_like
        Binned halo representation and halo bias.
    Lbox : float
        Simulation box size [Mpc/h].
    hodshape : int
        HOD family selector.
    fsat_target : float
        Target satellite fraction.
    bgal_target : float
        Desired large-scale galaxy bias.
    mu_lo, mu_hi : float
        Lower and upper bounds of a valid bracket.
    tol : float, optional
        Absolute tolerance in either objective value or bracket width.
    max_iter : int, optional
        Maximum number of bisection iterations.
    ngal_target : float, optional
        Target galaxy number density.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    float
        Best-fit value of 'mu'.

    Raises
    ------
    ValueError
        If the supplied interval does not bracket a sign change.
    """
    f_lo = mu_objective_binned(
        mu_lo,
        logM_centers,
        M_centers,
        counts,
        bias_centers,
        Lbox,
        hodshape,
        fsat_target,
        bgal_target,
        ngal_target=ngal_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
    )
    f_hi = mu_objective_binned(
        mu_hi,
        logM_centers,
        M_centers,
        counts,
        bias_centers,
        Lbox,
        hodshape,
        fsat_target,
        bgal_target,
        ngal_target=ngal_target,        
        conformity=conformity,
        K1_global=K1_global,       
        K2_global=K2_global,
    )

    if f_lo == 0.0:
        return mu_lo
    if f_hi == 0.0:
        return mu_hi
    if f_lo * f_hi > 0:
        raise ValueError("Bisection requires a sign change in the bracket.")

    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        f_mid = mu_objective_binned(
            mu_mid,
            logM_centers,
            M_centers,
            counts,
            bias_centers,
            Lbox,
            hodshape,
            fsat_target,
            bgal_target,
            ngal_target=ngal_target,
            conformity=conformity,
            K1_global=K1_global,
            K2_global=K2_global,
        )

        if abs(f_mid) < tol or abs(mu_hi - mu_lo) < tol:
            return mu_mid

        if f_lo * f_mid <= 0:
            mu_hi = mu_mid
            f_hi = f_mid
        else:
            mu_lo = mu_mid
            f_lo = f_mid

    return 0.5 * (mu_lo + mu_hi)

def solve_mu_Ac_As_from_targets(
    logM_halos,
    Lbox,
    hodshape,
    ngal_target,
    fsat_target,
    bgal_target,
    bias_coeffs,
    mu_min,
    mu_max,
    mu_tol,
    mu_max_iter,
    conformity=False,
    K1_global=1.0,
    K2_global=1.0,
):
    """
    Solve the full inverse calibration problem '(mu, A_c, A_s)' from targets.

    This routine performs the following steps:

    1. Compress the halo mass catalogue into a binned representation.
    2. Evaluate the halo bias polynomial on the binned mass centres.
    3. Search for a bracket in 'mu' such that 'bgal(mu) - bgal_target' changes sign.
    4. If a bracket exists, solve for 'mu' by bisection.
       Otherwise, retain the scan point with the smallest absolute residual.
    5. Derive the corresponding HOD parameters '(M0, M1, alpha, sig, gamma)'.
    6. Compute the final amplitudes '(A_c, A_s)' using the full halo catalogue.
    7. Return a summary dictionary with the solved parameters and diagnostics.

    Parameters
    ----------
    logM_halos : array_like
        Full halo mass catalogue in log10 units.
    Lbox : float
        Simulation box size [Mpc/h].
    hodshape : int
        HOD family selector.
    ngal_target : float
        Target galaxy number density.
    fsat_target : float
        Target satellite fraction.
    bgal_target : float
        Target large-scale galaxy bias.
    bias_coeffs : array_like
        Polynomial coefficients describing halo bias as a function of log-mass.
    mu_min, mu_max : float
        Search interval for 'mu'.
    mu_tol : float
        Bisection tolerance.
    mu_max_iter : int
        Maximum number of bisection iterations.
    conformity : bool, optional
        Whether to include conformity.
    K1_global, K2_global : float, optional
        Global conformity factors.

    Returns
    -------
    dict
        Dictionary containing the solved parameters and key diagnostics:
        'mu', 'Ac', 'As', 'bgal', 'Ic', 'Is', 'Jc', 'Js',
        'M0', 'M1', 'alpha', 'sig', 'gamma', 'mu_bracket'.

    Notes
    -----
    The final amplitudes are computed with the full halo catalogue rather than
    the compressed one, in order to preserve accuracy in the final model
    normalization.
    """
    # -------------------------------------------------
    # 1) Compress halo masses for fast mu solving
    # -------------------------------------------------
    logM_centers, counts = compress_halo_masses(
        logM_halos,
        nbins=120,   # puedes exponerlo luego como parámetro
    )
    M_centers = 10.0 ** logM_centers

    # Evaluate halo bias polynomial once on the binned centers
    bias_centers = halo_bias_poly(logM_centers, bias_coeffs)

    # -------------------------------------------------
    # 2) Find mu bracket using binned representation
    # -------------------------------------------------
    mu_lo, mu_hi, mu_grid, f_grid, bgal_grid = find_mu_bracket_binned(
        logM_centers=logM_centers,
        M_centers=M_centers,
        counts=counts,
        bias_centers=bias_centers,
        Lbox=Lbox,
        hodshape=hodshape,
        fsat_target=fsat_target,
        bgal_target=bgal_target,
        mu_min=mu_min,
        mu_max=mu_max,
        n_scan=25,
        ngal_target=ngal_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
    )

    if mu_lo is None:
        i_best = np.argmin(np.abs(f_grid))
        mu_sol = mu_grid[i_best]

        print("[mu-solver] No sign-change bracket found.")
        print(f"[mu-solver] Requested target bgal = {bgal_target:.6f}")
        print(f"[mu-solver] Scanned bgal range = [{np.min(bgal_grid):.6f}, {np.max(bgal_grid):.6f}]")
        print(f"[mu-solver] Using closest mu from scan: mu = {mu_sol:.6f}, bgal = {bgal_grid[i_best]:.6f}")
        mu_bracket = None
    else:
        if mu_lo == mu_hi:
            mu_sol = mu_lo
        else:
            mu_sol = solve_mu_bisection_binned(
                logM_centers=logM_centers,
                M_centers=M_centers,
                counts=counts,
                bias_centers=bias_centers,
                Lbox=Lbox,
                hodshape=hodshape,
                fsat_target=fsat_target,
                bgal_target=bgal_target,
                mu_lo=mu_lo,
                mu_hi=mu_hi,
                tol=mu_tol,
                max_iter=mu_max_iter,
                ngal_target=ngal_target,
                conformity=conformity,
                K1_global=K1_global,
                K2_global=K2_global,
            )
        mu_bracket = (mu_lo, mu_hi)

    # -------------------------------------------------
    # 4) Compute final HOD params from mu
    # -------------------------------------------------
    M0, M1, alpha, sig, gamma = get_hod_params_from_mu(mu_sol, hodshape)

    # Final Ac, As using full halo catalog (more accurate)
    Ac, As, Ic, Is = compute_Ac_As_from_halo_masses(
        logM_halos=logM_halos,
        Lbox=Lbox,
        mu=mu_sol,
        sig=sig,
        gamma=gamma,
        hodshape=hodshape,
        M0=M0,
        M1=M1,
        alpha=alpha,
        ngal_target=ngal_target,
        fsat_target=fsat_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
    )

    # Final bgal prediction using binned fast representation
    bgal, Ic2, Is2, Jc, Js, _, _, _, _, _ = predict_bgal_given_mu_binned(
        logM_centers=logM_centers,
        M_centers=M_centers,
        counts=counts,
        bias_centers=bias_centers,
        Lbox=Lbox,
        mu=mu_sol,
        hodshape=hodshape,
        fsat_target=fsat_target,
        conformity=conformity,
        K1_global=K1_global,
        K2_global=K2_global,
        ngal_target=ngal_target,
    )

    return {
        "mu": mu_sol,
        "Ac": Ac,
        "As": As,
        "bgal": bgal,
        "Ic": Ic2,
        "Is": Is2,
        "Jc": Jc,
        "Js": Js,
        "M0": M0,
        "M1": M1,
        "alpha": alpha,
        "sig": sig,
        "gamma": gamma,
        "mu_bracket": (mu_lo, mu_hi),
    }