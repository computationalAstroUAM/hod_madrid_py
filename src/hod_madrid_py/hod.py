# hod.py

"""
Main execution engine for HOD-based mock-galaxy generation.

Overview
--------
This module coordinates the full construction of a mock galaxy catalogue from
an input halo catalogue. It connects the different physical components of the
pipeline:

- halo occupation statistics,
- satellite radial placement,
- satellite velocity assignment,
- optional conformity corrections,
- optional derivation of analytical amplitudes and characteristic mass scale,
- chunk-wise I/O for large catalogues.

Architecture
------------
The workflow is organized in three layers:

1. Per-halo realization
   For each halo, the code decides whether it hosts a central galaxy and how
   many satellites it contains, then assigns positions and velocities to all
   galaxies associated with that halo.

2. Per-chunk processing
   Haloes are read in chunks from the input catalogue in order to control
   memory usage and allow large-volume processing.

3. Full-run orchestration
   The top-level driver loads the required auxiliary inputs, validates the
   configuration, optionally derives analytical HOD parameters, and launches
   the chunk-wise realization of the full mock.

Conventions
-----------
- Halo masses are handled internally both in linear form 'M' and logarithmic
  form 'logM', depending on the requirements of each submodule.
- Positions are expressed in comoving Mpc/h.
- Velocities are expressed in km/s.
- Galaxy catalogues are written in the internal 16-column format adopted by
  the repository.
"""
import numpy as np
from numba import jit
import time
import src.hod_madrid_py.hod_amplitudes as amp

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self):
        t = time.perf_counter()
        dt = t - self.t0
        self.t0 = t
        return dt

import src.hod_madrid_py.hod_io as io
import src.hod_madrid_py.hod_const as c
import src.hod_madrid_py.hod_r_profile as rp
import src.hod_madrid_py.hod_v_profile as vp
import src.hod_madrid_py.hod_shape as shape
import src.hod_madrid_py.hod_pdf as pdf
import src.hod_madrid_py.hod_cosmology as cosm

def process_halos_chunk_analytical(x, y, z, vx, vy, vz, logM, Rvir, Rs, halo_ids, random_seeds,
                        hodshape, mu, Ac, As,alpha, sig, gamma, vfact, beta,
                        K, vt, vtdisp, M0, M1, omega_M, Lbox, zsnap,
                        alpha_r, beta_r, N0, r0, Rmax, kappa_r, extended_NFW = True,
                        vr1=None, vr2=None, vr3=None, mu1=None, mu2=None, mu3=None,
                        sigma1=None, sigma2=None, sigma3=None, extended_vp=True,
                        v0_tan=None, epsilon_tan=None, omega_tan=None, delta_tan=None,
                        read_concentrations=False, conformity=False, K1_global=1.0, K2_global=1.0):
    
    """
    Generate central and satellite galaxies for a chunk of halos using an 
    analytical HOD (Halo Occupation Distribution) prescription.

    For each halo in the input arrays, this function:
        1. Computes the expected number of central and satellite galaxies 
           using the chosen analytical HOD model.
        2. Places the central galaxy at the halo center (with halo velocity).
        3. Samples satellite positions from either a standard or extended NFW 
           radial profile.
        4. Samples satellite velocities from either a virial velocity model 
           or extended analytical velocity distributions (radial Gaussian mixture 
           + tangential law).
        5. Applies periodic boundary conditions to ensure galaxies remain inside 
           the simulation box.
        6. Stores galaxy properties in a preallocated array.

    Parameters
    ----------
    x, y, z : array_like
        Halo center positions [Mpc/h].
    vx, vy, vz : array_like
        Halo bulk velocities [km/s].
    logM : array_like
        Log10 of halo mass [Msun/h].
    halo_ids : array_like
        Unique halo identifiers.
    random_seeds : array_like
        Seed values for reproducible random number generation per halo.
    hodshape : int
        Identifier for the HOD shape (1, 2, or 3).
    mu, Ac, As, alpha, sig, gamma : float
        HOD model parameters.
    vfact, beta, K, vt, vtdisp, M0, M1 : float
        Additional HOD and velocity model parameters.
    omega_M : float
        Cosmological matter density parameter.
    Lbox : float
        Simulation box size [Mpc/h].
    zsnap : float
        Snapshot redshift.
    alpha_r, beta_r, N0, r0, kappa_r : float
        Extended NFW radial profile parameters.
    extended_NFW : bool, optional
        If True, use extended NFW radial profile; otherwise use standard NFW.
    vr1, vr2, vr3 : float, optional
        Amplitudes (weights) of the radial velocity Gaussian mixture.
    mu1, mu2, mu3 : float, optional
        Means of the radial velocity Gaussian mixture [km/s].
    sigma1, sigma2, sigma3 : float, optional
        Standard deviations of the radial velocity Gaussian mixture [km/s].
    extended_vp : bool, optional
        If True, use extended analytical velocity profile; 
        otherwise use virial theorem velocity.
    v0_tan, epsilon_tan, omega_tan, delta_tan : float, optional
        Parameters of the tangential velocity distribution.

    Returns
    -------
    galaxies : ndarray of shape (Ngal, 16)
        Array containing galaxy properties:
            [0-2]  : x, y, z positions [Mpc/h]
            [3-5]  : vx, vy, vz velocities [km/s]
            [6]    : host halo mass [Msun/h]
            [7]    : number of satellites in the halo
            [8-10] : velocity offsets Dvx, Dvy, Dvz
            [11-13]: position offsets Dx, Dy, Dz
            [14]   : host halo ID
            [15]   : is_central flag (1 = central, 0 = satellite)
    Notes
    -----
    - Satellite positions are sampled isotropically.
    - Periodic boundary conditions are applied to all positions.
    - Randomness is controlled via `random_seeds` to ensure reproducibility.
    """

    n_halos = len(x)
    t_occ = 0.0
    t_cen = 0.0
    t_rp  = 0.0
    t_vp  = 0.0
    t_sat_write = 0.0
    t_over = 0.0
    t_norm = 0.0
    
    # Start with a reasonable initial estimate but allow growth
    initial_estimate = n_halos * 50 
    max_galaxies = int(initial_estimate)
    
    # Pre-allocate output arrays
    galaxies = np.zeros((max_galaxies, 16))
    galaxy_count = 0
    
    t0 = time.perf_counter()
    occ_rng = np.random.default_rng(int(random_seeds[0]))
    Ncen_arr, Nsat_arr, has_gal = shape.compute_hod_arrays(logM,mu, Ac, As, alpha, sig, gamma,M0, M1, hodshape, beta, conformity, K1_global, K2_global, occ_rng)
    t_occ = time.perf_counter() - t0 

    t_loop_start = time.perf_counter()
    for i in range(n_halos):
        if has_gal[i] == 0:
            continue
        
        # Set random seed for this halo
        rng_halo = np.random.default_rng(int(random_seeds[i]))

        # Extract halo properties
        halo_x, halo_y, halo_z = x[i], y[i], z[i]
        halo_vx, halo_vy, halo_vz = vx[i], vy[i], vz[i]
        halo_logM = logM[i]
        halo_Rvir = Rvir[i]
        halo_Rs = Rs[i]
        halo_id = halo_ids[i]
        M = 10.0**halo_logM
        Ncen = Ncen_arr[i]
        Nsat = Nsat_arr[i]

        # Check if we need to grow the array before processing this halo
        galaxies_needed = Ncen + Nsat
        if galaxy_count + galaxies_needed >= max_galaxies:

            # Grow array by doubling size or adding what we need, whichever is larger
            new_size = int(max(max_galaxies * 2, galaxy_count + galaxies_needed + 1000))
            new_galaxies = np.zeros((new_size, 16))
            new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
            galaxies = new_galaxies
            max_galaxies = new_size
        
        # Write central galaxy if present
        if Ncen == 1:
            t0 = time.perf_counter()
            if galaxy_count >= max_galaxies:
                # Grow array
                new_size = int(max(max_galaxies * 2, galaxy_count + 1000))
                new_galaxies = np.zeros((new_size, 16))
                new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
                galaxies = new_galaxies
                max_galaxies = new_size

            galaxies[galaxy_count, 0] = halo_x
            galaxies[galaxy_count, 1] = halo_y
            galaxies[galaxy_count, 2] = halo_z
            galaxies[galaxy_count, 3] = halo_vx
            galaxies[galaxy_count, 4] = halo_vy
            galaxies[galaxy_count, 5] = halo_vz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = 0.0        # Dvx (no offset for central)
            galaxies[galaxy_count, 9] = 0.0        # Dvy
            galaxies[galaxy_count, 10] = 0.0       # Dvz
            galaxies[galaxy_count, 11] = 0.0       # Dx
            galaxies[galaxy_count, 12] = 0.0       # Dy
            galaxies[galaxy_count, 13] = 0.0       # Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 1.0       # is_central == 1
            galaxy_count += 1

            t_cen += time.perf_counter() - t0
        # Generate satellite galaxies
        for j in range(Nsat):
            if galaxy_count >= max_galaxies:
                # Grow array
                new_size = int(max(max_galaxies * 2, galaxy_count + 1000))
                new_galaxies = np.zeros((new_size, 16))
                new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
                galaxies = new_galaxies
                max_galaxies = new_size

            # Generate position and velocity offsets
            if extended_NFW == True:
                t0 = time.perf_counter()
                Dx, Dy, Dz = rp.generate_extended_position(M, zsnap, omega_M, c.rho_crit, cosm.Delta_vir, halo_Rvir, Rmax, r0, alpha_r, beta_r, kappa_r, N0, rng=rng_halo)
                t_rp += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                Rvir_use = rp.get_effective_Rvir(M, zsnap, omega_M, halo_Rvir)
                c_eff = rp.get_effective_concentration(M=M,zsnap=zsnap,Rvir=Rvir_use,Rs=halo_Rs,K=K,read_concentrations=read_concentrations)
                Dx, Dy, Dz = rp.generate_nfw_position_from_concentration(Rvir=Rvir_use,c_eff=c_eff,rng=rng_halo)
                t_rp += time.perf_counter() - t0

            if extended_vp == True:
                t0 = time.perf_counter()
                r_hat = np.array([Dx, Dy, Dz])
                r_hat /= np.linalg.norm(r_hat)
                t_norm += time.perf_counter() - t0

                t0 = time.perf_counter()
                Dvx, Dvy, Dvz = vp.sample_velocity_analytic(r_hat, vr1, vr2, vr3, mu1, mu2, mu3, sigma1, sigma2, sigma3, v0_tan, epsilon_tan, omega_tan, delta_tan, vtan_min=0.0, vtan_max=2000.0, rng=rng_halo)
                t_vp += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                Dvx, Dvy, Dvz = vp.generate_virial_velocity(M, zsnap, omega_M, rng=rng_halo)
                t_vp += time.perf_counter() - t0

            t0 = time.perf_counter()
            # Apply periodic boundary conditions
            xgal = (halo_x + Dx) % Lbox
            ygal = (halo_y + Dy) % Lbox
            zgal = (halo_z + Dz) % Lbox
            
            # Store satellite galaxy
            galaxies[galaxy_count, 0] = xgal
            galaxies[galaxy_count, 1] = ygal
            galaxies[galaxy_count, 2] = zgal
            galaxies[galaxy_count, 3] = halo_vx + Dvx
            galaxies[galaxy_count, 4] = halo_vy + Dvy
            galaxies[galaxy_count, 5] = halo_vz + Dvz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = Dvx
            galaxies[galaxy_count, 9] = Dvy
            galaxies[galaxy_count, 10] = Dvz
            galaxies[galaxy_count, 11] = Dx
            galaxies[galaxy_count, 12] = Dy
            galaxies[galaxy_count, 13] = Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 0.0        # is_central == 0
            galaxy_count += 1
            t_sat_write += time.perf_counter() - t0
    t_over = time.perf_counter() - t_loop_start
    
    overhead = t_over - (t_occ + t_cen + t_rp + t_norm + t_vp + t_sat_write)

    print(
        f"[analytical internal] "
        f"occ={t_occ:.3f}s | "
        f"cen={t_cen:.3f}s | "
        f"rp={t_rp:.3f}s | "
        f"norm={t_norm:.3f}s | "
        f"vp={t_vp:.3f}s | "
        f"write_sat={t_sat_write:.3f}s | "
        f"overhead={overhead:.3f}s | "
        f"loop_total={t_over:.3f}s"
    )


    # Return only the filled portion of the array
    return galaxies[:galaxy_count, :]

def process_halos_chunk_bins(x, y, z, vx, vy, vz, logM, Rvir, Rs, halo_ids, random_seeds,
                        vfact, beta, K, vt, vtdisp, omega_M, Lbox, zsnap,
                        alpha_r, beta_r, N0, r0, Rmax, kappa_r, extended_NFW = True,
                        analytical_vp = None, analytical_rp = None,
                        conformity = False, M_min = None, M_max = None, Ncen_mean = None, Nsat_mean = None,
                        vr_bin_edges = None, vr_probs = None, vtan_bin_edges = None, vtan_probs = None,
                        r_bin_edges=None, probs_r=None, Nsat_r=None, K1_global=None, K2_global=None,
                        vr1=None, vr2=None, vr3=None, mu1=None, mu2=None, mu3=None,
                        sigma1=None, sigma2=None, sigma3=None, extended_vp=True,
                        v0_tan=None, epsilon_tan=None, omega_tan=None, delta_tan=None,
                        read_concentrations=False):
    
    """
    Generate central and satellite galaxies for a chunk of halos using 
    **binned empirical HOD inputs** (rather than an analytical HOD model).

    For each halo in the input arrays, this function:
        1. Determines the expected number of centrals and satellites by looking up
           the appropriate halo-mass bin (M_min, M_max, Ncen_mean, Nsat_mean).
        2. Assigns a central galaxy probabilistically from Ncen_mean (Bernoulli trial),
           with optional conformity corrections (K1_global, K2_global).
        3. Samples the number of satellites from the chosen satellite-number PDF
           (Poisson, nearest-integer, binomial, or negative-binomial depending on β).
        4. Assigns satellite positions either from analytical radial profiles 
           (standard or extended NFW) or from an empirical radial histogram.
        5. Assigns satellite velocities either from analytical models 
           (Gaussian mixture radial + extended tangential law, or virial theorem) 
           or from empirical velocity histograms.
        6. Applies periodic boundary conditions to all galaxy positions.
        7. Stores galaxy properties in a structured array.

    Parameters
    ----------
    x, y, z : array_like
        Halo center positions [Mpc/h].
    vx, vy, vz : array_like
        Halo bulk velocities [km/s].
    logM : array_like
        Log10 of halo mass [Msun/h].
    halo_ids : array_like
        Unique halo identifiers.
    random_seeds : array_like
        Seed values for reproducible random number generation per halo.

    vfact, beta, K, vt, vtdisp : float
        Parameters for velocity and satellite-number distributions.
    omega_M : float
        Cosmological matter density parameter.
    Lbox : float
        Simulation box size [Mpc/h].
    zsnap : float
        Snapshot redshift.

    Radial profiles
    ---------------
    alpha_r, beta_r, N0, r0, kappa_r : float
        Extended NFW radial profile parameters.
    extended_NFW : bool, optional
        If True, use extended NFW profile; otherwise standard NFW.
    analytical_rp : bool, optional
        If True, use analytical radial profile; otherwise use empirical histogram.
    r_bin_edges, probs_r, Nsat_r : array_like, optional
        Empirical radial histogram inputs.

    Velocity profiles
    -----------------
    analytical_vp : bool, optional
        If True, use analytical velocity profile; otherwise empirical.
    vr1, vr2, vr3 : float, optional
        Amplitudes (weights) for Gaussian mixture of radial velocities.
    mu1, mu2, mu3 : float, optional
        Means of Gaussian mixture [km/s].
    sigma1, sigma2, sigma3 : float, optional
        Standard deviations of Gaussian mixture [km/s].
    extended_vp : bool, optional
        If True, use extended tangential velocity distribution; otherwise virial.
    v0_tan, epsilon_tan, omega_tan, delta_tan : float, optional
        Parameters of the tangential velocity law.
    vr_bin_edges, vr_probs : array_like, optional
        Empirical radial velocity histogram inputs.
    vtan_bin_edges, vtan_probs : array_like, optional
        Empirical tangential velocity histogram inputs.

    HOD occupation
    --------------
    M_min, M_max : array_like
        Halo mass bin edges.
    Ncen_mean, Nsat_mean : array_like
        Mean central and satellite occupations per bin.
    conformity : bool, optional
        If True, apply conformity correction to Nsat expectations.
    K1_global, K2_global : float, optional
        Conformity adjustment factors.

    Returns
    -------
    galaxies : ndarray of shape (Ngal, 16)
        Array containing galaxy properties:
            [0-2]  : x, y, z positions [Mpc/h]
            [3-5]  : vx, vy, vz velocities [km/s]
            [6]    : host halo mass [Msun/h]
            [7]    : number of satellites in the halo
            [8-10] : velocity offsets Dvx, Dvy, Dvz
            [11-13]: position offsets Dx, Dy, Dz
            [14]   : host halo ID
            [15]   : is_central flag (1 = central, 0 = satellite)

    Notes
    -----
    - This function complements `process_halos_chunk_analytical` by relying on 
      **binned/empirical occupation functions** instead of analytical formulas.
    - Randomness is controlled by `random_seeds` to ensure reproducibility.
    - Both radial and velocity distributions can be chosen analytically or empirically.
    """
    t_occ = 0.0     # ocupación (Ncen / Nsat)
    t_rp = 0.0      # radial profile
    t_vp = 0.0      # velocity profile
    n_halos = len(x)
    
    # Start with a reasonable initial estimate but allow growth
    initial_estimate = n_halos * 50 
    max_galaxies = int(initial_estimate)
    
    # Pre-allocate output arrays
    galaxies = np.zeros((max_galaxies, 16))
    galaxy_count = 0

    rng_occ = np.random.default_rng(int(random_seeds[0]))
    Ncen_arr, Nsat_arr, has_hal = pdf.compute_hod_arrays_binned(logM, M_max, Ncen_mean, Nsat_mean, beta, conformity, K1_global, K2_global, rng_occ)

    for i in range(n_halos):
        if has_hal[i] == 0:
            continue
        t0 = time.perf_counter()
        # Extract halo properties
        halo_x, halo_y, halo_z = x[i], y[i], z[i]
        halo_vx, halo_vy, halo_vz = vx[i], vy[i], vz[i]
        halo_logM = logM[i]
        halo_Rvir = Rvir[i]
        halo_Rs = Rs[i]
        halo_id = halo_ids[i]

        # Set random seed for this halo
        rng_halo = np.random.default_rng(int(random_seeds[i]))
        
        # Convert to linear mass
        M = 10.0**halo_logM
        Ncen = Ncen_arr[i]
        Nsat = Nsat_arr[i]

        '''
        k = np.searchsorted(M_max, halo_logM)
        if k == len(M_max):
            Ncen = 0
            Nsat = 0
        else:
            mean_Ncen = Ncen_mean[k]
            mean_Nsat = Nsat_mean[k]

        #idx = np.where((halo_logM >= M_min) & (halo_logM < M_max))[0]
        #if len(idx) == 0:
            #Ncen = 0
            #Nsat = 0
        #else:
            #i = idx[0]
            #mean_Ncen = Ncen_mean[i]
            #mean_Nsat = Nsat_mean[i]

            # Bernouilli trial for central galaxy
            r = np.random.rand()
            if r < mean_Ncen:
                Ncen = 1
                if conformity:
                    mean_Nsat_adj = K1_global * mean_Nsat
                else:
                    mean_Nsat_adj = mean_Nsat
            else:
                Ncen = 0
                if conformity:
                    mean_Nsat_adj = K2_global * mean_Nsat
                else:
                    mean_Nsat_adj = mean_Nsat

            # PDF sampling for satellites
            if beta < -1.0:
                Nsat = pdf.next_integer(mean_Nsat_adj)
            elif beta <= 0.0 and beta >= -1.0/171.0:
                Nsat = pdf.poisson_sample(mean_Nsat_adj)
            elif beta < -1.0/171.0 and beta >= -1.0:
                Nsat = pdf.binomial_sample(mean_Nsat_adj, beta)
            elif beta > 0.0:
                Nsat = pdf.neg_binomial_sample(mean_Nsat_adj, beta)
            else:
                Nsat = pdf.poisson_sample(mean_Nsat_adj)
        '''
        t_occ += time.perf_counter() - t0
        # Check if we need to grow the array before processing this halo
        galaxies_needed = Ncen + Nsat
        if galaxy_count + galaxies_needed >= max_galaxies:
            # Grow array by doubling size or adding what we need, whichever is larger
            new_size = int(max(max_galaxies * 2, galaxy_count + galaxies_needed + 1000))
            new_galaxies = np.zeros((new_size, 16))
            new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
            galaxies = new_galaxies
            max_galaxies = new_size
        
        # Write central galaxy if present
        if Ncen == 1:
            if galaxy_count >= max_galaxies:
                # Grow array
                new_size = int(max(max_galaxies * 2, galaxy_count + 1000))
                new_galaxies = np.zeros((new_size, 16))
                new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
                galaxies = new_galaxies
                max_galaxies = new_size

            galaxies[galaxy_count, 0] = halo_x
            galaxies[galaxy_count, 1] = halo_y
            galaxies[galaxy_count, 2] = halo_z
            galaxies[galaxy_count, 3] = halo_vx
            galaxies[galaxy_count, 4] = halo_vy
            galaxies[galaxy_count, 5] = halo_vz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = 0.0         # Dvx (no offset for central)
            galaxies[galaxy_count, 9] = 0.0         # Dvy
            galaxies[galaxy_count, 10] = 0.0        # Dvz
            galaxies[galaxy_count, 11] = 0.0        # Dx
            galaxies[galaxy_count, 12] = 0.0        # Dy
            galaxies[galaxy_count, 13] = 0.0        # Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 1.0        # is_central == 1
            galaxy_count += 1

        # Generate satellite galaxies
        for j in range(Nsat):
            if galaxy_count >= max_galaxies:
                # Grow array
                new_size = int(max(max_galaxies * 2, galaxy_count + 1000))
                new_galaxies = np.zeros((new_size, 16))
                new_galaxies[:galaxy_count, :] = galaxies[:galaxy_count, :]
                galaxies = new_galaxies
                max_galaxies = new_size

            # Generate position and velocity offsets
            if analytical_rp == True:
                if extended_NFW == True:
                    t0 = time.perf_counter()
                    Dx, Dy, Dz = rp.generate_extended_position(M, zsnap, omega_M, c.rho_crit, cosm.Delta_vir, halo_Rvir, Rmax, r0, alpha_r, beta_r, kappa_r, N0, rng=rng_halo)
                    t_rp += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    Rvir_use = rp.get_effective_Rvir(M, zsnap, omega_M, halo_Rvir)
                    c_eff = rp.get_effective_concentration(M=M,zsnap=zsnap,Rvir=Rvir_use,Rs=halo_Rs,K=K,read_concentrations=read_concentrations)
                    Dx, Dy, Dz = rp.generate_nfw_position_from_concentration(Rvir=Rvir_use,c_eff=c_eff,rng=rng_halo)
                    t_rp += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                Dx, Dy, Dz = rp.sample_empirical_position(r_bin_edges, probs_r, rng=rng_halo)
                t_rp += time.perf_counter() - t0

            if analytical_vp == True:
                if extended_vp == True:
                    r_hat = np.array([Dx, Dy, Dz])
                    r_hat /= np.linalg.norm(r_hat)

                    t0 = time.perf_counter()
                    Dvx, Dvy, Dvz = vp.sample_velocity_analytic(r_hat, vr1, vr2, vr3, mu1, mu2, mu3, sigma1, sigma2, sigma3, v0_tan, epsilon_tan, omega_tan, delta_tan, vtan_min=0.0, vtan_max=2000.0, rng=rng_halo)
                    t_vp += time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    Dvx, Dvy, Dvz = vp.generate_virial_velocity(M, zsnap, omega_M, rng=rng_halo)
                    t_vp += time.perf_counter() - t0
            else:
                r_hat = np.array([Dx, Dy, Dz])
                r_hat /= np.linalg.norm(r_hat)

                t0 = time.perf_counter()
                Dvx, Dvy, Dvz = vp.sample_empirical_velocity(
                    r_hat,
                    vr_bin_edges, vr_probs,
                    vtan_bin_edges, vtan_probs, rng=rng_halo
                    )
                t_vp += time.perf_counter() - t0

            # Apply periodic boundary conditions
            xgal = (halo_x + Dx) % Lbox
            ygal = (halo_y + Dy) % Lbox
            zgal = (halo_z + Dz) % Lbox
            
            # Store satellite galaxy
            galaxies[galaxy_count, 0] = xgal
            galaxies[galaxy_count, 1] = ygal
            galaxies[galaxy_count, 2] = zgal
            galaxies[galaxy_count, 3] = halo_vx + Dvx
            galaxies[galaxy_count, 4] = halo_vy + Dvy
            galaxies[galaxy_count, 5] = halo_vz + Dvz
            galaxies[galaxy_count, 6] = M
            galaxies[galaxy_count, 7] = Nsat
            galaxies[galaxy_count, 8] = Dvx
            galaxies[galaxy_count, 9] = Dvy
            galaxies[galaxy_count, 10] = Dvz
            galaxies[galaxy_count, 11] = Dx
            galaxies[galaxy_count, 12] = Dy
            galaxies[galaxy_count, 13] = Dz
            galaxies[galaxy_count, 14] = halo_id
            galaxies[galaxy_count, 15] = 0.0       # is_central == 0
            galaxy_count += 1

    # Return only the filled portion of the array
    total = t_occ + t_rp + t_vp
    print(
        f"[bins internal] "
        f"occ={t_occ:.3f}s | "
        f"rp={t_rp:.3f}s | "
        f"vp={t_vp:.3f}s | "
        f"sum={total:.3f}s"
    )
    return galaxies[:galaxy_count, :]


def process_halo_file_chunked(params, chunk_size=c.chunk_size, verbose=False, M_min=None, M_max=None,
                              Ncen_mean=None, Nsat_mean=None, vr_bin_edges=None, vr_probs=None,
                              vtan_bin_edges=None, vtan_probs=None, r_bin_edges=None, probs_r=None, Nsat_r=None,
                              K1_global=None, K2_global=None, Rvir=None, Rs=None):
    
    """
    Process a halo catalogue in chunks to generate mock galaxy catalogues 
    using Halo Occupation Distribution (HOD) models.

    This function is the top-level driver for chunk-wise reading and processing 
    of halo catalogues. It efficiently handles large input files by splitting 
    them into manageable chunks, generating galaxies for each halo, and writing 
    the results incrementally to disk.

    Workflow
    --------
    1. Halo data are read in chunks from `params.infile` (txt or h5 depending on `params.ftype`).
    2. For each chunk, halo properties (x, y, z, vx, vy, vz, logM, ID) are extracted.
    3. Reproducible random seeds are generated per halo (from `params.seed`).
    4. Depending on `params.analytical_shape`:
        - **Analytical shape mode**: calls `process_halos_chunk_analytical`. 
          The *HOD shape* is fully analytical, and the radial/velocity profiles 
          are always analytical (standard NFW or extended).
        - **Empirical shape mode**: calls `process_halos_chunk_bins`. 
          The *HOD shape* is taken from empirical/binned measurements 
          (e.g. from HODfit2sim output file). However, the radial and velocity profiles 
          can still be chosen as either analytical or empirical (from histograms).
    5. Central and satellite galaxies are generated with positions, velocities, 
       halo IDs, and stored in a 16-column array.
    6. Results are written incrementally to `params.outfile`.
    7. Progress is optionally printed (`verbose=True`).

    Parameters
    ----------
    params : Namespace or object
        Container of global HOD parameters and file paths. Must include:
            infile, outfile, ftype, seed,
            analytical_shape, analytical_rp, analytical_vp,
            extended_NFW, extended_vp,
            hodshape, mu, Ac, As, alpha, sig, gamma, M0, M1,
            vfact, beta, K, vt, vtdisp,
            alpha_r, beta_r, N0, r0, kappa_r,
            vr1, vr2, vr3, mu1, mu2, mu3, sigma1, sigma2, sigma3,
            v0_tan, epsilon_tan, omega_tan, delta_tan,
            conformity, omega_M, Lbox, zsnap.
    chunk_size : int, optional
        Number of halos per chunk (default: `c.chunk_size`).
    verbose : bool, optional
        If True, print progress reports during processing.
    M_min, M_max : array_like, optional
        Halo mass bin edges for empirical HOD shape mode.
    Ncen_mean, Nsat_mean : array_like, optional
        Mean central and satellite numbers per halo-mass bin (empirical shape mode).
    vr_bin_edges, vr_probs : array_like, optional
        Radial velocity histogram inputs (empirical velocity profile).
    vtan_bin_edges, vtan_probs : array_like, optional
        Tangential velocity histogram inputs (empirical velocity profile).
    r_bin_edges, probs_r, Nsat_r : array_like, optional
        Radial histogram inputs (empirical radial profile).
    K1_global, K2_global : float, optional
        Conformity adjustment factors for satellite occupation.

    Returns
    -------
    line_count : int or None
        Total number of halos processed. Returns None if an error occurred.

    Notes
    -----
    - The choice between **analytical** and **empirical** refers only to the HOD shape.
    - In empirical-shape mode, radial and velocity profiles can still be configured 
      as analytical (NFW/extended, Gaussian/exponential laws) or empirical (histograms).
    - In analytical-shape mode, both the HOD shape and the radial/velocity profiles 
      are always analytical.
    - Reproducibility is ensured by assigning one RNG seed per halo, derived 
      from a global seed (`params.seed`).

    See Also
    --------
    process_halos_chunk_analytical : Halo processing with analytical HOD shape.
    process_halos_chunk_bins        : Halo processing with empirical HOD shape 
                                      (radial/velocity profiles can be analytical or empirical).
    """

    start_time = time.time()
    line_count = 0
    galaxy_count = 0
    

    try:
        with open(params.outfile, 'w') as f_out:
            # Process file in chunks
            timer = Timer()
            for chunk_data in io.read_halo_data_chunked(params.infile, params.ftype, chunk_size):
                t_read = timer.lap()
                if len(chunk_data) == 0:
                    continue
                
                # Extract arrays from chunk
                x = chunk_data[:, 0]
                y = chunk_data[:, 1]
                z = chunk_data[:, 2]
                vx = chunk_data[:, 3]
                vy = chunk_data[:, 4]
                vz = chunk_data[:, 5]
                logM = chunk_data[:, 6]
                Rvir = chunk_data[:, 7]
                Rs = chunk_data[:, 8]
                halo_ids = chunk_data[:, 9].astype(np.int64)
                t_extract = timer.lap()

                Rvir = Rvir/1000.0 # Convert from kpc/h to Mpc/h
                Rs = Rs/1000.0       # Convert from kpc/h to Mpc/h

                # Generate random seeds for each halo
                random_seeds = np.random.randint(0, 2**31, size=len(x))
                t_seed = timer.lap()
                
                
                # Process chunk with Numba
                
                if params.analytical_shape:
                    galaxies = process_halos_chunk_analytical(
                                x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, logM=logM, Rvir=Rvir, Rs=Rs, halo_ids=halo_ids, random_seeds=random_seeds,
                                hodshape=params.hodshape, mu=params.mu, Ac=params.Ac, As=params.As,
                                alpha=params.alpha, sig=params.sig, gamma=params.gamma,
                                vfact=params.vfact, beta=params.beta,
                                K=params.K, vt=params.vt, 
                                vtdisp=params.vtdisp, M0=params.M0, M1=params.M1,
                                omega_M=params.omega_M, Lbox=params.Lbox, zsnap=params.zsnap,
                                alpha_r=params.alpha_r, beta_r=params.beta_r, N0=params.N0, r0=params.r0, Rmax=params.Rmax,
                                kappa_r=params.kappa_r, extended_NFW=params.extended_NFW,
                                vr1= params.vr1, vr2=params.vr2, vr3=params.vr3,
                                mu1=params.mu1, mu2=params.mu2, mu3=params.mu3,
                                sigma1=params.sigma1, sigma2=params.sigma2, sigma3=params.sigma3,
                                extended_vp=params.extended_vp, v0_tan=params.v0_tan, epsilon_tan=params.epsilon_tan,
                                omega_tan=params.omega_tan, delta_tan=params.delta_tan,
                                conformity=params.conformity, K1_global=K1_global, K2_global=K2_global,
                                read_concentrations=params.read_concentrations
                                )
                else:
                    galaxies = process_halos_chunk_bins(
                                x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, logM=logM, Rvir=Rvir, Rs=Rs, halo_ids=halo_ids, random_seeds=random_seeds,
                                vfact=params.vfact, beta=params.beta, K=params.K, vt=params.vt, vtdisp=params.vtdisp,
                                omega_M=params.omega_M, Lbox=params.Lbox, zsnap=params.zsnap,
                                alpha_r=params.alpha_r, beta_r=params.beta_r,
                                N0=params.N0,
                                r0=params.r0, Rmax=params.Rmax, kappa_r=params.kappa_r, extended_NFW=params.extended_NFW,
                                analytical_vp=params.analytical_vp, analytical_rp=params.analytical_rp,
                                vr1=params.vr1, vr2=params.vr2, vr3=params.vr3,
                                mu1=params.mu1, mu2=params.mu2, mu3=params.mu3,
                                sigma1=params.sigma1, sigma2=params.sigma2, sigma3=params.sigma3,
                                extended_vp=params.extended_vp, v0_tan=params.v0_tan, epsilon_tan=params.epsilon_tan,
                                omega_tan=params.omega_tan, delta_tan=params.delta_tan,
                                conformity=params.conformity, M_min=M_min, M_max=M_max,
                                Ncen_mean=Ncen_mean, Nsat_mean=Nsat_mean, vr_bin_edges=vr_bin_edges,
                                vr_probs=vr_probs, vtan_bin_edges=vtan_bin_edges, vtan_probs=vtan_probs,
                                r_bin_edges=r_bin_edges, probs_r=probs_r, Nsat_r=Nsat_r,
                                K1_global=K1_global, K2_global=K2_global,
                                read_concentrations=params.read_concentrations
                                )
                t_process = timer.lap()
                # Write results to file
                io.write_galaxies_to_file(galaxies, f_out, True)
                t_write = timer.lap()  

                if verbose:
                    print(
                        f"[Chunk {line_count}] "
                        f"read={t_read:.3f}s | "
                        f"extract={t_extract:.3f}s | "
                        f"seed={t_seed:.3f}s | "
                        f"process={t_process:.3f}s | "
                        f"write={t_write:.3f}s | "
                        f"ngal={len(galaxies)}"
                        )

                line_count += len(chunk_data)
                galaxy_count += len(galaxies)

                # Progress report
                if line_count % c.report_after_nlines == 0:
                    elapsed = time.time() - start_time
                    rate = line_count / elapsed
                    if verbose:
                        print(f"Processed {line_count} halos, {galaxy_count} galaxies, rate: {rate:.1f} halos/sec")
        
        elapsed = time.time() - start_time
        rate = line_count / elapsed
        if verbose:
            print(f"Completed {line_count} halos, {galaxy_count} galaxies in {elapsed:.2f}s, rate: {rate:.1f} halos/sec")
        return line_count
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run_hod_model(params,verbose=False):

    """
    Run the Halo Occupation Distribution (HOD) model to generate a mock galaxy catalogue.

    This is the main driver function for building galaxy mocks from halo catalogues.
    It validates input parameters, loads analytical or empirical HOD shape/radial/velocity
    profiles as required, and processes the halo catalogue in chunks using the most
    efficient backend available. The resulting mock galaxy catalogue is written to file.

    Workflow
    --------
    1. Validate the input parameters.
    2. Depending on `params.analytical_shape`:
        - **Analytical shape**: Use analytical HOD functions for centrals and satellites.
        - **Empirical shape**: Load tabulated HOD occupation numbers (from HODfit2sim or 
          a text file). Conformity corrections (k1, k2) are also loaded if enabled.
    3. Depending on `params.analytical_rp`:
        - **Analytical radial profile**: Standard or extended NFW profile.
        - **Empirical radial profile**: Histogram-based sampling from input files.
    4. Depending on `params.analytical_vp`:
        - **Analytical velocity profile**: Virial scaling, Gaussian mixtures, or extended 
          laws depending on configuration.
        - **Empirical velocity profile**: Histogram-based sampling from input files.
    5. Process the halo catalogue in chunks (`process_halo_file_chunked`), assigning 
       galaxies to halos according to the chosen model(s).
    6. Print a run summary and performance statistics if `verbose=True`.

    Parameters
    ----------
    params : Namespace or dict
        Container of global HOD parameters, file paths, and control flags.
        Must include:
            - Input/output files: `infile`, `outfile`, `hod_shape_file`, `hod_rp_file`, `hod_vp_file`
            - Flags: `analytical_shape`, `analytical_rp`, `analytical_vp`, `extended_NFW`,
              `extended_vp`, `HODfit2sim`, `conformity`
            - HOD shape params: `hodshape`, `mu`, `Ac`, `As`, `alpha`, `sig`, `gamma`, `M0`, `M1`
            - Radial profile params: `alpha_r`, `beta_r`, `kappa_r`, `N0`, `r0`, `K`
            - Velocity profile params: `vfact`, `vt`, `vtdisp`, `vr1`, `vr2`, `vr3`,
              `mu1`, `mu2`, `mu3`, `sigma1`, `sigma2`, `sigma3`,
              `v0_tan`, `epsilon_tan`, `omega_tan`, `delta_tan`
            - Cosmology/simulation: `omega_M`, `zsnap`, `Lbox`, `seed`
    verbose : bool, optional
        If True, print detailed progress and performance information (default: False).

    Returns
    -------
    int
        Status code: 
        - 0 if processing completed successfully,
        - -1 if parameter validation or processing failed.

    Notes
    -----
    - The distinction **analytical vs empirical** applies only to the HOD *shape*.  
      Radial and velocity profiles can still be chosen independently as analytical 
      or empirical in the empirical mode. For **analytical** shapes, radial and velocity
      profiles are always analytical.
    - Processing is done in chunks for scalability on large halo catalogues.
    - A fixed global seed (`params.seed`) ensures reproducibility.
    - Conformity parameters (k1_global, k2_global) are supported when using empirical HOD shapes 
      with appropriate input files.

    See Also
    --------
    process_halo_file_chunked : Chunk-wise halo processing and galaxy generation.
    process_halos_chunk_analytical : Per-halo galaxy generation for analytical HOD shapes.
    process_halos_chunk_bins : Per-halo galaxy generation for empirical HOD shapes.
    """
    if params.conformity == True:
        if params.HODfit2sim:
            K1_global, K2_global = io.read_global_conformity_factors(params.hod_shape_file)
        else:
            K1_global, K2_global = np.loadtxt(params.conformity_file, unpack=True)
    else:
        K1_global, K2_global = 1.0, 1.0
    
    sep = io.line_separator()
    print(sep+"\nHOD Galaxy Mock Generation\n"+sep)

    # Validate parameters
    if not io.validate_parameters(params):
        print("Parameter validation failed. Please check your parameters.")
        io.print_parameter_info()
        return -1
    
    if params.derive_mu:
        logM_halos = amp.load_logM_from_halo_catalog(
            infile=params.infile,
            ftype=params.ftype,
            chunk_size=c.chunk_size,
        )

        bias_coeffs, bias_logM, bias_vals = amp.fit_bias_polynomial_from_file(
            bias_file=params.bias_file,
            degree=params.bias_poly_degree,
        )

        print("Fitted halo-bias polynomial for derive_mu:")
        print(f"  bias_file        = {params.bias_file}")
        print(f"  polynomial degree= {params.bias_poly_degree}")
        print(f"  coefficients     = {bias_coeffs}")
        print(io.line_separator())
        
        solved = amp.solve_mu_Ac_As_from_targets(
            logM_halos=logM_halos,
            Lbox=params.Lbox,
            hodshape=params.hodshape,
            ngal_target=params.ngal_target,
            fsat_target=params.fsat_target,
            bgal_target=params.bgal_target,
            bias_coeffs=bias_coeffs,
            mu_min=params.mu_min,
            mu_max=params.mu_max,
            mu_tol=params.mu_tol,
            mu_max_iter=params.mu_max_iter,
            conformity=params.conformity,
            K1_global=K1_global, 
            K2_global=K2_global,
        )

        params = params._replace(
            mu=solved["mu"],
            Ac=solved["Ac"],
            As=solved["As"],
            M0=solved["M0"],
            M1=solved["M1"],
            alpha=solved["alpha"],
            sig=solved["sig"],
            gamma=solved["gamma"],
        )

        

        print("Derived HOD parameters from ngal, fsat and bgal:")
        print(f"  mu = {params.mu:.6f}")
        print(f"  Ac = {params.Ac:.6e}")
        print(f"  As = {params.As:.6e}")
        print(f"  bgal(pred) = {solved['bgal']:.6f}")
        print(f"  bracket = {solved['mu_bracket']}")
        print(io.line_separator())

        ncen_pred, nsat_pred, ngal_pred, fsat_pred = amp.predict_number_densities_from_halo_masses(
            logM_halos=logM_halos,
            Lbox=params.Lbox,
            mu=params.mu,
            sig=params.sig,
            gamma=params.gamma,
            hodshape=params.hodshape,
            M0=params.M0,
            M1=params.M1,
            alpha=params.alpha,
            Ac=params.Ac,
            As=params.As,
            conformity=params.conformity,
            K1_global=K1_global,
            K2_global=K2_global,
        )

        print("Check of derived Ac and As:")
        print(f"  ngal_target = {params.ngal_target:.6e}")
        print(f"  ngal_pred   = {ngal_pred:.6e}")
        print(f"  fsat_target = {params.fsat_target:.6f}")
        print(f"  fsat_pred   = {fsat_pred:.6f}")
        print(f"  ncen_pred   = {ncen_pred:.6e}")
        print(f"  nsat_pred   = {nsat_pred:.6e}")
        print(io.line_separator())

    if params.derive_amplitudes and not params.derive_mu:
        logM_halos = amp.load_logM_from_halo_catalog(
            infile=params.infile,
            ftype=params.ftype,
            chunk_size=c.chunk_size,
        )

        Ac_new, As_new, Ic, Is = amp.compute_Ac_As_from_halo_masses(
            logM_halos=logM_halos,
            Lbox=params.Lbox,
            mu=params.mu,
            sig=params.sig,
            gamma=params.gamma,
            hodshape=params.hodshape,
            M0=params.M0,
            M1=params.M1,
            alpha=params.alpha,
            ngal_target=params.ngal_target,
            fsat_target=params.fsat_target,
            conformity=params.conformity,
            K1_global=K1_global,
            K2_global=K2_global,

        )

        params = params._replace(Ac=Ac_new, As=As_new)

        ncen_pred, nsat_pred, ngal_pred, fsat_pred = amp.predict_number_densities_from_halo_masses(
            logM_halos=logM_halos,
            Lbox=params.Lbox,
            mu=params.mu,
            sig=params.sig,
            gamma=params.gamma,
            hodshape=params.hodshape,
            M0=params.M0,
            M1=params.M1,
            alpha=params.alpha,
            Ac=params.Ac,
            As=params.As,
            conformity=params.conformity,
            K1_global=K1_global,
            K2_global=K2_global,

        )

        print("Derived analytical amplitudes from halo masses:")
        print(f"  ngal_target = {params.ngal_target:.6e}")
        print(f"  fsat_target = {params.fsat_target:.6f}")
        print(f"  Ic = {Ic:.6e}")
        print(f"  Is = {Is:.6e}")
        print(f"  Ac = {params.Ac:.6e}")
        print(f"  As = {params.As:.6e}")
        print("Predicted densities with derived amplitudes:")
        print(f"  ncen = {ncen_pred:.6e}")
        print(f"  nsat = {nsat_pred:.6e}")
        print(f"  ngal = {ngal_pred:.6e}")
        print(f"  fsat = {fsat_pred:.6f}")
        print(io.line_separator())
    
    if params.analytical_shape == False:
        if params.HODfit2sim == True:
            M_min, M_max, Ncen_mean, Nsat_mean = io.read_occupation_from_h5(params.hod_shape_file)
        else:
            M_min, M_max, Ncen_mean, Nsat_mean = np.loadtxt(params.hod_shape_file, unpack=True)

    if params.analytical_rp == False:
        if params.HODfit2sim == True:
            r_bin_edges, Nsat_r, probs_r = rp.load_radial_histogram_from_h5(params.hod_rp_file)
        else:
            r_min, r_max, Nsat_r, probs_r = np.loadtxt(params.hod_rp_file, unpack=True)
            r_bin_edges = np.concatenate([r_min, [r_max[-1]]])

    if params.analytical_vp == False:
        if params.HODfit2sim == True:
            vr_bin_edges, vr_probs, vtan_bin_edges, vtan_probs = vp.load_velocity_histograms_from_h5(params.hod_vp_file)
        else:
            vr_min, vr_max, Nsat_vr, vr_probs, vtan_min, vtan_max, Nsat_vtan, vtan_probs = np.loadtxt(params.hod_vp_file, unpack=True)
            vr_bin_edges = np.concatenate([vr_min, [vr_max[-1]]])
            vtan_bin_edges = np.concatenate([vtan_min, [vtan_max[-1]]])

    if params.analytical_shape == True:
        M_min = M_max = Ncen_mean = Nsat_mean = None

    if params.analytical_rp == True:
        r_bin_edges = probs_r = None
        Nsat_r = None

    if params.analytical_vp == True:
        vr_bin_edges = vr_probs = vtan_bin_edges = vtan_probs = None

    # Initialize random seed and time
    np.random.seed(params.seed)
    start_time = time.time()

    nhalos = process_halo_file_chunked(params,verbose=verbose, M_min=M_min, M_max=M_max,
                                       Ncen_mean=Ncen_mean, Nsat_mean=Nsat_mean,
                                       vr_bin_edges=vr_bin_edges, vr_probs=vr_probs,
                                       vtan_bin_edges=vtan_bin_edges, vtan_probs=vtan_probs,
                                       r_bin_edges=r_bin_edges, probs_r=probs_r, Nsat_r=Nsat_r,
                                       K1_global=K1_global, K2_global=K2_global)
    if nhalos is None:
        print(f"ERROR: Processing failed")
        return -1
    
    # End performance monitoring
    total_time = time.time() - start_time
    
    if verbose:
        io.print_run_summary(params)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {nhalos/total_time:.1f} halos/second")
    print(f"SUCCESS: Completed processing {nhalos} halos")

    mock = np.loadtxt(params.outfile)
    if mock.ndim == 1:
        mock = mock[None, :]

    Ngal = len(mock)
    Nsat_gal = np.sum(mock[:, 15] == 0)
    ngal_mock = Ngal / (params.Lbox ** 3)
    fsat_mock = Nsat_gal / Ngal if Ngal > 0 else 0.0

    print("DEBUG realized mock densities:")
    print(f"  Ngal      = {Ngal}")
    print(f"  Nsat_gal  = {Nsat_gal}")
    print(f"  ngal_mock = {ngal_mock:.6e}")
    print(f"  fsat_mock = {fsat_mock:.6f}")
    print(io.line_separator())
    
    return 0
