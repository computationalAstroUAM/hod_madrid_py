# hod_corrfunc.py

'''
Utilities for extracting galaxy positions and computing the two-point
correlation function with Corrfunc.

Overview
--------
This module provides a lightweight interface between HOD-generated mock galaxy
catalogues and the Corrfunc library. Its main purposes are:

1. Extract Cartesian galaxy positions from a full mock catalogue and write them
   to a three-column text file suitable for downstream clustering analysis.

2. Build redshift-space position catalogues by shifting one coordinate along a
   chosen line of sight according to the galaxy peculiar velocity.

3. Compute the real-space or redshift-space two-point correlation function
   ξ(r) in a periodic box using 'Corrfunc.theory.xi'.

The functions defined here are intended for diagnostic and validation stages of
the pipeline, allowing direct comparison between generated mocks and reference
clustering statistics.

Conventions
-----------
- Positions are assumed to be comoving and expressed in Mpc/h.
- Velocities are assumed to be in km/s.
- The simulation volume is assumed to be a periodic cube of side 'boxsize'.
- Input mock catalogues are assumed to be plain-text files with columns ordered
  consistently with the internal HOD output format.

Dependencies
------------
This module requires the 'Corrfunc' package to be installed and available in
the runtime environment.
'''

import numpy as np
from typing import Tuple, Literal
from Corrfunc.theory import xi         

def extract_positions_from_galaxy_catalog(
    input_catalog_file: str,
    output_positions_file: str,
    verbose: bool = True,
) -> str:
    """
    Extract the Cartesian galaxy positions from a mock catalogue.

    This function reads the first three columns of an HOD-generated galaxy
    catalogue, interprets them as the Cartesian coordinates '(x, y, z)', and
    writes them to a separate plain-text file. The resulting file is intended
    for downstream clustering calculations, such as direct use with Corrfunc.

    Parameters
    ----------
    input_catalog_file : str
        Path to the input mock galaxy catalogue. The file must contain at least
        three numeric columns corresponding to 'x', 'y', and 'z'.
    output_positions_file : str
        Path to the output file where the extracted three-column position
        catalogue will be written.
    verbose : bool, optional
        If True, print the output path once the extraction is complete.

    Returns
    -------
    str
        Path to the written positions file.

    Examples
    --------
    >>> extract_positions_from_galaxy_catalog('galaxies.txt', 'positions_xyz.txt')
    'positions_xyz.txt'
    """

    data = np.loadtxt(input_catalog_file, usecols=(0, 1, 2))
    np.savetxt(output_positions_file, data, fmt="%.6f")
    if verbose:
        print(f"Positions saved to: {output_positions_file}")
    return output_positions_file



def extract_positions_from_galaxy_catalog_rs(
    input_catalog_file: str,       
    output_positions_file: str,
    pos_key: Tuple[int, int, int] = (0, 1, 2),
    vel_key: Tuple[int, int, int] = (3, 4, 5),
    z_snap: float = 1.321,
    Omega_m: float = 0.3089,
    Omega_L: float = 0.6911,
    h: float = 0.6774,
    los_axis: Literal["x", "y", "z"] = "z",
    verbose: bool = True,
) -> str:
    
    """
    Construct a redshift-space position catalogue from a mock galaxy catalogue.

    The function reads galaxy positions and velocities from a plain-text mock
    catalogue and shifts the coordinate along a selected line-of-sight axis
    according to the mapping

        s_los = r_los + v_los / H(z),

    where 'H(z)' is evaluated for the supplied cosmological parameters.

    The output is a three-column text file containing the transformed Cartesian
    positions. The two coordinates transverse to the chosen line of sight are
    left unchanged.

    Parameters
    ----------
    input_catalog_file : str
        Path to the input mock galaxy catalogue.
    output_positions_file : str
        Path to the output three-column redshift-space position file.
    pos_key : tuple of int, optional
        Zero-based column indices of the position components '(x, y, z)' in the
        input catalogue.
    vel_key : tuple of int, optional
        Zero-based column indices of the velocity components '(vx, vy, vz)' in
        the input catalogue.
    z_snap : float, optional
        Snapshot redshift used to evaluate 'H(z)'.
    Omega_m : float, optional
        Present-day matter density parameter.
    Omega_L : float, optional
        Present-day dark-energy density parameter.
    h : float, optional
        Reduced Hubble constant, such that 'H0 = 100 h km/s/Mpc'.
    los_axis : {'x', 'y', 'z'}, optional
        Axis used as the line of sight.
    verbose : bool, optional
        If True, print the adopted 'H(z)' and the output path.

    Returns
    -------
    str
        Path to the written redshift-space positions file.

    Raises
    ------
    ValueError
        If 'pos_key' or 'vel_key' are not 3-tuples, or if 'los_axis' is not one
        of 'x', 'y', or 'z'.

    Notes
    -----
    The mapping implemented here uses the convention

        s_los = r_los + v_los / H(z).

    If a comoving convention based on 'v_los / [a H(z)]' is preferred, the
    relevant factor must be modified in the function body.
    """
    # Basic validation of column tuples
    if len(pos_key) != 3 or len(vel_key) != 3:
        raise ValueError("pos_key and vel_key must be 3-tuples of column indices.")
    if los_axis not in ("x", "y", "z"):
        raise ValueError("los_axis must be one of: 'x', 'y', 'z'.")

    # Read positions + velocities in one pass
    usecols = tuple(pos_key) + tuple(vel_key)
    data = np.loadtxt(input_catalog_file, usecols=usecols)

    # Split blocks (expects shape (N, 6))
    pos = data[:, :3]
    vel = data[:, 3:6]

    x, y, z  = pos[:, 0], pos[:, 1], pos[:, 2]
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]

    # H(z) in km/s/Mpc
    H0 = 100.0 * h
    Hz = H0 * np.sqrt(Omega_m * (1.0 + z_snap)**3 + Omega_L)
    if verbose:
        print(f"[RS] z={z_snap:.3f} -> H(z)={Hz:.3f} km/s/Mpc, LOS='{los_axis}'")

    # LOS displacement factor (default: 1 / H(z))
    # To use 1/(a*H), uncomment:
    # a = 1.0 / (1.0 + z_snap)
    # factor = 1.0 / (a * Hz)
    factor = 1 / Hz

    # Apply shift along the chosen LOS
    if los_axis == "x":
        s = x + vx * factor
        positions = np.column_stack([s, y, z])
    elif los_axis == "y":
        s = y + vy * factor
        positions = np.column_stack([x, s, z])
    else:  # 'z'
        s = z + vz * factor
        positions = np.column_stack([x, y, s])

    # Write out (three columns, scientific notation for safety)
    np.savetxt(output_positions_file, positions, fmt="%.6e", delimiter=" ")

    if verbose:
        print(f"Positions saved to: {output_positions_file} "
              f"({positions.shape[0]} rows x {positions.shape[1]} columns)")

    return output_positions_file

def compute_correlation_corrfunc(
    positions_file: str,
    output_file: str,
    boxsize: float,
    rmin: float,
    rmax: float,
    n_bins: int,
    n_threads: int = 4,
    binning: Literal["log", "linear"] = "log", 
    verbose: bool = True
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Compute the two-point correlation function ξ(r) with Corrfunc.

    This function reads a three-column position catalogue, constructs radial
    separation bins, evaluates the periodic-box two-point correlation function
    using 'Corrfunc.theory.xi', estimates simple Poisson-like uncertainties,
    and writes the result to disk.

    The output file contains three columns:

        r_center_Mpch, xi, err_poisson

    Parameters
    ----------
    positions_file : str
        Path to a plain-text file containing at least three numeric columns:
        'x', 'y', and 'z' positions in Mpc/h.
    output_file : str
        Path to the output CSV file.
    boxsize : float
        Periodic box side length in Mpc/h.
    rmin : float
        Minimum separation in Mpc/h. Must satisfy '0 < rmin < rmax'.
    rmax : float
        Maximum separation in Mpc/h.
    n_bins : int
        Number of separation bins.
    n_threads : int, optional
        Number of OpenMP threads used by Corrfunc.
    binning : {'log', 'linear'}, optional
        Radial bin spacing.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    tuple
        A tuple containing:
        - 'output_file' : str
            Path to the written CSV file.
        - 'r_centers' : ndarray
            Bin-center radii in Mpc/h.
        - 'xi_vals' : ndarray
            Correlation-function values in each bin.
        - 'errors' : ndarray
            Poisson-like uncertainty estimate per bin.

    Raises
    ------
    ValueError
        If the radial range is invalid, if the number of bins is not positive,
        if the binning mode is unsupported, or if the input file does not
        contain at least three columns.

    Notes
    -----
    The uncertainty estimate returned here is the simple approximation

        sigma_ξ ≈ (1 + ξ) / sqrt(N_pairs),

    where 'N_pairs' is the number of galaxy pairs in the bin.
    """
    if rmin <= 0 or rmax <= rmin:
        raise ValueError("Require 0 < rmin < rmax.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if binning not in ("log", "linear"):
        raise ValueError("binning must be 'log' or 'linear'.")

    if verbose:
        print(f"Loading positions from: {positions_file}")
    data = np.loadtxt(positions_file)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Input file must have at least 3 numeric columns (x, y, z).")
    x_data, y_data, z_data = data[:, 0], data[:, 1], data[:, 2]

    # Build bins
    if verbose:
        print(f"Generating {binning}-spaced bins...")
    if binning == "log":
        rbins = np.logspace(np.log10(rmin), np.log10(rmax), n_bins + 1)
    else:  # 'linear'
        rbins = np.linspace(rmin, rmax, n_bins + 1)

    r_centers = 0.5 * (rbins[:-1] + rbins[1:])

    if verbose:
        print("Computing ξ(r) with Corrfunc...")
    results = xi(
        boxsize=boxsize,
        nthreads=n_threads,
        binfile=rbins,
        X=x_data, Y=y_data, Z=z_data
    )

    xi_vals = np.array([b['xi'] for b in results], dtype=float)
    npairs  = np.array([b['npairs'] for b in results], dtype=float)

    errors = np.zeros_like(xi_vals)
    mask = npairs > 0
    errors[mask] = (1.0 + xi_vals[mask]) / np.sqrt(npairs[mask])

    output_data = np.column_stack((r_centers, xi_vals, errors))
    if verbose:
        print(f"Saving correlation to: {output_file}")
    header = "r_center_Mpch,xi,err_poisson"
    np.savetxt(output_file, output_data, delimiter=",", header=header, comments='')

    if verbose:
        print("Correlation computation complete.")
    return output_file, r_centers, xi_vals, errors