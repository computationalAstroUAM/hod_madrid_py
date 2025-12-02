#hod_corrfunc.py

import numpy as np
from typing import Tuple, Literal
from Corrfunc.theory import xi         # requires Corrfunc installed

def extract_positions_from_galaxy_catalog(
    input_catalog_file: str,
    output_positions_file: str,
    verbose: bool = True,
) -> str:
    
    """
    Extract the 3D positions (x, y, z) from a galaxy catalog and save them to a new file.

    This function reads the first three columns (assumed to be x, y, z coordinates)
    from a text-based galaxy catalog produced with an HOD pipeline, and writes them
    to a new plain-text file for downstream correlation-function tools.

    Parameters
    ----------
    input_catalog_file : str
        Path to the input text file (e.g., galaxy HOD-based catalog). The file must be
        space- or whitespace-delimited and have at least three numeric columns in
        positions 0, 1, and 2 corresponding to x, y, z.
    output_positions_file : str
        Path to the output text file that will contain only the x, y, z columns.

    Returns
    -------
    str
        The path to the written positions file (`output_positions_file`).

    Notes
    -----
    - The function assumes the first three columns of the input are x, y, z in
      consistent units (e.g., Mpc/h). It doesn't attempt unit conversion or validation.
    - Values are written with six decimal places using `fmt="%.6f"`.

    Examples
    --------
    >>> extract_positions_from_galaxy_catalog("galaxies.txt", "positions_xyz.txt")
    Positions saved to: positions_xyz.txt
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
    Project real-space galaxy positions to redshift space along a chosen LOS axis,
    and save the result as a 3-column TXT. Default case: (x, y, s_LOS).

    The function reads positions and velocities from a plain-text galaxy catalog,
    shifts the coordinate along the line-of-sight (LOS) by v_LOS / H(z), and writes out
    the modified coordinates. It assumes the file is whitespace-delimited.

    Redshift-space mapping used
    ---------------------------
    s_LOS = r_LOS + v_LOS / H(z)

    If you prefer the “comoving Kaiser” convention (s_LOS = r_LOS + v_LOS / (a·H)),
    replace the factor `1/Hz` by `1/(a*Hz)` with `a = 1/(1+z)` (see note below).

    Parameters
    ----------
    input_catalog_file : str
        Path to the input TXT catalog. Must contain at least 6 numeric columns.
        The columns for positions and velocities are selected by `pos_key` and `vel_key`.
    output_positions_file : str
        Path to the output TXT file (three columns: x, y, s_LOS).
    pos_key : (int, int, int), optional
        Zero-based indices of the (x, y, z) columns in the input file. Default: (0, 1, 2).
    vel_key : (int, int, int), optional
        Zero-based indices of the (vx, vy, vz) columns in the input file. Default: (3, 4, 5).
    z_snap : float, optional
        Snapshot redshift used to compute H(z). Default: 1.321.
    Omega_m : float, optional
        Matter density parameter at z=0. Default: 0.3089.
    Omega_L : float, optional
        Cosmological constant density parameter at z=0. Default: 0.6911.
    h : float, optional
        Dimensionless Hubble parameter (H0 = 100*h km/s/Mpc). Default: 0.6774.
    los_axis : {'x','y','z'}, optional
        Axis to use as the line-of-sight. The coordinate on this axis is shifted by v_LOS / H(z).
        Default: 'z'.
    verbose : bool, optional
        If True, prints a short summary (H(z) and output path).

    Returns
    -------
    str
        The path to the written file (`output_positions_file`).

    Units and conventions
    -------------------
    - Positions are assumed to be in Mpc/h (comoving) and velocities in km/s.
    - H(z) is computed as: H0 * sqrt(Ω_m (1+z)^3 + Ω_Λ), in km/s/Mpc.
    - If you need s_LOS = r_LOS + v_LOS/(a·H), uncomment the two lines in the code:
        a = 1/(1+z_snap)
        factor = 1.0 / (a * Hz)
      and remove the line `factor = 1.0 / Hz`.

    Examples
    --------
    >>> extract_positions_from_galaxy_catalog_rs(
    ...     "galaxies.txt", "positions_rs.txt", los_axis="z"
    ... )
    [RS] z=1.321 -> H(z)=XXX.XXX km/s/Mpc, LOS='z'
    Positions saved to: positions_rs.txt (N rows x 3 columns)
    'positions_rs.txt'
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
    Compute the real-space two-point correlation function ξ(r) with Corrfunc and
    save results to disk. Per-bin errors are the simple Poisson approximation.

    Steps:
      1) Load 3-column positions (x,y,z) from a text file.
      2) Build radial bins between rmin and rmax:
         - binning='log'   -> logarithmically spaced (default)
         - binning='linear'-> linearly spaced
      3) Call Corrfunc.theory.xi.xi (periodic box of size `boxsize`).
      4) Errors: sigma_ξ ≈ (1 + ξ) / sqrt(Npairs).
      5) Write CSV: r_center, xi, err_poisson.

    Parameters
    ----------
    positions_file : str
        Path to a plain-text file with three numeric columns: x, y, z (Mpc/h).
    output_file : str
        Path where the CSV output will be written.
    boxsize : float
        Periodic box side length (Mpc/h).
    rmin : float
        Minimum separation (Mpc/h), must be > 0 and < rmax.
    rmax : float
        Maximum separation (Mpc/h), must be > rmin.
    n_bins : int
        Number of bins (>=1).
    n_threads : int, optional
        Number of OpenMP threads for Corrfunc (default: 4).
    binning : {'log','linear'}, optional
        Bin spacing type. Default: 'log'.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    output_file : str
        Path to the written CSV.
    r_centers : ndarray, shape (n_bins,)
        Bin-center radii (Mpc/h), arithmetic mean of edges.
    xi_vals : ndarray, shape (n_bins,)
        Corrfunc xi values per bin.
    errors : ndarray, shape (n_bins,)
        Poisson-like errors per bin: (1 + xi) / sqrt(Npairs), zero where Npairs=0.

    Notes
    -----
    - Positions are treated as comoving and the box is periodic with side `boxsize`.
    - Errors are computed with simple Poisson statistics: sigma_ξ ≈ (1 + ξ) / √Npairs.
    - Output is CSV with a one-line header and no comment marker.
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