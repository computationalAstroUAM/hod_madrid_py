# hod_io.py

"""
I/O helpers and typed configuration structures for HOD simulations.

Overview
--------
This module defines:
  • A strongly-typed parameter container (`HODParams`) holding everything needed
    to run a Halo Occupation Distribution (HOD) realization and produce a mock
    galaxy catalogue.
  • Name-resolution maps (imported from `hod_config`) that help read halo
    catalogues in HDF5/TXT formats with heterogeneous column names.

Assumptions & Units
-------------------
- Cosmology: flat ΛCDM (Ω_M + Ω_Λ = 1) unless otherwise stated by the caller.
- Distances: comoving, in Mpc/h.
- Masses: in Msun/h.
- Velocities: km/s.
- Randomness: callers should pass a reproducible `seed` if determinism is required.

Typical usage
-------------
1) Gather file paths and physical/HOD parameters (from config or CLI).
2) Build a `HODParams` instance (directly or via a helper like `create_hod_params`).
3) Pass it to the HOD driver (e.g., `run_hod_model(params, ...)`).

"""

import math
import h5py
import os
from pathlib import Path
from typing import NamedTuple, Optional, Union, Literal, Dict, Sequence, Mapping, Generator, Tuple, TextIO
import numpy as np
import src.hod_const as c
from src.hod_config import (
    H5_NAME_MAP_WITH_R, H5_NAME_MAP_NO_R,
    TXT_NAME_MAP_WITH_R, TXT_NAME_MAP_NO_R,
    TXT_POS_WITH_R, TXT_POS_NO_R
)

class HODParams(NamedTuple):
    """
    Typed container for all inputs needed to run an HOD mock.

    The fields are grouped by theme for readability. See “Assumptions & Units”
    in the module docstring for global conventions.

    FILE PATHS & FORMATS
    --------------------
    infile : str
        Path to the input halo catalogue (HDF5 or TXT).
    outfile : str
        Path to the output mock galaxy catalogue to be written.
    ftype : str
        Input file type: "h5" or "txt".

    REPRODUCIBILITY & BOX
    ---------------------
    seed : int
        Random seed for reproducibility.
    zsnap : float
        Snapshot redshift (dimensionless).
    Lbox : float
        Periodic box size in comoving Mpc/h.
    omega_M : float
        Present-day matter density parameter Ω_M(0).

    HOD SHAPE (⟨N⟩ vs M)
    --------------------
    analytical_shape : bool
        If True, use analytical HOD shape; otherwise read it from file.
    HODfit2sim : bool
        If True, fit HOD to the statistical parameters from HODfit2sim code.
    hod_shape_file : str
        File with HOD shape (when 'analytical_shape=False').
    hodshape : int
        Analytical shape selector (e.g., 1, 2, or 3).
    mu : float
        log10 of the characteristic mass scale (dimensionless).
    Ac : float
        Central occupation amplitude (dimensionless).
    As : float
        Satellite occupation amplitude (dimensionless).
    M0 : float
        Central mass threshold (Msun/h).
    M1 : float
        Satellite mass scale (Msun/h).
    alpha : float
        Satellite power-law slope (dimensionless).
    sig : float
        Scatter in the central term (dimensionless, typically sigma_logM).
    gamma : float
        High-mass slope modifier for centrals (dimensionless).

    CENTRAL/SATELLITE PDF
    ---------------------
    analytical_pdf : bool
        Use an analytical PDF for satellite number if True; else read from file.
    beta : float
        PDF shape/dispersion control (e.g., 0 = Poisson, -2 = nearest-integer).
    hod_pdf_file : str
        Empirical PDF file (if 'analytical_pdf=False').

    CONFORMITY (OPTIONAL)
    ---------------------
    conformity : bool
        Enable 1-halo conformity (satellite occupation depends on central).
    conformity_file : str
        Path to text file with conformity parameters, if enabled.

    RADIAL PROFILE (SATELLITES)
    ---------------------------
    analytical_rp : bool
        Analytical radial profile if True; otherwise read from file.
    read_concentrations : bool
        Read per-halo concentration from input catalogue if True.
    K : float
        Radial normalization/truncation factor (e.g., 1 → up to R_vir).
    extended_NFW : bool
        If True, use an “extended/modified” NFW form.
    hod_rp_file : str
        File with radial profile data (if 'analytical_rp=False').
    N0 : float
        Radial normalization (dimensionless).
    r0 : float
        Characteristic radius (Mpc/h).
    alpha_r : float
        Inner slope parameter (dimensionless).
    beta_r : float
        Outer slope parameter (dimensionless).
    kappa_r : float
        Shape/curvature parameter (dimensionless).

    VELOCITY PROFILE (SATELLITES)
    -----------------------------
    analytical_vp : bool
        Analytical velocity profile if True; otherwise read from file.
    hod_vp_file : str
        File with velocity profile data (if 'analytical_vp=False').
    extended_vp : bool
        Enable extended/multi-component velocity models.
    vfact : float
        Global velocity normalization factor (alpha_v; dimensionless).
    vt : float
        Mean tangential velocity (km/s).
    vtdisp : float
        Tangential velocity dispersion (km/s).
    vr1, vr2, vr3 : float
        Parameters for the radial velocity PDF/components (units follow model).
    mu1, mu2, mu3 : float
        Means of velocity components in extended models (units follow model).
    sigma1, sigma2, sigma3 : float
        Dispersions of velocity components in extended models (units follow model).
    v0_tan : float
        Tangential velocity normalization (km/s).
    epsilon_tan : float
        Tangential anisotropy/deviation from isotropy (dimensionless).
    omega_tan : float
        Angular frequency for tangential modulation (1/time; model-dependent units).
    delta_tan : float
        Phase offset for tangential modulation (radians).
    """

    #------------
    infile: str
    outfile: str
    ftype: str
    #------------
    seed: int
    zsnap: float
    Lbox: float
    omega_M: float
    #------------
    analytical_shape: bool
    HODfit2sim: bool
    hod_shape_file: str
    hodshape: int
    mu: float
    Ac: float
    As: float
    M0: float
    M1: float
    alpha: float
    sig: float 
    gamma: float
    #------------
    analytical_pdf: bool
    beta: float
    hod_pdf_file: str
    #------------
    conformity: bool
    conformity_file: str
    #------------
    analytical_rp: bool
    read_concentrations: bool
    K: float
    extended_NFW: bool
    hod_rp_file: str
    N0: float
    r0: float
    alpha_r: float
    beta_r: float
    kappa_r: float
    #------------
    analytical_vp: bool
    hod_vp_file: str
    extended_vp: bool
    vfact: float
    vt: float
    vtdisp: float
    vr1: float
    vr2: float
    vr3: float
    mu1: float
    mu2: float
    mu3: float
    sigma1: float
    sigma2: float
    sigma3: float
    v0_tan: float
    epsilon_tan: float
    omega_tan: float
    delta_tan: float

def create_hod_params(
    # ---------------- FILE PATHS & FORMATS ----------------
    infile: Union[str, Path],
    outdir: Union[str, Path],
    ftype: Literal["txt", "h5"] = "txt",

    # ---------------- REPRODUCIBILITY & BOX ----------------
    seed: int = 50,
    zsnap: float = 0.0,
    Lbox: float = 100.0,                       # Mpc/h
    omega_M: float = 0.3,

    # ---------------- HOD SHAPE (⟨N⟩ vs M) ----------------
    analytical_shape: bool = True,
    HODfit2sim: bool = False,
    hod_shape_file: Optional[Union[str, Path]] = None,
    hodshape: int = 3,
    mu: float = 12.0,                        # log10 M*
    Ac: float = 1.0,                         # central amplitude
    As: float = 0.5,                         # satellite amplitude
    M0: float = 10**11.95,                   # Msun/h
    M1: float = 10**12.35,                   # Msun/h
    alpha: float = 0.9,
    sig: float = 0.08,
    gamma: Optional[float] = -1.4,

    # ---------------- CENTRAL/SATELLITE PDF ----------------
    analytical_pdf: bool = True,
    beta: float = 0.0,
    hod_pdf_file: Optional[Union[str, Path]] = None,

    # ---------------- CONFORMITY (OPTIONAL) ----------------
    conformity: bool = False,
    conformity_file: Optional[Union[str, Path]] = None,

    # ---------------- RADIAL PROFILE (SATELLITES) ----------------
    analytical_rp: bool = True,
    read_concentrations: bool = False,
    K: float = 1.0,                          # radial scaling/truncation
    extended_NFW: bool = True,
    hod_rp_file: Optional[Union[str, Path]] = None,
    N0: Optional[float] = None,
    r0: Optional[float] = None,              # Mpc/h
    alpha_r: Optional[float] = None,
    beta_r: Optional[float] = None,
    kappa_r: Optional[float] = None,

    # ---------------- VELOCITY PROFILE (SATELLITES) ----------------
    analytical_vp: bool = True,
    hod_vp_file: Optional[Union[str, Path]] = None,
    extended_vp: bool = True,
    vfact: float = 1.0,
    vt: float = 0.0,                         # km/s
    vtdisp: float = 0.0,                     # km/s
    vr1: Optional[float] = None,
    vr2: Optional[float] = None,
    vr3: Optional[float] = None,
    mu1: Optional[float] = None,
    mu2: Optional[float] = None,
    mu3: Optional[float] = None,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    sigma3: Optional[float] = None,
    v0_tan: Optional[float] = None,
    epsilon_tan: Optional[float] = None,
    omega_tan: Optional[float] = None,
    delta_tan: Optional[float] = None,
) -> Optional[HODParams]:
    
    """
    Build and validate a complete `HODParams` configuration object.

    This helper:
      1) Validates essential inputs (`infile`, `outdir`, `ftype`, `zsnap`, `Lbox`, `omega_M`).
      2) Ensures required auxiliary files are present when *analytical* flags are False.
      3) Creates a standardized output filename inside `outdir`.
      4) Returns a fully populated `HODParams` (or None on failure).

    Parameters
    ----------
    (Grouped and ordered exactly as in 'HODParams'; see function signature.)

    Returns
    -------
    HODParams or None
        A ready-to-use configuration container, or None if a blocking validation
        failed (a message is printed explaining the issue).

    Notes
    -----
    - Units: distances in Mpc/h, masses in Msun/h, velocities in km/s.
    - 'gamma': if None, falls back to 'src.hod_const.default_gamma'.
    - Output filename pattern (inside 'outdir'):
        galaxies_{Lbox}Mpc_NFW_mu{mu}_Ac{Ac}_As{As}_vfact{vfact}_beta{beta}_K{K}_vt{vt}pm{vtdisp}.dat
      Values are formatted for compact reproducibility tags.

    Examples
    --------
    >>> params = create_hod_params(
    ...     infile="data/halos.txt", outdir="output", ftype="txt",
    ...     seed=42, zsnap=1.0, Lbox=1000.0, omega_M=0.3089
    ... )
    >>> params.outfile
    'output/galaxies_1000Mpc_NFW_mu12.000_Ac1.0000_As0.50000_vfact1.00_beta0.000_K1.00_vt0pm0.dat'
    """
    # ---- defaults/normalization of simple inputs ----
    if gamma is None:
        gamma = c.default_gamma

    # Normalize paths early
    infile_path = Path(infile) if infile is not None else None
    outdir_path = Path(outdir) if outdir is not None else None

    # ---- basic validation of required fields ----
    if infile_path is None or not check_input_file(infile_path):
        print("STOP (hod_io): Please check your input file.")
        return None

    if outdir_path is None:
        print("STOP (hod_io): Output directory not provided.")
        return None

    # Ensure the output directory exists
    outdir_path.mkdir(parents=True, exist_ok=True)

    if ftype not in ("txt", "h5"):
        print("STOP (hod_io): ftype must be 'txt' or 'h5'.")
        return None

    # Required cosmology/box metadata
    if zsnap is None or Lbox is None or omega_M is None:
        print("STOP (hod_io): zsnap, Lbox and omega_M must be provided.")
        return None

    # ---- auxiliary file checks when not using analytical options ----
    if not analytical_shape and (hod_shape_file is None):
        print("STOP (hod_io): For a non-analytical HOD shape, provide 'hod_shape_file'.")
        return None

    if not analytical_rp and (hod_rp_file is None):
        print("STOP (hod_io): For a non-analytical radial profile, provide 'hod_rp_file'.")
        return None

    if not analytical_vp and (hod_vp_file is None):
        print("STOP (hod_io): For a non-analytical velocity profile, provide 'hod_vp_file'.")
        return None

    # ---- build a reproducible output filename (inside outdir) ----
    # Keep your original pattern; format floats compactly.
    outnom = (
        f"galaxies_{Lbox:.0f}Mpc_NFW_"
        f"mu{mu:.3f}_Ac{Ac:.4f}_As{As:.5f}_"
        f"vfact{vfact:.2f}_beta{beta:.3f}_K{K:.2f}_"
        f"vt{vt:.0f}pm{vtdisp:.0f}.dat"
    )
    outfile_path = outdir_path / outnom

    if not check_output_file(outfile_path):
        print("STOP (hod_io): Please check your output path.")
        return None

    # ---- assemble and return the typed configuration ----
    return HODParams(
        # FILE PATHS & FORMATS
        infile=str(infile_path),
        outfile=str(outfile_path),
        ftype=ftype,

        # REPRODUCIBILITY & BOX
        seed=seed,
        zsnap=zsnap,
        Lbox=Lbox,
        omega_M=omega_M,

        # HOD SHAPE (⟨N⟩ vs M)
        analytical_shape=analytical_shape,
        HODfit2sim=HODfit2sim,
        hod_shape_file=str(hod_shape_file) if hod_shape_file is not None else "",
        hodshape=hodshape if hodshape is not None else -1,
        mu=mu, Ac=Ac, As=As,
        M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma,

        # CENTRAL/SATELLITE PDF
        analytical_pdf=analytical_pdf,
        beta=beta,
        hod_pdf_file=str(hod_pdf_file) if hod_pdf_file is not None else "",

        # CONFORMITY (OPTIONAL)
        conformity=conformity,          
        conformity_file=str(conformity_file) if conformity_file is not None else "",

        # RADIAL PROFILE (SATELLITES)
        analytical_rp=analytical_rp,
        read_concentrations=read_concentrations,
        K=K,
        extended_NFW=extended_NFW,
        hod_rp_file=str(hod_rp_file) if hod_rp_file is not None else "",
        N0=N0 if N0 is not None else 0.0,
        r0=r0 if r0 is not None else 0.0,
        alpha_r=alpha_r if alpha_r is not None else 0.0,
        beta_r=beta_r if beta_r is not None else 0.0,
        kappa_r=kappa_r if kappa_r is not None else 0.0,

        # VELOCITY PROFILE (SATELLITES)
        analytical_vp=analytical_vp,
        hod_vp_file=str(hod_vp_file) if hod_vp_file is not None else "",
        extended_vp=extended_vp,
        vfact=vfact, vt=vt, vtdisp=vtdisp,
        vr1=vr1 if vr1 is not None else 0.0,
        vr2=vr2 if vr2 is not None else 0.0,
        vr3=vr3 if vr3 is not None else 0.0,
        mu1=mu1 if mu1 is not None else 0.0,
        mu2=mu2 if mu2 is not None else 0.0,
        mu3=mu3 if mu3 is not None else 0.0,
        sigma1=sigma1 if sigma1 is not None else 0.0,
        sigma2=sigma2 if sigma2 is not None else 0.0,
        sigma3=sigma3 if sigma3 is not None else 0.0,
        v0_tan=v0_tan if v0_tan is not None else 0.0,
        epsilon_tan=epsilon_tan if epsilon_tan is not None else 0.0,
        omega_tan=omega_tan if omega_tan is not None else 0.0,
        delta_tan=delta_tan if delta_tan is not None else 0.0,
    )

    

def create_output_directory(output_file: Union[str, Path]) -> bool:
    """
    Ensure the parent directory of `output_file` exists.

    Parameters
    ----------
    output_file : str | Path
        Target file path whose parent directory should exist.

    Returns
    -------
    bool
        True if the directory exists or was created successfully; False otherwise.
    """
    output_dir = Path(output_file).parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}")
        return False
    
def check_input_file(filename: Union[str, Path]) -> bool:
    """
    Check that an input file exists and is readable.

    Performs:
      - existence check
      - is a file (not a directory)
      - read access check (os.access)
      - attempts to open the file for reading to catch FS/lock issues

    Parameters
    ----------
    filename : str | Path
        Path to the input file.

    Returns
    -------
    bool
        True if the file passes all checks; False otherwise.
    """
    p = Path(filename)

    if not p.exists():
        print(f"ERROR: Input file does not exist: {p}")
        return False

    if not p.is_file():
        print(f"ERROR: Input path is not a file: {p}")
        return False

    if not os.access(p, os.R_OK):
        print(f"ERROR: Input file is not readable (permissions): {p}")
        return False

    # Final sanity check: try to open in text mode without loading content
    try:
        with p.open("rb"):
            pass
    except Exception as e:
        print(f"ERROR: Cannot open input file {p}: {e}")
        return False

    return True
    
def check_output_file(outfile: Union[str, Path]) -> bool:
    """
    Check that the parent directory of the output file is writable,
    creating it if necessary.

    Parameters
    ----------
    outfile : str | Path
        Target output file path.

    Returns
    -------
    bool
        True if we can write in the parent directory; False otherwise.
    """
    outpath = Path(outfile)
    # Ensure directory exists
    if not create_output_directory(outpath):
        return False

    # Check if file already exists and warn user
    if outpath.exists():
        print(f"WARNING: Output file already exists and will be overwritten: {outfile}")

    # Try creating a small temp file in the same directory to test writability
    try:
        test_path = outpath.with_suffix(outpath.suffix + ".tmp_write_test")
        with test_path.open("wb") as fh:
            fh.write(b"")
        test_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"ERROR: Output path is not writable ({outpath.parent}): {e}")
        return False
    
def validate_parameters(params) -> bool:
    """
    Validate HOD configuration values for basic physical/logic consistency.

    Returns
    -------
    bool
        True if all checks pass; False otherwise. Prints WARNINGS/ERRORS lists.
    """
    warnings = []
    errors = []

    # ---------- Basic numeric sanity ----------
    def _finite(name: str, value: float):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            errors.append(f"{name} = {value!r} must be a finite number")

    for name in ("mu","Ac","As","vfact","K","M0","M1","alpha","sig",
                 "Lbox","zsnap","omega_M","vt","vtdisp"):
        _finite(name, getattr(params, name))

    # ---------- Physical ranges ----------
    # Distances / masses / scales
    if params.Lbox <= 0:
        errors.append(f"Lbox = {params.Lbox:.6g} must be > 0 (Mpc/h)")
    if params.M0 <= 0:
        errors.append(f"M0 = {params.M0:.6g} must be > 0 (Msun/h)")
    if params.M1 <= 0:
        errors.append(f"M1 = {params.M1:.6g} must be > 0 (Msun/h)")
    if params.M1 < params.M0:
        errors.append(f"M1 = {params.M1:.6g} must be ≥ M0 = {params.M0:.6g}")

    # Redshift, HOD “shape”
    if params.zsnap < 0:
        errors.append(f"zsnap = {params.zsnap:.6g} must be ≥ 0")
    # mu is log10(M*). Typically 10 ≤ mu ≤ 14 for LSS; not an error, but warn if extreme
    if not (8.0 <= params.mu <= 16.0):
        warnings.append(f"mu = {params.mu:.3f} (log10 mass) is unusual; check units")

    # Occupation amplitudes/slopes
    if params.Ac < 0:
        errors.append(f"Ac = {params.Ac:.6g} must be ≥ 0")
    if params.As < 0:
        errors.append(f"As = {params.As:.6g} must be ≥ 0")
    if params.alpha <= 0:
        errors.append(f"alpha = {params.alpha:.6g} must be > 0")
    if params.sig < 0:
        errors.append(f"sig = {params.sig:.6g} must be ≥ 0")
    # gamma can be negative; no bound enforced

    # Cosmology
    if not (0.0 < params.omega_M < 1.0):
        errors.append(f"omega_M = {params.omega_M:.4f} must be in (0, 1)")

    # Radial profile
    if params.K <= 0:
        errors.append(f"K = {params.K:.6g} must be > 0 (radial scaling/truncation)")
    if params.read_concentrations not in (True, False):
        errors.append("read_concentrations must be a boolean")

    # Velocity profile
    if params.vfact <= 0:
        errors.append(f"vfact = {params.vfact:.6g} must be > 0")
    if params.vtdisp < 0:
        errors.append(f"vtdisp = {params.vtdisp:.6g} must be ≥ 0")
    # vt can be any real (mean shift), so no hard bound

    # Integers/enums
    if not isinstance(params.hodshape, int) or params.hodshape < 0:
        errors.append(f"hodshape = {params.hodshape!r} must be a non-negative integer")
    # If you only support certain shapes, turn this into a strict set check:
    # if params.hodshape not in (1, 2, 3): errors.append("hodshape must be 1, 2, or 3")

    # ---------- Gentle heuristics (warnings) ----------
    if params.As > 10*max(1e-12, params.Ac):
        warnings.append(f"As ({params.As:.3g}) >> Ac ({params.Ac:.3g}); "
                        "satellites may dominate at all masses.")
    if params.K < 0.2:
        warnings.append(f"K = {params.K:.3f} is very small; allows satellites beyond ~5 R_vir")

    # ---------- Print + return ----------
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        print()

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print()
        return False

    return True

def line_separator() -> str:
    """Return a standard 60-char separator for CLI/log output."""
    return "=" * 60

def resolve_names_from_header(
    header_names: Sequence[str],
    name_map: Mapping[str, Sequence[str]],
    *,
    case_sensitive: bool = False,
    raise_on_missing: bool = True,
) -> Dict[str, int]:
    """
    Resolve standard column keys to indices in a text header.

    Parameters
    ----------
    header_names : sequence of str
        Header tokens as read from the file (e.g., first line split on whitespace).
    name_map : dict[str, list[str]]
        Mapping from standard keys (e.g., "x","y","z") to a list of accepted
        synonyms in the header.
    case_sensitive : bool, optional
        If False (default), matching is case-insensitive.
    raise_on_missing : bool, optional
        If True (default), raise KeyError when any standard key is not found.
        If False, missing keys are omitted from the returned dict.

    Returns
    -------
    dict[str, int]
        Mapping standard key -> column index in `header_names`.

    Raises
    ------
    KeyError
        If a required key is missing and `raise_on_missing=True`.

    Notes
    -----
    - If multiple aliases for the same key appear in the header, the first match
      in `name_map[std]` order is used.
    - Matching is O(n) to build the index + O(1) per alias lookup.
    """
    # Normalize header (and build index for O(1) lookups)
    if case_sensitive:
        norm = list(h.strip() for h in header_names)
        lookup = {h: i for i, h in enumerate(norm)}
        def norm_token(s: str) -> str: return s
    else:
        norm = list(h.strip().lower() for h in header_names)
        lookup = {h: i for i, h in enumerate(norm)}
        def norm_token(s: str) -> str: return s.lower()

    out: Dict[str, int] = {}
    missing: list[str] = []

    for std, aliases in name_map.items():
        idx = None
        for alias in aliases:
            i = lookup.get(norm_token(alias))
            if i is not None:
                idx = i
                break
        if idx is None:
            missing.append(std)
        else:
            out[std] = idx

    if missing and raise_on_missing:
        raise KeyError(f"Missing columns in header: {', '.join(missing)}")
    return out

def try_mapping_hdf5(
    dtype_names: Sequence[str],
    name_map: Mapping[str, Sequence[str]],
    *,
    case_sensitive: bool = True,
    raise_on_missing: bool = True,
) -> Dict[str, str]:
    """
    Map standard keys to actual HDF5 dataset field names.

    Parameters
    ----------
    dtype_names : sequence of str
        Field names from the HDF5 dtype (e.g., dataset.dtype.names).
    name_map : dict[str, list[str]]
        Mapping from standard keys to accepted synonyms in the HDF5 dtype.
    case_sensitive : bool, optional
        If False, match case-insensitively (default True for HDF5 as names
        are often case-stable).
    raise_on_missing : bool, optional
        If True (default), raise KeyError if any standard key cannot be resolved.
        If False, missing keys are omitted.

    Returns
    -------
    dict[str, str]
        Mapping standard key -> actual field name present in the HDF5 dataset.

    Raises
    ------
    KeyError
        If a required key is missing and `raise_on_missing=True`.

    Notes
    -----
    - If multiple aliases exist for the same standard key, the first one found
      in `name_map[std]` is chosen.
    """
    if case_sensitive:
        available = {n: n for n in dtype_names}
        def norm_token(s: str) -> str: return s
    else:
        # keep original for return, but index by lower
        available = {n.lower(): n for n in dtype_names}
        def norm_token(s: str) -> str: return s.lower()

    out: Dict[str, str] = {}
    missing: list[str] = []

    for std, aliases in name_map.items():
        matched_name = None
        for a in aliases:
            key = norm_token(a)
            if key in available:
                matched_name = available[key]
                break
        if matched_name is None:
            missing.append(std)
        else:
            out[std] = matched_name

    if missing and raise_on_missing:
        raise KeyError(f"Missing fields in HDF5: {', '.join(missing)}")
    return out

def read_halo_data_chunked(
    filename: Union[str, Path],
    ftype: Literal["txt", "hdf5"] = "txt",
    chunk_size: int = c.chunk_size,
) -> Generator[np.ndarray, None, None]:
    """
    Stream halo data in fixed-size chunks to limit memory usage.

    Parameters
    ----------
    filename : str | Path
        Input catalog path.
    ftype : {'txt','hdf5'}, default 'txt'
        Input format selector. Dispatches to the appropriate reader.
    chunk_size : int, default c.chunk_size
        Maximum number of halos per yielded chunk.

    Yields
    ------
    np.ndarray, shape (n, 10)
        A block of rows with the canonical column order:
        [x, y, z, vx, vy, vz, logM, Rvir, Rs, id].
        The last chunk may have n < chunk_size.

    Notes
    -----
    - Actual parsing/mapping from headers/fields is handled in the backend
      readers: 'read_hdf5_chunked' and 'read_txt_chunked'.
    - This function only dispatches and forwards the yielded blocks.
    """
    ftype_norm = str(ftype).lower()
    if ftype_norm == "hdf5":
        yield from read_hdf5_chunked(str(filename), chunk_size)
    elif ftype_norm == "txt":
        yield from read_txt_chunked(str(filename), chunk_size)
    else:
        raise ValueError("ftype must be 'txt' or 'hdf5'")

    
def read_txt_chunked(
    filename: Union[str, Path],
    chunk_size: int = c.chunk_size,
) -> Generator[np.ndarray, None, None]:
    """
    Stream a TXT halo catalog in fixed-size chunks with canonical columns.

    The function reads a whitespace-delimited text file that may optionally
    include a header line (stored in comment lines starting with '#').
    It returns blocks (chunks) of rows as NumPy arrays with shape (n, 10)
    and the **canonical column order**:

        [x, y, z, vx, vy, vz, logM, Rvir, Rs, id]

    If the catalog does not contain Rvir/Rs, those columns are filled with NaN.
    Rvir and Rs are **converted from kpc/h to Mpc/h** (division by 1000).

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the input catalog (whitespace-delimited). Lines starting with
        '#' are treated as comments; the last such commented line is considered
        a header with column names (if present).
    chunk_size : int, default c.chunk_size
        Maximum number of rows per yielded chunk. The last chunk may have fewer.

    Yields
    ------
    np.ndarray, shape (n, 10), dtype float
        Chunk with canonical columns:
        x, y, z, vx, vy, vz, logM, Rvir[Mpc/h], Rs[Mpc/h], id

    Expected input schemas
    ----------------------
    - **With header** (in comment lines):
        The last comment line is parsed as header tokens. We attempt to resolve
        columns using:
          1) TXT_NAME_MAP_WITH_R  (expects Rvir and Rs)
          2) TXT_NAME_MAP_NO_R    (without radii)
        If neither mapping succeeds, a KeyError is raised upstream.
    - **Without header**:
        We infer by number of tokens in the first data row:
          - >= 10 → TXT_POS_WITH_R (fixed positions, includes radii)
          - >=  8 → TXT_POS_NO_R   (fixed positions, no radii)
          - <   8 → ValueError

    Notes
    -----
    - Separator: any whitespace (str.split()).
    - Comments: lines starting with '#'.
    - Units: Rvir and Rs are assumed to be in kpc/h if present; they are converted
      to Mpc/h. If your file already stores Mpc/h, remove the /1000 conversion.
    - The function performs minimal validation per-line: it skips lines with too
      few tokens (<8 or <10 depending on the schema).

    Examples
    --------
    >>> for block in read_txt_chunked("halos.txt", chunk_size=20000):
    ...     x, y, z = block[:,0], block[:,1], block[:,2]
    ...     logM     = block[:,6]
    ...     # process the chunk...

    """
    filename = str(filename)  # ensure str for open()
    try:
        with open(filename, "r") as f:
            # ---- Peek the first non-empty, non-comment line (to know column count) ----
            first_line = ""
            while True:
                line = f.readline()
                if not line:  # EOF
                    break
                s = line.strip()
                if s and not s.startswith("#"):
                    first_line = s
                    break
            if not first_line:
                # Empty or comment-only file → nothing to yield
                return

            first_tokens = first_line.split()

            # ---- Try to capture the last header line from leading comments ----
            f.seek(0)
            header_tokens = None
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    cand = s.lstrip("#").strip().split()
                    if cand:
                        header_tokens = cand  # keep LAST comment line as header
                    continue
                else:
                    # Reached first data line; stop scanning for header
                    break

            # ---- Decide schema: WITH_R or NO_R, and produce an index map ----
            has_r = None
            idx_map = None
            if header_tokens is not None:
                # Try name-based resolution WITH radii, else WITHOUT radii
                try:
                    idx_map = resolve_names_from_header(header_tokens, TXT_NAME_MAP_WITH_R)
                    has_r = True
                except KeyError:
                    idx_map = resolve_names_from_header(header_tokens, TXT_NAME_MAP_NO_R)
                    has_r = False
            else:
                # No header → infer by column count of first data line we saw
                ncols = len(first_tokens)
                if ncols >= 10:
                    has_r = True
                    idx_map = TXT_POS_WITH_R
                elif ncols >= 8:
                    has_r = False
                    idx_map = TXT_POS_NO_R
                else:
                    raise ValueError(f"TXT requires at least 8 columns; found {ncols}.")

            # ---- Stream rows again from top, build chunks in canonical order ----
            f.seek(0)
            chunk = []
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()

                need = 10 if has_r else 8
                if len(parts) < need:
                    # Skip malformed/short lines
                    continue

                # Build canonical row
                row = [0.0] * 10
                # x..logM
                row[0] = float(parts[idx_map["x"]])
                row[1] = float(parts[idx_map["y"]])
                row[2] = float(parts[idx_map["z"]])
                row[3] = float(parts[idx_map["vx"]])
                row[4] = float(parts[idx_map["vy"]])
                row[5] = float(parts[idx_map["vz"]])
                row[6] = float(parts[idx_map["logM"]])

                # Rvir, Rs (kpc/h → Mpc/h) or NaN if not present
                if has_r:
                    row[7] = float(parts[idx_map["Rvir"]]) / 1000.0
                    row[8] = float(parts[idx_map["Rs"]])   / 1000.0
                else:
                    row[7] = np.nan
                    row[8] = np.nan

                # id (robust to '123.0')
                row[9] = int(float(parts[idx_map["id"]]))

                chunk.append(row)
                if len(chunk) >= chunk_size:
                    yield np.asarray(chunk, dtype=float)
                    chunk = []

            # Yield the tail
            if chunk:
                yield np.asarray(chunk, dtype=float)

    except FileNotFoundError:
        print(f"ERROR: Could not open input file {filename}")
        return
    except Exception as e:
        print(f"ERROR: Error reading file {filename}: {e}")
        return

def read_hdf5_chunked(
    filename: Union[str, Path],
    chunk_size: int = 10000,
) -> Generator[np.ndarray, None, None]:
    """
    Stream an HDF5 halo catalog in fixed-size chunks with canonical columns.

    This reader yields NumPy arrays with shape (n, 10) in the canonical order:
        [x, y, z, vx, vy, vz, logM, Rvir, Rs, id]

    It supports two HDF5 layouts:
      1) **Structured dataset** (with named fields, i.e. `dtype.names` present).
         - Resolves field names via your synonym maps:
           `H5_NAME_MAP_WITH_R` (expects Rvir/Rs) or `H5_NAME_MAP_NO_R` (no radii).
         - Converts `Rvir`, `Rs` from **kpc/h → Mpc/h** (division by 1000).
      2) **Dense 2D array** (no field names, shape (N, C)).
         - Infers schema by column count: C ≥ 10 ⇒ with R, else C ≥ 8 ⇒ no R.
         - Assumes fixed column order in the first 8–10 columns (x..id).
         - Converts `Rvir`, `Rs` from **kpc/h → Mpc/h** (as above).

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the input HDF5 file.
    chunk_size : int, default 10000
        Maximum number of rows per yielded block.

    Yields
    ------
    np.ndarray, shape (n, 10), dtype float
        Chunk with the canonical column order:
        x, y, z, vx, vy, vz, logM, Rvir[Mpc/h], Rs[Mpc/h], id

    Notes
    -----
    - The function tries to auto-detect a main dataset under names:
        'halos', 'Halo', 'data', 'catalog'.
      If none matches, it lists the available paths and raises a ValueError.
    - If your file already stores `Rvir`, `Rs` in **Mpc/h**, remove the `/1000.0`
      conversions where indicated.
    - The `id` column is cast to float in the structured branch to fit the (n,10)
      float array. If you need integer IDs downstream, cast back with `.astype(int)`.

    Examples
    --------
    >>> for blk in read_hdf5_chunked("halos.h5", chunk_size=50000):
    ...     x, y, z = blk[:,0], blk[:,1], blk[:,2]
    ...     logM     = blk[:,6]
    ...     Rvir     = blk[:,7]   # Mpc/h (or NaN if not present)
    ...     # process each chunk...
    """
    try:
        import h5py
        with h5py.File(filename, "r") as f:
            # --- try to find a main dataset by common names ---
            possible_names = ["halos", "Halo", "data", "catalog"]
            dataset = None
            for name in possible_names:
                if name in f:
                    dataset = f[name]
                    break
            if dataset is None:
                print(f"Available objects in {filename}:")
                f.visit(print)
                raise ValueError(
                    "Could not find halo data. Please specify the correct dataset name."
                )

            # --- structured dataset (named fields) vs dense 2D matrix ---
            if hasattr(dataset, "dtype") and dataset.dtype.names:
                # Structured: map standard keys to actual field names
                names = dataset.dtype.names
                try:
                    colmap = try_mapping_hdf5(names, H5_NAME_MAP_WITH_R)
                    has_r = True
                except KeyError:
                    colmap = try_mapping_hdf5(names, H5_NAME_MAP_NO_R)
                    has_r = False

                n = len(dataset)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    view = dataset[start:end]

                    # Allocate full (n,10)
                    out = np.empty((end - start, 10), dtype=float)

                    # x..logM
                    out[:, 0] = view[colmap["x"]]
                    out[:, 1] = view[colmap["y"]]
                    out[:, 2] = view[colmap["z"]]
                    out[:, 3] = view[colmap["vx"]]
                    out[:, 4] = view[colmap["vy"]]
                    out[:, 5] = view[colmap["vz"]]
                    out[:, 6] = view[colmap["logM"]]

                    # Rvir, Rs (kpc/h → Mpc/h) or NaN
                    if has_r:
                        out[:, 7] = np.asarray(view[colmap["Rvir"]], dtype=float) / 1000.0
                        out[:, 8] = np.asarray(view[colmap["Rs"]],   dtype=float) / 1000.0
                    else:
                        out[:, 7] = np.nan
                        out[:, 8] = np.nan

                    # id (cast to float to fit array dtype)
                    out[:, 9] = np.asarray(view[colmap["id"]], dtype=float)

                    yield out

            else:
                # Dense 2D array (no field names)
                shape = dataset.shape
                if len(shape) != 2:
                    raise ValueError(f"Expected 2D array, got shape {shape}")
                nrows, ncols = shape
                if ncols < 8:
                    raise ValueError(f"Expected >=8 columns, got {ncols}")

                has_r = (ncols >= 10)

                for start in range(0, nrows, chunk_size):
                    end = min(start + chunk_size, nrows)
                    arr = dataset[start:end, :]

                    out = np.empty((end - start, 10), dtype=float)
                    # x..logM (assumed in the first 7 columns)
                    out[:, 0:7] = arr[:, 0:7]

                    if has_r:
                        # Rvir, Rs assumed in columns 7–8 (kpc/h → Mpc/h)
                        out[:, 7] = np.asarray(arr[:, 7], dtype=float) / 1000.0
                        out[:, 8] = np.asarray(arr[:, 8], dtype=float) / 1000.0
                        # id in column 9
                        out[:, 9] = np.asarray(arr[:, 9], dtype=float)
                    else:
                        out[:, 7] = np.nan
                        out[:, 8] = np.nan
                        # id in column 7 when there are no radii
                        out[:, 9] = np.asarray(arr[:, 7], dtype=float)

                    yield out

    except ImportError:
        print("ERROR: h5py is required to read HDF5 files. Install with: pip install h5py")
        return
    except Exception as e:
        print(f"ERROR: Error reading HDF5 file {filename}: {e}")
        return

def print_parameter_info() -> None:
    """
    Print a concise, structured help message describing configurable parameters
    for the HOD mock generation pipeline.

    This helper prints:
      - How to run the script
      - Expected input/output formats and units
      - A practical guide to key HOD/velocity/cosmology parameters,
        with typical ranges and qualitative effects

    Returns
    -------
    None
    """
    print(
        """
HOD Mock Generation — Parameter Reference
=========================================
Usage
-----
    python produce_hod_mock.py

Configuration
-------------
- Edit physics/model parameters in the “PARAMETER DEFINITION SECTION”.
- Adjust performance thresholds in “PERFORMANCE CONFIGURATION”.
- Modify file paths in “FILE PATH CONFIGURATION”.

Input format (TXT)
------------------
- Whitespace-separated columns:
    x  y  z  vx  vy  vz  logM  Rvir  Rs  halo_id
- Comment lines start with '#'.
- Units:
    • Positions  : Mpc/h (comoving)
    • Velocities : km/s
    • Masses     : Msun/h
    • Rvir, Rs   : kpc/h

Output format (mock catalogue)
------------------------------
Columns (whitespace-separated):
    x  y  z  vx  vy  vz  M  Nsat  Dvx  Dvy  Dvz  Dx  Dy  Dz  halo_id  is_central
Notes:
    • is_central ∈ {0, 1} (1 = central, 0 = satellite).
    • Displacement/velocity offsets (Dx, Dy, Dz, Dvx, Dvy, Dvz) follow the
      conventions of your pipeline modules (documented where they are created).

HOD Parameters (occupation)
---------------------------
- mu   : log10 of characteristic halo mass (dimensionless).
         Typical range depends on the target sample (e.g. ~10-13).
         Affects both central and satellite occupations.

- Ac   : Central amplitude (0-1; often ~1).
         Ac=1 ⇒ all halos above threshold can host a central.

- As   : Satellite amplitude (≈ 0.1-2.0).
         Larger As ⇒ more satellites at fixed halo mass.

- alpha: Satellite power-law slope (≈ 0.8-1.2).
         Larger alpha ⇒ steeper increase of N_sat with halo mass.

- sig  : Scatter (log-space) in the central term (≈ 0.05-0.2).
         Larger sig ⇒ smoother transition in central occupation.

- gamma: High-mass slope modifier for centrals (often negative, e.g. -2 to -1).

- beta : Satellite count PDF control.
         beta = 0   → Poisson (standard)
         beta > 0   → super-Poisson (extra variance; e.g., negative binomial)
         beta < 0   → sub-Poisson (more regular than Poisson; e.g., nearest-integer)

Radial Profile (satellites)
---------------------------
- K      : NFW truncation/normalization factor (≈ 1).
           K = 1   ⇒ truncate near R_vir
           K < 1   ⇒ allows orbits extending beyond R_vir (larger effective max radius)
           K > 1   ⇒ tighter truncation inside R_vir

- (If used) Rvir, Rs are expected in Mpc/h in your pipeline after I/O conversion.

Velocity Parameters (satellites)
--------------------------------
- vfact : Velocity scale factor (≈ 0.5-1.5).
          vfact = 1.0 ⇒ “virial-like” internal velocities.
          Scales the dispersion (and any velocity model you use).

- vt    : Mean tangential streaming speed (km/s).
          Controls a net tangential component (often 0 or a few hundred km/s).

- vtdisp: Tangential velocity dispersion (km/s).
          Extra random tangential scatter (often 0-200 km/s).

Cosmology & Box
---------------
- zsnap   : Snapshot redshift (≥ 0).
            Affects Δ_vir(z), c(M,z), and H(z) in RS mappings.

- omega_M : Present-day matter density parameter (~0.3 in ΛCDM).

- Lbox    : Periodic simulation box size in Mpc/h.

Tips
----
- Keep units consistent: positions [Mpc/h], velocities [km/s], masses [Msun/h],
  radii [Mpc/h].
- For large files, prefer chunked readers and set a suitable chunk size to match
  your I/O bandwidth and RAM.
"""
    )
    print(line_separator())
    return


def print_run_summary(params) -> None:
    """
    Print a concise, human-readable summary of the current HOD run configuration.

    Parameters
    ----------
    params : HODParams-like
        Object with attributes used by the HOD pipeline (hodshape, mu, Ac, ...).
        Only read access is required; `NamedTuple` or any object with these
        attributes works.

    Notes
    -----
    - Units: Lbox [Mpc/h], masses [Msun/h], velocities [km/s].
    - Prints a separator line at the end for CLI readability.
    """
    print("HOD Parameters:")
    print(f"  hodshape = {params.hodshape}")
    print(f"  mu       = {params.mu:.3f}")
    print(f"  Ac       = {params.Ac:.4f}")
    print(f"  As       = {params.As:.5f}")
    print(f"  vfact    = {params.vfact:.2f}")
    print(f"  beta     = {params.beta:.3f}")
    print(f"  K        = {params.K:.2f}")
    print(f"  vt       = {params.vt:.0f}")
    print(f"  vtdisp   = {params.vtdisp:.0f}")
    print(f"  alpha    = {params.alpha:.1f}")
    print(f"  sig      = {params.sig:.2f}")
    print(f"  gamma    = {params.gamma:.1f}")

    print("Derived parameters:")
    print(f"  M0 = {params.M0:.6e}")
    print(f"  M1 = {params.M1:.6e}")

    print("Cosmology:")
    print(f"  zsnap   = {params.zsnap:.4f}")
    print(f"  omega_M = {params.omega_M:.4f}")
    print(f"  Lbox    = {params.Lbox:.1f} Mpc/h")

    print("Files:")
    print(f"  Input  : {params.infile}")
    print(f"  Output : {params.outfile}")
    print(line_separator())
    return


def print_file_info(input_file: Union[str, Path], nhalos: int) -> None:
    """
    Print basic information about the input halo file and the number of halos parsed.

    Parameters
    ----------
    input_file : str | Path
        Path to the input catalog file (TXT or HDF5).
    nhalos : int
        Number of halos detected/parsed (e.g., total rows).

    Notes
    -----
    - Attempts to fetch file size and prints a human-friendly value.
    - Silently skips size reporting if the path is not accessible.
    """
    p = Path(input_file)
    print(f"Input file: {p}")
    print(f"Found {nhalos} halos in the catalogue")

    # Try to determine file size (best-effort)
    try:
        size = p.stat().st_size
        if size >= 1024**3:
            size_str = f"{size / 1024**3:.1f} GB"
        elif size >= 1024**2:
            size_str = f"{size / 1024**2:.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"File size: {size_str}")
    except (FileNotFoundError, PermissionError, OSError):
        # Don't fail if we cannot read metadata
        pass


def write_galaxies_to_file(
    galaxies: np.ndarray,
    f_out: TextIO,
    more_info: bool = True,
) -> None:
    """
    Write galaxy data rows to an open text file handle.

    Parameters
    ----------
    galaxies : ndarray, shape (n_galaxies, 16)
        Array in the canonical order:
        [x, y, z, vx, vy, vz, M, Nsat, Dvx, Dvy, Dvz, Dx, Dy, Dz, halo_id, is_central].
        (Indices 0..15; is_central is 0/1.)
    f_out : file-like
        Open file handle in text mode (e.g., with open(..., 'w')).
    more_info : bool, default False
        If True, writes the full “MORE” format (16 columns).
        If False, writes only the essential subset: x y z vx vy vz M Nsat.

    Notes
    -----
    - The function writes line-by-line using Python's f-string formatting.
    - For very large outputs, consider vectorized `np.savetxt` if format allows.
    """
    nrows = len(galaxies)
    for i in range(nrows):
        g = galaxies[i, :]
        if more_info:
            # x y z vx vy vz M Nsat Dvx Dvy Dvz Dx Dy Dz halo_id is_central
            f_out.write(
                f"{g[0]:.5f} {g[1]:.5f} {g[2]:.5f} "
                f"{g[3]:.5f} {g[4]:.5f} {g[5]:.5f} "
                f"{g[6]:.6e} {int(g[7])} "
                f"{g[8]:.4f} {g[9]:.4f} {g[10]:.4f} "
                f"{g[11]:.4f} {g[12]:.4f} {g[13]:.4f} "
                f"{int(g[14])} {int(g[15])}\n"
            )
        else:
            # x y z vx vy vz M Nsat
            f_out.write(
                f"{g[0]:.5f} {g[1]:.5f} {g[2]:.5f} "
                f"{g[3]:.5f} {g[4]:.5f} {g[5]:.5f} "
                f"{g[6]:.6e} {int(g[7])}\n"
            )


def read_occupation_from_h5(h5file: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read mass-bin edges and mean occupations from an HDF5 produced by HODfit2sim code.

    Parameters
    ----------
    h5file : str | Path
        Path to `h2s_output.h5` (expects a group 'data' with datasets:
        'M_min', 'M_max', 'Ncen', 'Nsat', 'N_halo').

    Returns
    -------
    M_min : ndarray
        Lower edges of mass bins.
    M_max : ndarray
        Upper edges of mass bins.
    Ncen_mean : ndarray
        Mean centrals per halo per bin (= Ncen / N_halo, 0 where N_halo == 0).
    Nsat_mean : ndarray
        Mean satellites per halo per bin (= Nsat / N_halo, 0 where N_halo == 0).

    Notes
    -----
    - Safe division is used to avoid NaNs/inf when N_halo == 0.
    """
    with h5py.File(h5file, "r") as f:
        data = f["data"]
        M_min = np.asarray(data["M_min"])
        M_max = np.asarray(data["M_max"])
        Ncen  = np.asarray(data["Ncen"], dtype=float)
        Nsat  = np.asarray(data["Nsat"], dtype=float)
        N_h   = np.asarray(data["N_halo"], dtype=float)

    # Safe means: 0 where N_halo == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        Ncen_mean = np.divide(Ncen, N_h, out=np.zeros_like(Ncen, dtype=float), where=(N_h > 0))
        Nsat_mean = np.divide(Nsat, N_h, out=np.zeros_like(Nsat, dtype=float), where=(N_h > 0))

    return M_min, M_max, Ncen_mean, Nsat_mean


def read_global_conformity_factors(h5file: str | Path) -> Tuple[float, float]:
    """
    Read global conformity factors (K1_global, K2_global) from the HDF5 header.

    Parameters
    ----------
    h5file : str | Path
        Path to the HDF5 file. Expects a group 'header' with attributes
        'K1_global' and 'K2_global'.

    Returns
    -------
    (K1_global, K2_global) : tuple of floats

    Raises
    ------
    KeyError
        If the expected attributes are missing from the file header.
    """
    with h5py.File(h5file, "r") as f:
        header = f["header"]
        try:
            K1_global = float(header.attrs["K1_global"])
            K2_global = float(header.attrs["K2_global"])
        except KeyError as e:
            raise KeyError(f"Missing conformity attribute in header: {e}") from e
    return K1_global, K2_global