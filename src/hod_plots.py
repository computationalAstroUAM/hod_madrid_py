import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec
import re
from glob import glob
from typing import Union, List, Optional, Tuple, Dict
import warnings
from Corrfunc.theory.xi import xi

import src.hod_io as io

def get_distinct_colors(n_colors):
    """Generate distinct colors for plotting"""
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, min(n_colors, 10)))
    if n_colors > 10:
        colors2 = cm.Set3(np.linspace(0, 1, n_colors - 10))
        colors = np.vstack([colors, colors2])
    return colors


def plot_satellite_radial_distribution(
    mock_file: str,
    output_dir: str = "output/plots",
    rmin: float = -3.0,
    rmax: float = 2.0,
    dr: float = 0.05,
    xmin: float = -2.0,
    xmax: float = 0.7,
    ymin: float = -5.0,
    ymax: float = -0.4,
    testing: bool = False,
    verbose: bool = False,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict:
    """
    Plot satellite radial distribution as function of distance from halo center.
    
    This function processes mock galaxy catalogs and plots the normalized number
    of satellite galaxies as a function of their radial distance from halo centers.
    
    Parameters:
    -----------
    mock_file : str  
        Path to mock file. If contains '_subvol' followed by a number,
        will automatically find and process all subvolumes. Expected format:
        mock000_subvolN.dat or similar
    output_dir : str, default="."
        Directory to save output plots
    rmin, rmax : float, default=-3.0, 2.0
        Log10 range for radial distance bins (Mpc/h)
    dr : float, default=0.05
        Bin width for radial distance
    xmin, xmax, ymin, ymax : float
        Plot axis limits
    testing : bool, default=False
        If True, only processes a subset of boxes for testing
    plot_title : str, optional
        Custom title for the plot
    figsize : tuple, default=(10, 8)
        Figure size for the plot
        
    Returns: path to plot
    --------
    """   
    # Setup radial bins
    rbins = np.arange(rmin, rmax, dr)
    rhist = rbins + dr * 0.5

    # Check if mock_file contains subvolume pattern and find all files
    subvol_pattern = r'_subvol(\d+)'
    subvols = re.search(subvol_pattern, mock_file)
    base_name = os.path.basename(mock_file)

    mock_files = []
    if subvols is not None:
        # Extract base pattern and find all subvolume files
        base_pattern = re.sub(subvol_pattern, '_subvol*', mock_file)
        # Replace the specific number with a wildcard pattern
        base_dir = os.path.dirname(mock_file)
        pattern = re.sub(subvol_pattern, '_subvol[0-9]*', base_name)
        
        if base_dir:
            search_pattern = os.path.join(base_dir, pattern)
        else:
            search_pattern = pattern
            
        # Find all matching files
        potential_files = []
        if base_dir:
            for f in os.listdir(base_dir):
                if re.search(re.sub(r'\[0-9\]\*', r'\\d+', pattern), f):
                    potential_files.append(os.path.join(base_dir, f))
        else:
            for f in os.listdir('.'):
                if re.search(re.sub(r'\[0-9\]\*', r'\\d+', pattern), f):
                    potential_files.append(f)
        
        mock_files = sorted(potential_files)
        if not mock_files or testing:
            # Fallback to single file
            mock_files = [mock_file]
            warnings.warn(f"No subvolume files found, using single file: {mock_file}")
    else:
        mock_files = [mock_file]

    if verbose:
        print(f"Radial profile plot with {len(mock_files)} file(s)")

    # Initialize figure
    fig = plt.figure(figsize=figsize)
    xtit = "${\\rm log}_{10}(r\\,h^{-1}{\\rm Mpc})$"
    ytit = "${\\rm log}_{10}(N_{\\rm sat,dex}/N_{\\rm sat, tot})$"
    
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xtit)
    ax.set_ylabel(ytit)
    
    if plot_title:
        ax.set_title(plot_title)
    
    # Get colors for plotting
    colors = get_distinct_colors(len(mock_files))
        
    # Track global min/max for logging
    global_minr = 999.0
    global_maxr = -999.0
    
    # Process each mock file
    for im, mock in enumerate(mock_files):
        if verbose:
            print(f"Processing file: {mock}")
        io.check_input_file(mock)

        # Extract K value if present in filename
        kval = "unknown"
        if 'K' in mock:
            try:
                kval1 = mock.split('K')[1]
                kval = kval1.split('_')[0]
            except:
                pass

        avgr = 0.0
        ntot = 0
        nsatr = np.zeros(len(rbins))

        # Read mock catalogue
        ldx, ldy, ldz = [], [], []
        try:
            # Load only the columns we need: dx, dy, dz, sat, is_central
            data = np.loadtxt(mock, usecols=[11, 12, 13, 7, 15])
            ngalsat = data[:,-1][data[:, -1] == 0]

            # Filter for satellite galaxies (cen != 1)
            sat_mask = data[:, 3] > 0
            sat_data = data[sat_mask]

            # Extract coordinates
            ldx.extend(sat_data[:, 0])
            ldy.extend(sat_data[:, 1]) 
            ldz.extend(sat_data[:, 2])
            is_sat = sat_data[:, 4] == 0

        except Exception as e:
            print(f"    Error reading {mock}: {e}")
            continue
        
        if not ldx:  # No satellites found
            continue
                        
        dx = np.array(ldx, dtype=float)
        dy = np.array(ldy, dtype=float)
        dz = np.array(ldz, dtype=float)
                    
        # Calculate radial distance for satellite galaxies (in Mpc/h)
        rsat = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Average and total number
        avgr += np.sum(rsat)
        ntot += len(rsat)

        # Track global limits
        ind = np.where(rsat > 0.)[0]
        if len(ind) > 0:
            lrsat = np.log10(rsat[ind])
            global_minr = min(global_minr, np.min(lrsat))
            global_maxr = max(global_maxr, np.max(lrsat))
            
            # Create histogram
            H, _ = np.histogram(lrsat, bins=np.append(rbins, rmax))
            nsatr += H

            print(f'    Satellites found: {len(ngalsat)}')

    if ntot > 0:
        ytot = np.sum(nsatr)
        avgr = avgr / ntot
            
        # Plot
        ind = np.where(nsatr > 0)[0]
        if len(ind) > 0:
            label = f'K={kval}'
            ax.plot(rhist[ind], np.log10(nsatr[ind]/ytot), 
                    label=label, color=colors[im])
                
            # Add arrow to show median
            dy = 0.05 * (ymax - ymin)
            ax.arrow(np.log10(avgr), ymax - dy, 0, dy, 
                     color=colors[im], length_includes_head=True,
                     head_width=dy*0.1, head_length=dy*0.3)

            if verbose:
                print(f'  {ntot} satellites, avg_r = {avgr:.3f} Mpc/h')
    
    # Add legend
    if len(mock_files) > 0:
        leg = ax.legend(loc='best')
        leg.draw_frame(False)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plotfile = os.path.join(output_dir, f'rp_sat_{base_name}.png')
    fig.savefig(plotfile, dpi=150, bbox_inches='tight')

    if verbose:
        print(f'Radial profile plot: {plotfile}')
        
    return plotfile



def make_test_plots(params,outroot,hmf_file=None,verbose=False):
    """
    Make test plots on mock galaxy catalogues
    
    Parameters:
    -----------
    params : Tuple
        Parameters used to ran the HOD model
    hmf_file : str
        Path to the Halo Mass Function file (for reference/metadata)
    verbose : bool
        True for progress messages
        
    Returns: 0 or -1 (errors)
    --------
    """
    mock_file = params.outfile
    #mock_file = '/home/violeta/buds/hods/galaxies_1000Mpc_V1.4more_NFW_mu12.000_Ac1.0000_As0.50000_vfact1.00_beta0.000_K1.00_vt0pm0_BVG_product_nosubhalos_trunc_binomialextended.dat'
    
    output_dir = outroot / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = io.line_separator()
    print(sep+"\n Plots: "+str(output_dir)+"\n"+sep)
    
    plotfile = plot_satellite_radial_distribution(
        mock_file=mock_file,output_dir=output_dir,verbose=verbose)

    return 0

def plot_correlation_function(
    filename,
    output_png=None,
    loglog=True,
    show=True,
    r_index=0,
    xi_index=1,
    err_index=2 
):
    """
    Plots xi(r) vs r (and optionally error bars) from a .txt file.
    Now supports plotting error bars if present (Corrfunc output).

    Parameters:
    -----------
    filename : str
        Path to the .txt file (output from Corrfunc or CUTE).
    output_png : str or None
        If provided, path to save the plot as a PNG file.
    loglog : bool
        If True, use log-log scale.
    show : bool
        If True, display the plot interactively.
    r_index : int
        Index of the r column.
    xi_index : int
        Index of the xi(r) column.
    err_index : int
        Index of the error column (if present).
    """
    # Load the data
    data = np.loadtxt(filename, comments='#', delimiter=',', skiprows=1)
    r = data[:, r_index]
    xi = data[:, xi_index]
    # If there is a third column, treat as error bars
    has_error = data.shape[1] > err_index

    plt.figure(figsize=(6, 5))
    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    if has_error:
        err = data[:, err_index]
        plt.errorbar(r, xi, yerr=err, fmt='+', capsize=2, label=r'$\xi(r)$ with errors')
    else:
        plt.plot(r, xi, marker='+', linestyle='-', label=r'$\xi(r)$')

    plt.xlabel(r"$r$ [$h^{-1}$ Mpc]")
    plt.ylabel(r"$\xi(r)$")
    plt.title("Galaxy 2-Point Correlation Function")
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()

    if output_png:
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        print(f" Plot saved to: {output_png}")

    if show:
        plt.show()

    plt.close()

def locate_interval(val, edges):
    '''
    Get the index, i, of the interval, [), to which val belongs.
    If outside the limits, using values -1 or the number of bins+1.

    Parameters
    ----------
    val : int or float or array of ints or floats
        Value to evaluate
    edges : array of int or floats
        Array of the n edges for the (n-1) intervals
        
    Returns
    -------
    jl : integer
        Index of the interval, [edges(jl),edges(jl+1)), where val is place.
    '''

    n = edges.size
    jl = np.searchsorted(edges, val, side='right') - 1
    jl = np.clip(jl, -1, n - 1)
    return jl

import numpy as np
import matplotlib.pyplot as plt

def plot_hod_occupation_from_mock(
    mock_file: str,
    halo_catalog_file: str,
    output_png: str = None,
    n_bins: int = 70,
    mass_col_mock: int = 6,
    halo_id_col_mock: int = 14,
    is_central_col: int = 15,
    mass_col_halo: int = 6,
    halo_id_col_halo: int = 9,
    logM_range: tuple = (10.5,14.5),
    show: bool = True
):
    """
    Plot average HOD occupation (centrals and satellites) from the mock catalog,
    using the original halo catalog for correct normalization.
    """
    print(f"Loading mock catalog: {mock_file}")
    mock_data = np.loadtxt(mock_file)
    masses_mock = mock_data[:, mass_col_mock]
    halo_ids_mock = mock_data[:, halo_id_col_mock].astype(int)
    is_central = mock_data[:, is_central_col]

    print(f"Loading halo catalog: {halo_catalog_file}")
    halos_data = np.loadtxt(halo_catalog_file)
    masses_halo = halos_data[:, mass_col_halo]
    halo_ids_halo = halos_data[:, halo_id_col_halo].astype(int)

    if logM_range is None:
        logM_min, logM_max = np.nanpercentile(masses_halo, [0.5, 99.5])
    else:
        logM_min, logM_max = logM_range
    bins = np.linspace(logM_min, logM_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Pre-indexing: for each halo_id in mock, get indices
    # Build a dictionary: halo_id -> indices in mock
    from collections import defaultdict
    halo_idx_dict = defaultdict(list)
    for i, hid in enumerate(halo_ids_mock):
        halo_idx_dict[hid].append(i)

    # Assign each halo in catalog to a mass bin
    bin_index = np.digitize(masses_halo, bins) - 1

    ncen_per_bin = np.zeros(n_bins)
    nsat_per_bin = np.zeros(n_bins)
    n_halos_bin = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        halos_in_bin = halo_ids_halo[bin_index == i]
        n_halos_b = len(halos_in_bin)
        n_halos_bin[i] = n_halos_b
        if n_halos_b > 0:
            centrals_in_bin = 0
            satellites_in_bin = 0
            for hid in halos_in_bin:
                idxs = halo_idx_dict.get(hid, [])
                if len(idxs) > 0:
                    centrals_in_bin += np.sum(is_central[idxs] == 1)
                    satellites_in_bin += np.sum(is_central[idxs] == 0)
            ncen_per_bin[i] = centrals_in_bin / n_halos_b
            nsat_per_bin[i] = satellites_in_bin / n_halos_b
        else:
            ncen_per_bin[i] = 0
            nsat_per_bin[i] = 0

    plt.figure(figsize=(7, 5))
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(bin_centers, ncen_per_bin, label=r'$\langle N_{\rm cen}\rangle$ (centrals)')
    plt.plot(bin_centers, nsat_per_bin, label=r'$\langle N_{\rm sat}\rangle$ (satellites)')
    plt.xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
    plt.ylabel('Average occupation per halo')
    plt.title('Halo Occupation Distribution (mock, normalized by halo catalog)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_png:
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        print(f"Plot saved to: {output_png}")
    if show:
        plt.show()
    plt.close()


def plot_velocity_distribution_mock(
    mock_file: str,
    n_bins: int = 200,
    output_dir: str = None,
    show: bool = True,
    vr_input_bins=None,
    vr_input_pdf=None,
    vtan_input_bins=None,
    vtan_input_pdf=None
):
    """
    Plots the histogram (not normalized) of radial and tangential velocity components
    of satellite galaxies in the mock catalog, using SAGE bin edges for fair comparison.
    Optionally overlays the input SAGE histograms.

    The y-axis is: N_counts / bin_width (not normalized to area=1).

    Parameters
    ----------
    mock_file : str
        Path to the mock catalog .txt file.
    n_bins : int
        Number of bins (only used if input bins are not given).
    output_dir : str or None
        If provided, saves the plot to this directory.
    show : bool
        If True, displays the plot.
    vr_input_bins, vr_input_pdf : arrays or None
        Input SAGE histogram for v_rad (optional; bins REQUIRED for comparison).
    vtan_input_bins, vtan_input_pdf : arrays or None
        Input SAGE histogram for |v_tan| (optional; bins REQUIRED for comparison).
    """
    print(f"Reading mock catalog: {mock_file}")
    data = np.loadtxt(mock_file)
    if data.ndim == 1:
        data = data[None, :]
    # Column indices
    dx_idx, dy_idx, dz_idx = 11, 12, 13
    dvx_idx, dvy_idx, dvz_idx = 8, 9, 10
    is_central_idx = 15

    # Select satellites only
    is_sat = data[:, is_central_idx] == 0
    if np.sum(is_sat) == 0:
        print("No satellites found in mock catalog.")
        return

    dx = data[is_sat, dx_idx]
    dy = data[is_sat, dy_idx]
    dz = data[is_sat, dz_idx]
    dvx = data[is_sat, dvx_idx]
    dvy = data[is_sat, dvy_idx]
    dvz = data[is_sat, dvz_idx]

    r_vecs = np.stack([dx, dy, dz], axis=1)
    r_hats = r_vecs / np.linalg.norm(r_vecs, axis=1, keepdims=True)
    dv_vecs = np.stack([dvx, dvy, dvz], axis=1)

    # Radial component of velocity
    v_rad = np.sum(dv_vecs * r_hats, axis=1)
    # Tangential modulus
    vtan_mod = np.sqrt(np.sum(dv_vecs**2, axis=1) - v_rad**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Radial ----
    if vr_input_bins is not None:
        hist_vr, bin_edges_vr = np.histogram(v_rad, bins=vr_input_bins, density=False)
    else:
        hist_vr, bin_edges_vr = np.histogram(v_rad, bins=n_bins, density=False)
    bin_widths_vr = np.diff(bin_edges_vr)
    bin_centers_vr = 0.5 * (bin_edges_vr[:-1] + bin_edges_vr[1:])
    pdf_vr = hist_vr / bin_widths_vr

    axes[0].plot(bin_centers_vr, pdf_vr, marker='o', linestyle='-', label="Mock satellites")
    axes[0].set_xlabel(r"$v_{\rm rad}$ [km/s]")
    axes[0].set_ylabel(r"$N_{\rm sat} / \Delta v$")
    axes[0].set_title("Radial velocity distribution")
    # Overlay input PDF if provided
    if vr_input_bins is not None and vr_input_pdf is not None:
        bin_centers = 0.5 * (vr_input_bins[:-1] + vr_input_bins[1:])
        axes[0].plot(bin_centers, vr_input_pdf, color="k", linestyle="--", label="Input SAGE")
    axes[0].legend()

    # ---- Tangential ----
    if vtan_input_bins is not None:
        hist_vtan, bin_edges_vtan = np.histogram(vtan_mod, bins=vtan_input_bins, density=False)
    else:
        hist_vtan, bin_edges_vtan = np.histogram(vtan_mod, bins=n_bins, density=False)
    bin_widths_vtan = np.diff(bin_edges_vtan)
    bin_centers_vtan = 0.5 * (bin_edges_vtan[:-1] + bin_edges_vtan[1:])
    pdf_vtan = hist_vtan / bin_widths_vtan

    axes[1].plot(bin_centers_vtan, pdf_vtan, marker='o', linestyle='-', label="Mock satellites")
    axes[1].set_xlabel(r"$|v_{\rm tan}|$ [km/s]")
    axes[1].set_ylabel(r"$N_{\rm sat} / \Delta v$")
    axes[1].set_title("Tangential velocity modulus distribution")
    if vtan_input_bins is not None and vtan_input_pdf is not None:
        bin_centers = 0.5 * (vtan_input_bins[:-1] + vtan_input_bins[1:])
        axes[1].plot(bin_centers, vtan_input_pdf, color="k", linestyle="--", label="Input SAGE")
    axes[1].legend()

    plt.tight_layout()
    if output_dir is not None:
        outpath = f"{output_dir}/velocity_distribution_mock.png"
        plt.savefig(outpath, dpi=150)
        print(f"Plot saved to: {outpath}")
    if show:
        plt.show()
    plt.close()