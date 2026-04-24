#!/usr/bin/env python3
"""
===============================================================================
        MOCK GALAXY GENERATOR: HOD-BASED GALAXY-HALO CONNECTION
===============================================================================

Authors:       Joaquín Delgado Amar, Violeta González Pérez
Supervisor:    Violeta González Pérez
Institution:   Universidad Autónoma de Madrid (UAM)
Created:       07 Jul 2025
Last Updated:  24 April 2026

-------------------------------------------------------------------------------
PURPOSE
-------------------------------------------------------------------------------
Generate mock galaxy catalogues from dark-matter halo catalogues using 
Halo Occupation Distribution (HOD) models. 
The code allows flexible combinations of analytical prescriptions and 
empirical inputs provided by the user.

-------------------------------------------------------------------------------
MODES
-------------------------------------------------------------------------------
• HOD shape:
    - Analytical: parametric forms (error function, Gaussian, asymmetric Gaussian).
    - Empirical: mean ⟨N_cen⟩, ⟨N_sat⟩ read from user-supplied tables (text or HDF5).
• Radial profiles:
    - Analytical: standard NFW or extended NFW (Reyes-Pedraza 2024).
    - Empirical: histograms of satellite radii (text or HDF5).
• Velocity profiles:
    - Analytical: virial theorem or extended laws (Reyes-Pedraza 2024):
      (triple Gaussian for v_rad, exponential-power law for v_tan).
    - Empirical: velocity histograms (v_rad, |v_tan|).
• Conformity: optional K1, K2 factors (Reyes-Pedraza 2024)
    - Only for empirical HOD shape.
    - Mass-dependent factors to be implemented.

-------------------------------------------------------------------------------
INPUTS
-------------------------------------------------------------------------------
1) Halo catalogue (ASCII): x, y, z, vx, vy, vz, logM, Rvir(optional), Rs(optional), ID
2) Optional external files from SAM (ASCII) for empirical models:
   - HOD shape tables
   - Radial distributions
   - Velocity histograms
   - Conformity parameters

-------------------------------------------------------------------------------
INTEGRATION WITH HODFIT2SIM
-------------------------------------------------------------------------------

HODfit2sim is a companion code that fits HOD prescriptions to semi-analytic
galaxy catalogues and outputs the calibrated occupation, radial, and velocity
distributions in HDF5 format. This mock generator can directly read those outputs, 
allowing both codes to operate in tandem for a seamless pipeline from calibration
to mock construction.

If you have run HODfit2sim, you can use its HDF5 output as external input file:
   - Set HODfit2sim = True
   - Provide the path to the HDF5 file

-------------------------------------------------------------------------------
OUTPUTS
-------------------------------------------------------------------------------
• Mock galaxy catalogue (ASCII): galaxy positions, velocities, halo IDs, type.
• Diagnostic plots (PNG): HOD shapes, radial/velocity profiles, 2PCF (real & RSD).
• Corrfunc-ready position files.

-------------------------------------------------------------------------------
WORKFLOW
-------------------------------------------------------------------------------
1) Configure analytical/empirical flags.
2) Load halo catalogue and optional external input files.
3) Populate haloes with centrals and satellites.
4) Assign positions, velocities, and conformity.
5) Save the final mock catalogue and diagnostics.

-------------------------------------------------------------------------------
PERFORMANCE
-------------------------------------------------------------------------------
Critical routines are accelerated with Numba (JIT). 

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
NumPy, SciPy, Matplotlib, h5py, Corrfunc, Numba, and project modules:
src.hod, src.hod_io, src.hod_shape, src.hod_r_profile, src.hod_v_profile, 
src.hod_cosmology, src.hod_pdf, src.hod_plots, src.hod_corrfunc, src.hod_const

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
$ python produce_hod_mock.py
Configuration is controlled via user-defined flags and parameter files.

-------------------------------------------------------------------------------
CONTACT
-------------------------------------------------------------------------------
joaquin.delgado@estudiante.uam.es
violetagp@protonmail.com

References: Avila et al. (2020); Reyes Pedraza (2024); HODfit2sim documentation.
===============================================================================
"""

from pathlib import Path
import src.hod_madrid_py.hod as hod
import src.hod_madrid_py.hod_io as io
import src.hod_madrid_py.hod_plots as plots
import src.hod_madrid_py.hod_corrfunc as corr


# =============================================================================
# =================== 1. PARAMETER DEFINITION SECTION =========================
# =============================================================================

# Control parameters
PRODUCE_MOCK = True                     # True to produce a mock galaxy catalogue
TEST_PLOTS   = True                     # True to make test plots
VERBOSE      = True                     # Set to True for progress messages
SEED         = 10                      # Random seed for reproducibility

# Cosmological and simulation parameters
ZSNAP = 1.321                           # Snapshot redshift
OMEGA_M = 0.3089                        # Matter density parameter
LBOX = 1000.0                           # Box size in Mpc/h

# Input file parameters
FTYPE = 'txt'                           # Possible input file type: txt, npy
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("data/example")
INFILE = "/home2/guillermo/TFM_JOAQUIN/data/Halos_file_for_hod_Rvir_Rs.txt"
#INFILE = INPUT_DIR / "Halos_tree_x_y_z_vx_vy_vz_logM_Rvir_Rs_id_500000.txt"

# =============================================================================
# ======================= 2. AVERAGE HOD SHAPE ================================
# =============================================================================

ANALYTICAL_SHAPE = False                 # If true, use analytical functions
TEST_HOD_OCCUPATION = True              # If true, plot HOD occupation

# For an analytical shape, the following parameters are needed:
HODSHAPE = 3                            # Shape of the average HOD: 1, 2 or 3
MU = 11.515                             # Log10 of characteristic halo mass
AC = 0.00537                            # Central galaxy amplitude
AS = 0.005301                           # Satellite galaxy amplitude
M0_factor = -0.05                       # mu offset for M0
M1_factor = 0.35                        # mu offset for M1
M0 = 10**(MU + M0_factor)               
M1 = 10**(MU + M1_factor)               
ALPHA = 0.9                             # Satellite power law slope
SIG = 0.08                              # Central galaxy scatter (HOD2 and 3)
GAMMA = -1.4                            # Power law slope for high mass centrals (HOD3)

# If analytical_shape is False, a file should be given:
HODFIT2SIM = True                       # Set to true if SAM was analyzed with HODFIT2SIM
CONFORMITY = True                       # If true, use global conformity parameters for HOD shape
CONFORMITY_FILE = None

if HODFIT2SIM:
    # Output file from HODfit2sim
    HOD_SHAPE_FILE = "/home2/guillermo/TFM_JOAQUIN/catalogues/h2s_output_shuffled.h5"  
else:
    # .txt file with  Mmax Ncen NsMminat in mass bins should be given
    HOD_SHAPE_FILE = "/home2/guillermo/TFM_JOAQUIN/thesis/mock_from_SAGE/HOD_shape_file.txt"
    # .txt file with K1_global K2_global should be given
    CONFORMITY_FILE = "/home2/guillermo/TFM_JOAQUIN/thesis/mock_from_SAGE/CONFORMITY_file.txt"

# =============================================================================
# =================== 2b. DERIVED HOD AMPLITUDES ==============================
# =============================================================================

DERIVE_AMPLITUDES = False                 # If true, derive Ac and As from target ngal and fsat
NGAL_TARGET = 6.731e-4
FSAT_TARGET = 0.0685

# =============================================================================
# ====================== 2c. DERIVED MU FROM BIAS =============================
# =============================================================================

DERIVE_MU = False
BGAL_TARGET = 1.37

BIAS_FILE = "/home2/guillermo/hod_madrid_py/data/example/bias_vs_mass.dat"
BIAS_POLY_DEGREE = 4

MU_MIN = 10.0
MU_MAX = 12.5
MU_TOL = 1.0e-4
MU_MAX_ITER = 100

# =============================================================================
# ============ 3. Probability distribution function for satellites ============
# =============================================================================

# For analytical PDF, the following parameters are needed:
BETA = 0.0                           # Poisson=0, Nearest integer=-2

# =============================================================================
# ============ 4. Radial distribution function for satellites =================
# =============================================================================

ANALYTICAL_RP = True                     # If true, use analytical functions

# For analytical radial distribution, the following parameters are needed:
READ_CONCENTRATIONS = True               # If true, reads individual halo concentrations
K = 1                                    # NFW truncation parameter (from 0 to 1)
# If not using Klypin concentrations, individual halo concentrations will be read from the input file

EXTENDED_NFW = True                     # If true, use extended NFW profile (Analytical)
# If using extended NFW profile (Analytical), the following parameters are needed:
N0 = 3899.85                             # Normalization factor
R0 = 0.33874                             # Scale radius (Mpc/h)
ALPHA_R = 1.23088                        # Inner slope
BETA_R = 3.20228                         # Outer slope
KAPPA_R = -2.08642                       # Transition slope
RMAX = 1.5                              # Maximum radius for the extended profile (Mpc/h)

# If analytical_rp is False, a file should be given:
if HODFIT2SIM:
    # Output file from HODfit2sim
    HOD_RP_FILE = "/home2/guillermo/TFM_JOAQUIN/catalogues/h2s_output_shuffled.h5"
else:
    # .txt file with rmin rmax Nsat Nsat/sum(Nsat) in radius bins should be given
    HOD_RP_FILE = "/home2/guillermo/TFM_JOAQUIN/thesis/mock_from_SAGE/HOD_rp_file.txt"


# =============================================================================
# =============== 5. Two Point Correlation Function (2PCF) ====================
# =============================================================================

TEST_2PCF = True                         # If true, compute the 2PCF
TEST_2PCF_RS = True                      # If true, compute the 2PCF in redshift space

# If using redshift space, the following parameters are needed:
AXIS_RS = 'z'                            # Axis to use for redshift space (x, y, or z)


# =============================================================================
# ============ 6. Velocity distribution function for satellites ===============
# =============================================================================

ANALYTICAL_VP = True                     # If true, use analytical functions

# For analytical velocity distribution, the following parameters are needed:
VFACT = 1.0                              # Velocity factor (alpha_v)
VT = 0.0                                 # Tangential velocity (km/s, =0 or =500)
VTDISP = 0.0                             # Tangential velocity dispersion (km/s, =0 or =200)
EXTENDED_VP = True                      # If true, use extended velocity profile (Analytical)

# If using extended velocity profile (analytical), the following parameters are needed:
# Radial velocity profile (3-Gaussians)
VR1 = -10601.7                         # Amplitude of first Gaussian
VR2 = -12038.5                         # Amplitude of second Gaussian
VR3 = -22749.8                         # Amplitude of third Gaussian
MU1 = -329.893                           # Mean of first Gaussian
MU2 = 255.23                         # Mean of second Gaussian
MU3 = -384.638                               # Mean of third Gaussian
SIGMA1 = -123.386                        # Standard deviation of first Gaussian
SIGMA2 = -246.083                     # Standard deviation of second Gaussian
SIGMA3 = -277.454                         # Standard deviation of third Gaussian

# Tangential velocity profile (Exponential * Power Law)
V0_TAN = 2.66464                          
EPSILON_TAN = 0.779307                       
OMEGA_TAN =  -0.000445626                    
DELTA_TAN = 1.35091                          

# If analytical_vp is False, a file should be given:
if HODFIT2SIM:
    # Output file from HODfit2sim
    HOD_VP_FILE = "/home2/guillermo/TFM_JOAQUIN/catalogues/h2s_output_shuffled.h5"
else:
    # A .txt file with vr_min vr_max Nsat_vr vr_probs vtan_min vtan_max	Nsat_vtan vtan_probs should be given
    HOD_VP_FILE = "/home2/guillermo/TFM_JOAQUIN/thesis/mock_from_SAGE/HOD_vp_file.txt"


def main():
    
    # =========================================================================
    # ======= Collect parameters into a dictionary and check them =============
    # =========================================================================
    
    hod_params = io.create_hod_params(infile=INFILE, outdir=OUTPUT_DIR, ftype=FTYPE, seed=SEED,
                                      zsnap=ZSNAP, Lbox=LBOX, omega_M=OMEGA_M,
                                      analytical_shape=ANALYTICAL_SHAPE,
                                      hod_shape_file=HOD_SHAPE_FILE,
                                      HODfit2sim=HODFIT2SIM, conformity=CONFORMITY, conformity_file=CONFORMITY_FILE,
                                      hodshape=HODSHAPE, mu=MU, Ac=AC, As=AS,
                                      M0=M0, M1=M1, alpha=ALPHA, sig=SIG, gamma=GAMMA,
                                      derive_amplitudes=DERIVE_AMPLITUDES,
                                      derive_mu=DERIVE_MU,
                                      ngal_target=NGAL_TARGET,
                                      fsat_target=FSAT_TARGET,
                                      bgal_target=BGAL_TARGET,
                                      bias_file=BIAS_FILE,
                                      bias_poly_degree=BIAS_POLY_DEGREE,
                                      mu_min=MU_MIN,
                                      mu_max=MU_MAX,
                                      mu_tol=MU_TOL,
                                      mu_max_iter=MU_MAX_ITER, beta=BETA,
                                      analytical_rp=ANALYTICAL_RP, read_concentrations=READ_CONCENTRATIONS,
                                      hod_rp_file=HOD_RP_FILE, K=K, extended_NFW=EXTENDED_NFW,
                                      N0 = N0, r0=R0, Rmax=RMAX, alpha_r=ALPHA_R, beta_r=BETA_R, kappa_r=KAPPA_R,
                                      analytical_vp=ANALYTICAL_VP,
                                      hod_vp_file=HOD_VP_FILE,
                                      vfact=VFACT,vt=VT, vtdisp=VTDISP,
                                      extended_vp=EXTENDED_VP,
                                      vr1=VR1, vr2=VR2, vr3=VR3,
                                      mu1=MU1, mu2=MU2, mu3=MU3,
                                      sigma1=SIGMA1, sigma2=SIGMA2, sigma3=SIGMA3,
                                      v0_tan=V0_TAN, epsilon_tan=EPSILON_TAN,
                                      omega_tan=OMEGA_TAN, delta_tan=DELTA_TAN)

    # =========================================================================
    # =========================== Run the HOD model ===========================
    # =========================================================================

    if hod_params is None:
        result = -1
    else:
        if PRODUCE_MOCK:
            result = hod.run_hod_model(hod_params,verbose=VERBOSE)
        if TEST_PLOTS:
            result = plots.make_test_plots(hod_params,OUTPUT_DIR,
                                     verbose=VERBOSE)
        if TEST_2PCF:
            positions_file = OUTPUT_DIR / "galaxy_positions.txt"
            corr.extract_positions_from_galaxy_catalog(
                hod_params.outfile, positions_file)
            xi_output_file = OUTPUT_DIR / "corrfunc_xi.txt"
            corr.compute_correlation_corrfunc(
                positions_file=str(positions_file),
                output_file=str(xi_output_file),
                boxsize=hod_params.Lbox,
                rmin=1.4e-3,     
                rmax=140.0,
                n_bins=80,
                binning='log',
                n_threads=4,
                verbose=VERBOSE
            )
            plots.plot_correlation_function(
                filename=str(xi_output_file),
                output_png=str(OUTPUT_DIR / "corrfunc_xi.png"),
                loglog=True,
                show=True
            )
        if TEST_2PCF_RS:
            positions_file_rs = OUTPUT_DIR / "galaxy_positions_rs.txt"
            corr.extract_positions_from_galaxy_catalog_rs(
                input_catalog_file=hod_params.outfile,
                output_positions_file=positions_file_rs,
                pos_key=(0, 1, 2),
                vel_key=(3, 4, 5),
                z_snap=1.321,
                Omega_m= OMEGA_M,
                Omega_L=0.6911,
                h=0.6774,
                los_axis=AXIS_RS,
                verbose=VERBOSE
            )
            xi_output_file_rs = OUTPUT_DIR / "corrfunc_xi_rs.txt"
            corr.compute_correlation_corrfunc(
                positions_file=str(positions_file_rs),
                output_file=str(xi_output_file_rs),
                boxsize=hod_params.Lbox,
                rmin=1.4e-3,
                rmax=140.0,
                n_bins=80,
                binning='log',
                n_threads=4,
                verbose=VERBOSE
            )
            plots.plot_correlation_function(
                filename=str(xi_output_file_rs),
                output_png=str(OUTPUT_DIR / "corrfunc_xi_rs.png"),
                loglog=True,
                show=True
            )
        if TEST_HOD_OCCUPATION:
            plots.plot_hod_occupation_from_mock(
                mock_file=hod_params.outfile,
                halo_catalog_file=hod_params.infile, 
                output_png=str(OUTPUT_DIR / "hod_occupation_mock.png"),
                n_bins=200,
                show=True)

    return result


if __name__ == "__main__":    
    # Run the main function
    exit_code = main()
    __import__('sys').exit(exit_code)
