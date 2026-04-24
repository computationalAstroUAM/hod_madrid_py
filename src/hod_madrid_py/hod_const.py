# hod_const.py

'''
Global constants and default configuration values for the HOD mock pipeline.

Overview
--------
This module centralizes a small set of numerical constants and default runtime
parameters that are shared across the codebase. Its purpose is to avoid
hard-coded values being scattered through different modules and to make the
global assumptions of the pipeline explicit.

The quantities defined here fall into two categories:

1. Runtime defaults
   Parameters controlling chunked I/O and progress reporting.

2. Physical constants
   Cosmological quantities used in halo structural calculations.

Conventions
-----------
- Distances are expressed in comoving Mpc/h.
- Masses are expressed in Msun/h.
- Densities are therefore expressed in Msun / (Mpc/h)^3.
- The critical density stored here corresponds to the z = 0 reference value
  adopted by the pipeline.
'''

# =============================================================================
# Default runtime parameters
# =============================================================================

# Number of halo rows processed per I/O chunk when streaming large input files.
# This value affects memory usage and throughput during mock generation.
chunk_size = 100000         

# Default high-mass slope parameter used in the analytical HOD3 central
# occupation when no explicit value is supplied by the user.
default_gamma = -1.4        

# Interval, in processed halo rows, at which progress information may be printed
# during chunked execution. Set to 0 or None downstream to disable reporting.
report_after_nlines = 100000 

# =============================================================================
# Cosmological constants
# =============================================================================

# Critical density at z = 0 in units of Msun / (Mpc/h)^3.
# This value is used, for example, in virial-radius calculations when halo
# radii are derived from mass and overdensity.
rho_crit = 27.755e10        

# Numerical value of pi.
pi = 3.141592653589793      