# hod_config.py

"""
Canonical column conventions and name-resolution maps for halo catalogues
used by the HOD mock-generation pipeline.

Overview
--------
This module centralizes the mapping between heterogeneous input catalogue
formats and the internal canonical halo representation expected by the
pipeline.

Halo catalogues may come from different simulation codes,
post-processing pipelines, or user-generated files, and therefore may differ in:

- column order,
- field names,
- presence or absence of structural radii,
- file format (TXT or HDF5).

To isolate that complexity, this module defines:

1. A canonical column order used throughout the codebase once a catalogue
   has been normalized.
2. Accepted name synonyms for each logical halo property in:
   - HDF5 structured datasets,
   - TXT catalogues with headers.
3. Default column positions for TXT files without headers.

Canonical representation
------------------------
After normalization, halo catalogues are interpreted using the following order:

    [x, y, z, vx, vy, vz, logM, Rvir, Rs, id]

where:
- 'x, y, z'   are halo positions,
- 'vx, vy, vz' are halo bulk velocities,
- 'logM'      is the base-10 logarithm of halo mass,
- 'Rvir'      is the virial radius,
- 'Rs'        is the scale radius,
- 'id'        is the halo identifier.

Units
-----
This module only defines naming and positional conventions; it does not itself
perform unit conversions. Unit handling is performed downstream in the I/O layer
(e.g. 'hod_io.py').

Design philosophy
-----------------
The purpose of this module is to keep catalogue-format assumptions centralized
and explicit. This has several advantages:

- easier maintenance when new input formats are added,
- clearer separation between I/O parsing and physical modelling,
- reduced risk of silent column mismatches,
- consistent normalization across TXT and HDF5 readers.

Notes
-----
- The 'WITH_R' variants correspond to catalogues that include both 'Rvir' and 'Rs'.
- The 'NO_R' variants correspond to catalogues where these radii are absent.
- These maps are consumed primarily by 'src.hod_madrid_py.hod_io'.
- No file I/O is performed here; this module is purely declarative.
"""

# =============================================================================
# Canonical order after catalogue normalization
# =============================================================================
# Standard internal column order used by the HOD pipeline once an input catalogue
# has been parsed and converted to a homogeneous representation.
CANONICAL_ORDER = ["x", "y", "z", "vx", "vy", "vz", "logM", "Rvir", "Rs", "id"]

# =============================================================================
# Name maps for HDF5 and TXT files with headers
# =============================================================================
# Each dictionary maps a logical internal key to a list of accepted synonyms that
# may appear in an external file. These synonym lists are used to resolve actual
# field names to canonical keys.
#
# Matching behavior (case-sensitive or not) is controlled downstream by the I/O
# utilities that consume these maps.
H5_NAME_MAP_WITH_R = {
    "x":   ["x", "X", "pos_x", "position_x"],
    "y":   ["y", "Y", "pos_y", "position_y"],
    "z":   ["z", "Z", "pos_z", "position_z"],
    "vx":  ["vx", "VX", "vel_x", "velocity_x"],
    "vy":  ["vy", "VY", "vel_y", "velocity_y"],
    "vz":  ["vz", "VZ", "vel_z", "velocity_z"],
    "logM":["logM", "log_mass", "log10_mass", "Mlog10", "M_log10"],
    "Rvir":["Rvir", "R_vir", "virial_radius"],
    "Rs":  ["Rs", "R_s", "scale_radius"],
    "id":  ["id", "ID", "halo_id", "haloid", "HaloID"],
}

# Variant for catalogues without structural radii.
# This is useful when the input halo catalogue does not provide virial radii
# or scale radii explicitly.
H5_NAME_MAP_NO_R = {
    k: v for k, v in H5_NAME_MAP_WITH_R.items() if k not in ("Rvir", "Rs")
}

# For TXT files with headers, the same accepted synonyms are used.
TXT_NAME_MAP_WITH_R = H5_NAME_MAP_WITH_R
TXT_NAME_MAP_NO_R   = H5_NAME_MAP_NO_R

# =============================================================================
# Default column positions for TXT files without headers
# =============================================================================
# These dictionaries are intended for plain-text catalogues with no header line.
# In that case, the reader assumes a fixed positional convention.

# TXT format with radii:
#   x y z vx vy vz logM Rvir Rs id
TXT_POS_WITH_R = {
    "x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "logM": 6, "Rvir": 7, "Rs": 8, "id": 9
}

# TXT format without radii:
#   x y z vx vy vz logM id
TXT_POS_NO_R = {
    "x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "logM": 6, "id": 7
}

