#hod_config.py

"""
Canonical column names and default positions for reading halo catalogs
in HDF5 or TXT within the HOD pipeline.

What this module defines:
- CANONICAL_ORDER: the standardized column order after normalizing a catalog.
- *_NAME_MAP_*: accepted name synonyms (per logical key x,y,z,vx,...) to resolve
  column names in HDF5/TXT with headers.
- TXT_POS_*: default column indices for TXT files without headers.

Typical usage (pseudo):
    # If an HDF5 file has dtype.names like
    # ('X','Y','Z','VX','VY','VZ','log_mass','R_vir','R_s','ID'):
    # map each logical key using H5_NAME_MAP_WITH_R.
    # If a TXT has NO header and 10 columns, use TXT_POS_WITH_R.

Notes:
- WITH_R = catalogs that include radii (Rvir and Rs).
- NO_R   = catalogs that do NOT include radii.
- If a required key cannot be resolved, the reader should raise a clear error.
"""

# Canonical order after standardizing a catalog
CANONICAL_ORDER = ["x", "y", "z", "vx", "vy", "vz", "logM", "Rvir", "Rs", "id"]

# =============================================================================
# Name maps for HDF5 and TXT WITH headers
# =============================================================================
# Accepted synonyms per logical key (case-sensitive).
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

# Variant for catalogs WITHOUT radii (Rvir, Rs)
H5_NAME_MAP_NO_R = {
    k: v for k, v in H5_NAME_MAP_WITH_R.items() if k not in ("Rvir", "Rs")
}

# For TXT WITH headers, same synonyms
TXT_NAME_MAP_WITH_R = H5_NAME_MAP_WITH_R
TXT_NAME_MAP_NO_R   = H5_NAME_MAP_NO_R

# =============================================================================
# Default positions for TXT WITHOUT headers
# =============================================================================
# Use these ONLY when the TXT file has no header line with names.

# With radii (10 columns)
TXT_POS_WITH_R = {
    "x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "logM": 6, "Rvir": 7, "Rs": 8, "id": 9
}

# Without radii (8 columns)
TXT_POS_NO_R = {
    "x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5, "logM": 6, "id": 7
}

