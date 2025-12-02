#hod_const.py

# ---------------- Default values ----------------
chunk_size = 10000          # rows per I/O batch when streaming large files
default_gamma = -1.4        # high-mass slope for central occupation (HOD3)
report_after_nlines = 100000  # print progress every N lines (set 0/None to disable)

# ------------- Cosmological constants -----------
rho_crit = 27.755e10        # critical density at z=0 [Msun / (Mpc/h)^3]
pi = 3.141592653589793      # π (double precision)