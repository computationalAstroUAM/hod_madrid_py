# hod_cosmology.py

"""
Basic cosmological helper functions for the HOD mock-generation pipeline.

Overview
--------
This module provides a minimal set of cosmological functions required by the
HOD-based mock generator. The current implementation assumes a flat ΛCDM
background cosmology, i.e.

    Ω_M + Ω_Λ = 1,

and exposes three frequently used quantities:

1. 'Ω_M(z)':
   the matter density parameter at redshift 'z'.

2. 'E(z)^2':
   the dimensionless Hubble expansion factor squared,
   defined through

       E(z)^2 = [H(z) / H0]^2.

3. 'Δ_vir(z)':
   the virial overdensity with respect to the critical density,
   computed using the fitting formula of Bryan & Norman (1998).

Physical assumptions
--------------------
The module assumes:
- a flat ΛCDM cosmology,
- matter density parameter 'omega_M = Ω_M(0)',
- cosmological constant density 'Ω_Λ = 1 - Ω_M'.

No radiation component or curvature term is included, which is appropriate for
the low-to-intermediate redshift regime relevant for the current HOD pipeline.

Units
-----
All functions are dimensionless:
- 'z' is redshift,
- 'omega_M' is the present-day matter density parameter,
- 'Ω_M(z)', 'E(z)^2', and 'Δ_vir(z)' are dimensionless outputs.

Implementation notes
--------------------
- The functions are decorated with 'numba.jit(nopython=True)' for efficient use
  inside performance-critical routines.
- 'Δ_vir(z)' follows the standard Bryan & Norman (1998) approximation for flat
  cosmologies, expressed relative to the critical density.

"""

import math
from numba import jit

@jit(nopython=True)
def omega_Mz(z, omega_M):
    """
    Compute the matter density parameter Ω_M(z) at redshift `z`.

    In a flat ΛCDM cosmology, the redshift evolution of the matter density
    parameter is given by

        Ω_M(z) = Ω_M (1 + z)^3 / [Ω_M (1 + z)^3 + Ω_Λ],

    where

        Ω_Λ = 1 - Ω_M.

    Parameters
    ----------
    z : float
        Redshift at which the matter density parameter is evaluated.
    omega_M : float
        Present-day matter density parameter, Ω_M(0).

    Returns
    -------
    float
        Matter density parameter at redshift `z`, i.e. Ω_M(z).
    """
    return (omega_M * (1.0 + z)**3) / (omega_M * (1.0 + z)**3 + (1.0 - omega_M))


@jit(nopython=True)
def E2(z, omega_M):
    """
    Compute the squared dimensionless Hubble parameter, E(z)^2.

    For a flat ΛCDM cosmology,

        E(z)^2 = [H(z)/H0]^2 = Ω_M (1 + z)^3 + Ω_Λ,

    with

        Ω_Λ = 1 - Ω_M.

    Parameters
    ----------
    z : float
        Redshift at which the expansion factor is evaluated.
    omega_M : float
        Present-day matter density parameter, Ω_M(0).

    Returns
    -------
    float
        The dimensionless quantity E(z)^2.
    """
    return omega_M * (1.0 + z)**3 + (1.0 - omega_M)


@jit(nopython=True)
def Delta_vir(z, omega_M):
    """
    Compute the virial overdensity Δ_vir(z) in a flat ΛCDM cosmology.

    The virial overdensity is defined here relative to the critical density
    and is evaluated using the fitting formula of Bryan & Norman (1998):

        Δ_vir(z) = 18 π^2 + 82 d - 39 d^2,

    where

        d = 1 - Ω_M(z).

    Parameters
    ----------
    z : float
        Redshift at which the virial overdensity is evaluated.
    omega_M : float
        Present-day matter density parameter, Ω_M(0).

    Returns
    -------
    float
        Virial overdensity relative to the critical density at redshift 'z'.
    """
    d = 1.0 - omega_Mz(z, omega_M)
    return 18 * math.pi**2 + 82 * d - 39 * d**2

