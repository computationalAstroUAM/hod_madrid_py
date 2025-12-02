# hod_cosmology.py

import math
from numba import jit

@jit(nopython=True)
def omega_Mz(z, omega_M):
    """
    Compute the matter density parameter Ω_M(z) at a given redshift z.

    This function calculates the evolution of the matter density parameter
    with redshift in a flat ΛCDM universe, assuming Ω_M + Ω_Λ = 1.

    Parameters
    ----------
    z : float
        Redshift at which Ω_M(z) is to be evaluated.
    omega_M : float
        Present-day matter density parameter Ω_M(0).

    Returns
    -------
    float
        Matter density parameter Ω_M(z) at redshift z.

    Notes
    -----
    The evolution follows:
        Ω_M(z) = Ω_M (1 + z)^3 / [Ω_M (1 + z)^3 + Ω_Λ]
    where Ω_Λ = 1 - Ω_M.
    """
    return (omega_M * (1.0 + z)**3) / (omega_M * (1.0 + z)**3 + (1.0 - omega_M))


@jit(nopython=True)
def E2(z, omega_M):
    """
    Compute the dimensionless squared Hubble parameter E(z)^2.

    This represents (H(z)/H0)^2 in a flat ΛCDM cosmology, where H0 is the
    Hubble constant at z=0. 

    Parameters
    ----------
    z : float
        Redshift at which E(z)^2 is to be evaluated.
    omega_M : float
        Present-day matter density parameter Ω_M(0).

    Returns
    -------
    float
        The dimensionless squared Hubble parameter E(z)^2.

    Notes
    -----
    Computed as:
        E(z)^2 = Ω_M (1 + z)^3 + Ω_Λ
    where Ω_Λ = 1 - Ω_M.
    """
    return omega_M * (1.0 + z)**3 + (1.0 - omega_M)


@jit(nopython=True)
def Delta_vir(z, omega_M):
    """
    Compute the virial overdensity Δ_vir(z) in a flat ΛCDM universe.

    The virial overdensity defines the density contrast (relative to the
    critical density) of a virialized halo at redshift z. This function uses
    the fitting formula from Bryan & Norman (1998), which is accurate for
    a wide range of cosmologies.

    Parameters
    ----------
    z : float
        Redshift at which the virial overdensity is evaluated.
    omega_M : float
        Present-day matter density parameter Ω_M(0).

    Returns
    -------
    float
        Virial overdensity Δ_vir(z).

    Notes
    -----
    The fitting formula used is:
        Δ_vir(z) = 18π² + 82d - 39d² where d = 1 - Ω_M(z).
    """
    d = 1.0 - omega_Mz(z, omega_M)
    return 18 * math.pi**2 + 82 * d - 39 * d**2

