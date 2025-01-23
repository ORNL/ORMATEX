"""
Species transport source and sink models.
- liquid to vapor mass transport source/sink
- arrhenius reaction source/sink
"""
import jax
import equinox as eqx
import numpy as np
from jax import numpy as jnp

rgas = 8.3145  # gas constant J/mol/K

@eqx.filter_jit
def mxf_liq_vapor_lin(u_a: jax.Array, u_g: jax.Array, k=1e-2, k_g=1.0, k_a=1.0, a=1.0):
    """
    Linear mass transfer.
    Simplified form of mxf_liq_vapor__nonlin.

    Args:
        u_a: species concentration in the aqueous phase (mol/cc)
        u_g: species concentration in the gas phase (mol/cc)
        k: mass transfer coeff (cm/s)
        a: surface area per unit volume (cm^2/cm^3)

    Returns:
        mass transfer rate in mol/cc/s
    """
    s = k * a * (k_g*u_g - k_a*u_a)
    return s

@eqx.filter_jit
def mxf_liq_vapor_nonlin(u_a: jax.Array, u_g: jax.Array, k=1e-2, k_g=1.0, k_a=1.0, k_s=1e-12):
    """
    Nonlinear species mass transfer.
    Simplified form of mxf_liq_vapor_bubble_ig.

    Args:
        u_a: species concentration in the aqueous phase (mol/cc)
        u_g: species concentration in the gas phase (mol/cc)
        k: mass transfer coeff (cm/s)

    Returns:
        mass transfer rate in mol/cc/s
    """
    s = k * (jnp.power(jnp.clip(u_g, 1e-16, None), 2/3.) + k_s) * (k_g*u_g - k_a*u_a)
    return s

@eqx.filter_jit
def mxf_liq_vapor_bubble_ig(u_a: jax.Array, u_g: jax.Array, cvol: float, alpha_g=0.01, k=1e-2, h=1e-4, nb=10., T=900.):
    """
    Model of dissolved aqueous species to vapor bubble mass
    tranport.  It is assumed vapor bubbles of some diameter
    and some number density already exist in the flow.
    The existing bubbles are termed carrier bubbles.
    The gas is assumed to be ideal.

    Refs:
        M. T. Robinson. A theoretical study of Xe poisoning
        kinetics in fluid-fueled, gas-sparged nuclear reactors.
        ORNL-1924. 1956. pg. 6, eq. 3.11.
        https://media.githubusercontent.com/media/openmsr/msr-archive/master/docs/ORNL-1924.pdf

        Engel, J. R., and Steffy, R. C. XENON BEHAVIOR IN
        THE MOLTEN SALT REACTOR EXPERIMENT.
        ORNL-TM-3464. 1971. pg. 33. Web. doi:10.2172/4731186.
        https://www.osti.gov/biblio/4731186

    Args:
        u_a: species concentration in the aqueous phase (mol/cc_tot)
        u_g: species concentration in the gas bubble (mol/cc_tot)
        cvol: cell volume (cc)
        alpha_g: total volume fraction of the carrier gas phase (cc_gas/cc_tot)
        k: mass transfer coefficient (cm/s)
        nb: bubble number density (#/cc)
        h: henry gas constant in (mol/J) or (mol/cc/Pa)
        T: temperature (K)

    Returns:
        mass transfer rate in (mol/cc/s)
    """
    # surface area per unit volume (cm^2/cm^3)
    a_b = bubble_surface_area(u_g, cvol, alpha_g=alpha_g, nb=nb, T=T)
    rate = k * a_b * ((u_g / alpha_g) * h * rgas * T - u_a / (1.-alpha_g))
    return rate


@eqx.filter_jit
def bubble_surface_area(u_g: jax.Array, cvol: float, alpha_g=0.01, nb=10, a_k=1e-1, T=900., P=101.3e3):
    """
    Computes a_b(u) where a_b is the bubble surface area
    per unit volume (cm^2/cm^3).

    Args:
        u_g: gas species concentration [mol/cc]
        cvol: finite element cell volume [cc]
        alpha_g: total volume fraction of the carrier gas phase
        nb: number of bubbles per cc [#/cm^3]
        T: temperature in [K]
        P: pressure in [Pa]
        a_k: bubble surface area area factor (unitless)
    """
    # [cc] carrier gas vol without any added species
    vg_carrier = cvol * alpha_g
    # volume of 1 carrier bubble
    vg_1cb = vg_carrier / nb
    # u_g in mol/cc, convert to volume PV=nRT, assuming P=1atm
    patm = P * 1e-6  # pressure in (J/cc)
    # V=nRT/P
    # [cc] * [mol/cc] * [cc/J] * [J/mol/K] * [K] => cc
    vg = (cvol*u_g / patm) * rgas * T
    vg_1b = a_k * vg / nb
    vg_tot = vg_1cb + vg_1b
    vg_radius = ((3/4.)*vg_tot/jnp.pi) ** (1/3.)
    ab = nb * 4.0 * jnp.pi * (vg_radius ** 2.0)
    return ab


@eqx.filter_jit
def mxf_diffusion_sorption(u_a: jax.Array, u_s: jax.Array, alpha_s, k=1e-4, a=1.0):
    """
    Computes rate of mass transfer due to diffusion into a solid

    Args:
        a: surface area per volume (cm^2/cm^3)
        alpha_s: solid volume per cell volume (cc_solid/cc_tot)
        k: mass transfer coefficinet (cm/s)
    """
    return -k * a * (u_a - u_s/alpha_s)


@eqx.filter_jit
def mxf_arrhenius(u_a: jax.Array, u_b: jax.Array, e: float, apre: float, m_a=1.0, m_b=1.0, T=900.):
    """
    Computes rate of mass transfer due to Arrhenius reaction.
    https://en.wikipedia.org/wiki/Reaction_rate_constant

    Args:
        u_a: species a concentration [mol/cc]
        u_b: species b concentration [mol/cc]
        m_a: stoichiometric coefficient of species a
        m_b: stoichiometric coefficient of species b
        e: activation energy [J/mol]
        apre: Arrhenius prefactor
    """
    u_stoich = (u_a ** m_a) * (u_b ** m_b)
    return u_stoich * apre * jnp.exp(-e / (rgas * T))


def mxf_liq_vapor_raoults(u_a: jax.Array, u_g: jax.Array, u_tot: jax.Array, alpha_g: float, p_si: float, gamma=1.0, phi=1.0, k=1e-2, p_tot=101.3e3):
    r"""
    TODO:  https://en.wikipedia.org/wiki/Raoult%27s_law
    """
    raise NotImplementedError
