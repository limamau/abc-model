from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...abstracts import AbstractState
from ...utils import PhysicalConstants
from ..abstracts import AbstractSurfaceLayerModel


@dataclass
class SimpleSurfaceLayerInitConds(AbstractState):
    """Minimal surface layer model initial state."""

    ustar: float
    """Surface friction velocity [m s-1]."""
    uw: float = jnp.nan
    """Zonal surface momentum flux [m2 s-2]."""
    vw: float = jnp.nan
    """Meridional surface momentum flux [m2 s-2]."""


class SimpleSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Minimal surface layer model with constant friction velocity."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """Calculate momentum fluxes from wind components and friction velocity."""
        state.ra = compute_ra(state.u, state.v, state.wstar, state.ustar)
        state.uw = compute_uw(state.u, state.v, state.ustar)
        state.vw = compute_vw(state.u, state.v, state.ustar)
        return state


def compute_ra(u: Array, v: Array, wstar: Array, ustar: Array) -> Array:
    """Calculate aerodynamic resistance from wind speed and friction velocity.

    Args:
        u: zonal wind speed [m s-1].
        v: meridional wind speed [m s-1].
        wstar: convective velocity scale [m s-1].
        ustar: friction velocity [m s-1].

    Returns:
        Aerodynamic resistance [s m-1].
    """
    ueff = jnp.sqrt(u**2.0 + v**2.0 + wstar**2.0)
    return ueff / jnp.maximum(1.0e-3, ustar) ** 2.0


def compute_uw(u: Array, v: Array, ustar: Array) -> Array:
    """Calculate zonal momentum flux from wind components and friction velocity.

    Args:
        u: zonal wind speed [m s-1].
        v: meridional wind speed [m s-1].
        ustar: friction velocity [m s-1].

    Returns:
        Zonal momentum flux [m2 s-2].
    """
    return jnp.where(
        u == 0.0,
        0.0,
        -jnp.sign(u) * (ustar**4.0 / (v**2.0 / u**2.0 + 1.0)) ** (0.5),
    )


def compute_vw(u: Array, v: Array, ustar: Array) -> Array:
    """Calculate meridional momentum flux from wind components and friction velocity.

    Args:
        u: zonal wind speed [m s-1].
        v: meridional wind speed [m s-1].
        ustar: friction velocity [m s-1].

    Returns:
        Meridional momentum flux [m2 s-2].
    """
    return jnp.where(
        v == 0.0,
        0.0,
        -jnp.sign(v) * (ustar**4.0 / (u**2.0 / v**2.0 + 1.0)) ** (0.5),
    )
