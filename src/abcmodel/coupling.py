from dataclasses import dataclass, field
from typing import Generic

import jax.numpy as jnp
from flax import nnx
from jax import Array

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    AtmosT,
    LandT,
    RadT,
)


@dataclass
class CoupledState(
    AbstractCoupledState[RadT, LandT, AtmosT], Generic[RadT, LandT, AtmosT]
):
    """Hierarchical coupled state, generic over component types."""

    rad: RadT
    land: LandT
    atmos: AtmosT
    t: Array = field(default_factory=lambda: jnp.array(-1))
    total_water_mass: Array = field(default_factory=lambda: jnp.array(0.0))


class ABCoupler(nnx.Module):
    """Coupling class to bound all the components."""

    def __init__(
        self,
        rad: AbstractRadiationModel,
        land: AbstractLandModel,
        atmos: AbstractAtmosphereModel,
    ):
        self.rad = rad
        self.land = land
        self.atmos = atmos

    def init_state(
        self,
        rad_state: RadT | None,
        land_state: LandT | None,
        atmos_state: AtmosT | None,
    ) -> CoupledState[RadT, LandT, AtmosT]:
        return CoupledState(
            rad=rad_state if rad_state is not None else self.rad.init_state(),
            land=land_state if land_state is not None else self.land.init_state(),
            atmos=atmos_state if atmos_state is not None else self.atmos.init_state(),
        )

    def compute_diagnostics(self, state: AbstractCoupledState) -> AbstractCoupledState:
        """Compute diagnostic variables for total water budget."""
        # limamau: this needs to be re-implemented
        total_water_mass = jnp.array(0.0)
        return state.replace(total_water_mass=total_water_mass)
