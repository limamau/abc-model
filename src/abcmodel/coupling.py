from dataclasses import dataclass, field
from typing import Generic, TypeVar

from simple_pytree import Pytree

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
    AbstractRadiationModel,
    AbstractRadiationState,
)
from .utils import PhysicalConstants


class DiagnosticsState(Pytree):
    """Diagnostic variables for the coupled system."""

    total_water_mass: float = 0.0
    total_energy: float = 0.0


R = TypeVar("R", bound=AbstractRadiationState)
L = TypeVar("L", bound=AbstractLandState)
A = TypeVar("A", bound=AbstractAtmosphereState)


@dataclass
class CoupledState(AbstractCoupledState[R, L, A], Generic[R, L, A]):
    """Hierarchical coupled state, generic over component types."""

    rad: R
    land: L
    atmos: A
    diagnostics: DiagnosticsState = field(default_factory=DiagnosticsState)


class ABCoupler:
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
        self.const = PhysicalConstants()

    @staticmethod
    def init_state(
        rad_state: R,
        land_state: L,
        atmos_state: A,
    ) -> CoupledState[R, L, A]:
        return CoupledState(
            rad=rad_state,
            land=land_state,
            atmos=atmos_state,
        )

    def compute_diagnostics(self, state: AbstractCoupledState) -> AbstractCoupledState:
        """Compute diagnostic variables for total water budget."""
        return state
