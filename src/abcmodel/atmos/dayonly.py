from dataclasses import dataclass, replace
from typing import Generic

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
)
from ..utils import PhysicalConstants
from .abstracts import (
    CL,
    ML,
    SL,
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)
from .clouds import NoCloudModel


@dataclass
class DayOnlyAtmosphereState(AbstractAtmosphereState, Generic[SL, ML, CL]):
    """Atmosphere state aggregating surface layer, mixed layer, and clouds."""

    surface: SL
    mixed: ML
    clouds: CL


class DayOnlyAtmosphereModel(AbstractAtmosphereModel[DayOnlyAtmosphereState]):
    """Atmosphere model aggregating surface layer, mixed layer, and clouds during the day-time."""

    def __init__(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
        clouds: AbstractCloudModel,
    ):
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.clouds = clouds

    def run(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> AbstractCoupledState:
        sl_state = self.surface_layer.run(state, const)
        new_atmos = replace(state.atmos, surface=sl_state)
        state_with_sl = replace(state, atmos=new_atmos)
        cl_state = self.clouds.run(state_with_sl, const)
        new_atmos = replace(new_atmos, clouds=cl_state)
        state_with_cl = replace(state_with_sl, atmos=new_atmos)
        ml_state = self.mixed_layer.run(state_with_cl, const)
        new_atmos = replace(new_atmos, mixed=ml_state)
        return new_atmos

    def statistics(
        self, state: AbstractCoupledState, t: int, const: PhysicalConstants
    ) -> AbstractCoupledState:
        """Update statistics."""
        return self.mixed_layer.statistics(state, t, const)

    def warmup(
        self,
        radmodel: AbstractRadiationModel,
        landmodel: AbstractLandModel,
        state: AbstractCoupledState,
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> AbstractCoupledState:
        """Warmup the atmos by running it for a few timesteps."""
        state = state.replace(
            atmos=self.statistics(state, t, const),
        )
        state = state.replace(rad=radmodel.run(state, t, dt, const))
        for _ in range(10):
            sl_state = self.surface_layer.run(state, const)
            atmostate = replace(state.atmos, surface=sl_state)
            state = state.replace(atmos=atmostate)
        land_state = landmodel.run(state, const)
        state = state.replace(land=land_state)

        # this is if clause is ok because it's outise the scan!
        if not isinstance(self.clouds, NoCloudModel):
            ml_state = self.mixed_layer.run(state, const)
            new_atmos = replace(state.atmos, mixed=ml_state)
            state = state.replace(atmos=new_atmos)
            cl_state = self.clouds.run(state, const)
            new_atmos = replace(state.atmos, clouds=cl_state)
            state = state.replace(atmos=new_atmos)
        ml_state = self.mixed_layer.run(state, const)
        new_atmos = replace(state.atmos, mixed=ml_state)
        state = state.replace(atmos=new_atmos)
        return state

    def integrate(
        self, state: DayOnlyAtmosphereState, dt: float
    ) -> DayOnlyAtmosphereState:
        ml_state = self.mixed_layer.integrate(state.mixed, dt)
        return replace(state, mixed=ml_state)
