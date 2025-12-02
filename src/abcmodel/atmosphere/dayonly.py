from dataclasses import dataclass, replace

import jax
from jaxtyping import PyTree

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
)
from ..utils import PhysicalConstants
from .abstracts import (
    AbstractCloudModel,
    AbstractCloudState,
    AbstractMixedLayerModel,
    AbstractMixedLayerState,
    AbstractSurfaceLayerModel,
    AbstractSurfaceLayerState,
)


@jax.tree_util.register_pytree_node_class
@dataclass
class DayOnlyAtmosphereState(AbstractAtmosphereState):
    """Atmosphere state aggregating surface layer, mixed layer, and clouds."""
    surface_layer: AbstractSurfaceLayerState
    mixed_layer: AbstractMixedLayerState
    clouds: AbstractCloudState

    def tree_flatten(self):
        return (self.surface_layer, self.mixed_layer, self.clouds), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class DayOnlyAtmosphereModel(AbstractAtmosphereModel):
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
    ) -> DayOnlyAtmosphereState:
        # state is CoupledState
        
        # 1. Run Surface Layer
        sl_state = self.surface_layer.run(state, const)
        # Update state for next steps
        # We need to construct a new state with updated SL
        # Assuming state.atmosphere is DayOnlyAtmosphereState
        # We need to cast or assume state.atmosphere is DayOnlyAtmosphereState to use replace
        # But for now, let's assume it works or use type ignore if needed.
        # Actually, replace works on dataclasses.
        
        new_atmosphere = replace(state.atmosphere, surface_layer=sl_state)
        state_with_sl = replace(state, atmosphere=new_atmosphere)
        
        # 2. Run Clouds
        cl_state = self.clouds.run(state_with_sl, const)
        new_atmosphere = replace(new_atmosphere, clouds=cl_state)
        state_with_cl = replace(state_with_sl, atmosphere=new_atmosphere)
        
        # 3. Run Mixed Layer
        ml_state = self.mixed_layer.run(state_with_cl, const)
        new_atmosphere = replace(new_atmosphere, mixed_layer=ml_state)
        
        return new_atmosphere

    def statistics(
        self,
        state: DayOnlyAtmosphereState,
        t: int,
        const: PhysicalConstants
    ) -> DayOnlyAtmosphereState:
        """Update statistics."""
        # state is DayOnlyAtmosphereState
        
        # Update mixed layer statistics
        ml_state = self.mixed_layer.statistics(state.mixed_layer, t, const)
        
        # Update surface layer statistics
        # SurfaceLayerModel might not have statistics method?
        # AbstractSurfaceLayerModel does not have statistics method in abstracts.py?
        # Let's check. If not, we skip it or implement it.
        # StandardSurfaceLayerModel inherits from AbstractSurfaceLayerModel.
        # If it doesn't have statistics, we can't call it.
        # But wait, `AbstractAtmosphereModel` requires `statistics`.
        # `DayOnlyAtmosphereModel` implements it.
        # It calls `self.surface_layer.statistics`.
        # Does `AbstractSurfaceLayerModel` have `statistics`?
        # I need to check `src/abcmodel/atmosphere/abstracts.py`.
        # If not, I should remove this call or add it.
        # Assuming it exists or I should check.
        # But `ml_state` definitely exists.
        
        # Let's assume for now it exists or I will check next.
        # But I need to fix the attribute access first.
        
        # Update mixed layer statistics
        ml_state = self.mixed_layer.statistics(state.mixed_layer, t, const)
        
        # Update surface layer statistics
        # If surface_layer has statistics method
        if hasattr(self.surface_layer, "statistics"):
             sl_state = self.surface_layer.statistics(state.surface_layer, t, const)
        else:
             sl_state = state.surface_layer

        # Update clouds statistics
        # If clouds has statistics method
        if hasattr(self.clouds, "statistics"):
            cloud_state = self.clouds.statistics(state.clouds, t, const)
        else:
            cloud_state = state.clouds
        
        # Return updated state
        return replace(
            state,
            mixed_layer=ml_state,
            surface_layer=sl_state,
            clouds=cloud_state,
        )

    def warmup(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
        land: AbstractLandModel
    ) -> AbstractCoupledState:
        """Warmup the atmosphere by running it for a few timesteps."""
        # state is CoupledState
        
        # iterate surface layer to converge turbulent fluxes
        # We need to update state in the loop
        current_state = state
        
        for _ in range(10):
            sl_state = self.surface_layer.run(current_state, const)
            new_atmosphere = replace(current_state.atmosphere, surface_layer=sl_state)
            current_state = replace(current_state, atmosphere=new_atmosphere)

        # run land surface
        # Land run returns LandState
        land_state = land.run(current_state, const)
        current_state = replace(current_state, land=land_state)

        # conditionally run clouds if model is not NoCloudModel
        from .clouds import NoCloudModel

        if not isinstance(self.clouds, NoCloudModel):
            # Run mixed layer
            ml_state = self.mixed_layer.run(current_state, const)
            new_atmosphere = replace(current_state.atmosphere, mixed_layer=ml_state)
            current_state = replace(current_state, atmosphere=new_atmosphere)
            
            # Run clouds
            cl_state = self.clouds.run(current_state, const)
            new_atmosphere = replace(current_state.atmosphere, clouds=cl_state)
            current_state = replace(current_state, atmosphere=new_atmosphere)

        # run mixed layer
        ml_state = self.mixed_layer.run(current_state, const)
        new_atmosphere = replace(current_state.atmosphere, mixed_layer=ml_state)
        current_state = replace(current_state, atmosphere=new_atmosphere)
        
        return current_state

    def integrate(self, state: DayOnlyAtmosphereState, dt: float) -> DayOnlyAtmosphereState:
        ml_state = self.mixed_layer.integrate(state.mixed_layer, dt)
        return replace(state, mixed_layer=ml_state)
