from dataclasses import dataclass, replace
from typing import Generic

from jax import Array

from ..abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    LandT,
    RadT,
)
from .abstracts import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
    CloudT,
    MixedT,
    SurfT,
)
from .clouds import NoCloudModel


@dataclass
class DayOnlyAtmosphereState(AbstractAtmosphereState, Generic[SurfT, MixedT, CloudT]):
    """Atmosphere state aggregating surface layer, mixed layer, and clouds."""

    surface: SurfT
    mixed: MixedT
    clouds: CloudT

    @property
    def theta(self) -> Array:
        """Potential temperature [K]."""
        return self.mixed.theta

    @property
    def q(self) -> Array:
        """Specific humidity [kg/kg]."""
        return self.mixed.q

    @property
    def co2(self) -> Array:
        """CO2 concentration [ppmv]."""
        return self.mixed.co2

    @property
    def surf_pressure(self) -> Array:
        """Surface pressure [Pa]."""
        return self.mixed.surf_pressure

    @property
    def u(self) -> Array:
        """Zonal wind speed [m/s]."""
        return self.mixed.u

    @property
    def v(self) -> Array:
        """Meridional wind speed [m/s]."""
        return self.mixed.v

    @property
    def ra(self) -> Array:
        """Aerodynamic resistance [s/m]."""
        return self.surface.ra

    @property
    def thetasurf(self) -> Array:
        """Surface potential temperature [K]."""
        return self.surface.thetasurf

    @property
    def h_abl(self) -> Array:
        """Boundary layer height [m]."""
        return self.mixed.h_abl

    @property
    def ustar(self) -> Array:
        """Friction velocity [m/s]."""
        return self.surface.ustar

    @property
    def uw(self) -> Array:
        """Zonal momentum flux [m²/s²]."""
        return self.surface.uw

    @property
    def vw(self) -> Array:
        """Meridional momentum flux [m²/s²]."""
        return self.surface.vw

    @property
    def wstar(self) -> Array:
        """Convective velocity scale [m/s]."""
        return self.mixed.wstar

    @property
    def thetav(self) -> Array:
        """Mixed-layer virtual potential temperature [K]."""
        return self.mixed.thetav

    @property
    def top_T(self) -> Array:
        """Temperature at top of mixed layer [K]."""
        return self.mixed.top_T

    @property
    def top_p(self) -> Array:
        """Pressure at top of mixed layer [Pa]."""
        return self.mixed.top_p

    @property
    def cc_mf(self) -> Array:
        """Cloud core mass flux [kg/kg/s]."""
        return self.clouds.cc_mf

    @property
    def cc_qf(self) -> Array:
        """Cloud core moisture flux [kg/kg/s]."""
        return self.clouds.cc_qf

    @property
    def wCO2M(self) -> Array:
        """Cloud core CO2 mass flux [mgC/m²/s]."""
        return self.clouds.wCO2M

    @property
    def cc_frac(self) -> Array:
        """Cloud core fraction [-]."""
        return self.clouds.cc_frac

    @property
    def wthetav(self) -> Array:
        """Virtual potential temperature flux at surface [K m/s]."""
        return self.mixed.wthetav

    @property
    def wqe(self) -> Array:
        """Entrainment moisture flux [kg/kg m/s]."""
        return self.mixed.wqe

    @property
    def dq(self) -> Array:
        """Specific humidity jump at h [kg/kg]."""
        return self.mixed.dq

    @property
    def dz_h(self) -> Array:
        """Transition layer thickness [m]."""
        return self.mixed.dz_h

    @property
    def deltaCO2(self) -> Array:
        """CO2 jump at h [ppm]."""
        return self.mixed.deltaCO2

    @property
    def wCO2e(self) -> Array:
        """Entrainment CO2 flux [mgC/m²/s]."""
        return self.mixed.wCO2e


# in this case we are sure that the coupled state being used here
# has the atmos as the day-only atmos
StateAlias = AbstractCoupledState[
    RadT,
    LandT,
    DayOnlyAtmosphereState[SurfT, MixedT, CloudT],
]


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

    def init_state(
        self, surface: SurfT, mixed: MixedT, clouds: CloudT
    ) -> DayOnlyAtmosphereState[SurfT, MixedT, CloudT]:
        """Initialize the model state.

        Args:
            surface: The initial surface layer state.
            mixed: The initial mixed layer state.
            clouds: The initial cloud state.

        Returns:
            The initial atmosphere state.
        """
        return DayOnlyAtmosphereState(surface=surface, mixed=mixed, clouds=clouds)

    def run(
        self,
        state: StateAlias,
    ) -> DayOnlyAtmosphereState:
        sl_state = self.surface_layer.run(state)
        atmostate = replace(state.atmos, surface=sl_state)
        state = state.replace(atmos=atmostate)
        cl_state = self.clouds.run(state)
        atmostate = replace(atmostate, clouds=cl_state)
        state = state.replace(atmos=atmostate)
        ml_state = self.mixed_layer.run(state)
        atmostate = replace(atmostate, mixed=ml_state)
        return atmostate

    def statistics(self, state: StateAlias, t: int) -> DayOnlyAtmosphereState:
        """Update statistics."""
        ml_state = self.mixed_layer.statistics(state, t)
        return state.atmos.replace(mixed=ml_state)

    def warmup(
        self,
        radmodel: AbstractRadiationModel,
        landmodel: AbstractLandModel,
        state: StateAlias,
        t: int,
        dt: float,
        tstart: float,
    ) -> StateAlias:
        """Warmup the atmos by running it for a few timesteps."""
        state = state.replace(
            atmos=self.statistics(state, t),
        )
        state = state.replace(rad=radmodel.run(state, t, dt, tstart))
        for _ in range(10):
            sl_state = self.surface_layer.run(state)
            atmostate = replace(state.atmos, surface=sl_state)
            state = state.replace(atmos=atmostate)
        landstate = landmodel.run(state)
        state = state.replace(land=landstate)

        # this is if clause is ok because it's outise the scan!
        if not isinstance(self.clouds, NoCloudModel):
            ml_state = self.mixed_layer.run(state)
            atmostate = replace(state.atmos, mixed=ml_state)
            state = state.replace(atmos=atmostate)
            cl_state = self.clouds.run(state)
            atmostate = replace(state.atmos, clouds=cl_state)
            state = state.replace(atmos=atmostate)
        ml_state = self.mixed_layer.run(state)
        atmostate = replace(state.atmos, mixed=ml_state)
        state = state.replace(atmos=atmostate)
        return state

    def integrate(
        self, state: DayOnlyAtmosphereState, dt: float
    ) -> DayOnlyAtmosphereState:
        ml_state = self.mixed_layer.integrate(state.mixed, dt)
        return replace(state, mixed=ml_state)
