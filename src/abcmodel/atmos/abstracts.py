"""Abstract classes for atmos sub-modules."""

from abc import abstractmethod
from typing import Generic, TypeVar

from jax import Array

from ..abstracts import AbstractCoupledState, AbstractModel, AbstractState


class AbstractSurfaceLayerState(AbstractState):
    """Abstract surface layer state."""

    ra: Array
    """Aerodynamic resistance [s/m]."""
    thetasurf: Array
    """Surface potential temperature [K]."""
    ustar: Array
    """Friction velocity [m/s]."""
    uw: Array
    """Zonal momentum flux [m²/s²]."""
    vw: Array
    """Meridional momentum flux [m²/s²]."""


class AbstractMixedLayerState(AbstractState):
    """Abstract mixed layer state."""

    h_abl: Array
    """Atmospheric boundary layer (ABL) height [m]."""
    theta: Array
    """Mixed-layer potential temperature [K]."""
    q: Array
    """Mixed-layer specific humidity [kg/kg]."""
    co2: Array
    """Mixed-layer CO2 [ppm]."""
    u: Array
    """Mixed-layer u-wind speed [m/s]."""
    v: Array
    """Mixed-layer v-wind speed [m/s]."""
    surf_pressure: Array
    """Surface pressure [Pa]."""
    wstar: Array
    """Convective velocity scale [m/s]."""
    thetav: Array
    """Mixed-layer virtual potential temperature [K]."""
    top_T: Array
    """Temperature at top of mixed layer [K]."""
    top_p: Array
    """Pressure at top of mixed layer [Pa]."""
    wthetav: Array
    """Virtual potential temperature flux at surface [K m/s]."""
    wqe: Array
    """Entrainment moisture flux [kg/kg m/s]."""
    dq: Array
    """Specific humidity jump at h [kg/kg]."""
    dz_h: Array
    """Transition layer thickness [m]."""
    deltaCO2: Array
    """CO2 jump at h [ppm]."""
    wCO2e: Array
    """Entrainment CO2 flux [mgC/m²/s]."""
    wtheta: Array
    """Surface kinematic heat flux [K m/s]."""
    wq: Array
    """Surface kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array
    """Surface kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


class AbstractCloudState(AbstractState):
    """Abstract cloud state."""

    cc_mf: Array
    """Cloud core mass flux [kg/kg/s]."""
    cc_qf: Array
    """Cloud core moisture flux [kg/kg/s]."""
    wCO2M: Array
    """Cloud core CO2 mass flux [mgC/m²/s]."""
    cc_frac: Array
    """Cloud core fraction [-]."""


SurfT = TypeVar("SurfT", bound=AbstractSurfaceLayerState)
MixedT = TypeVar("MixedT", bound=AbstractMixedLayerState)
CloudT = TypeVar("CloudT", bound=AbstractCloudState)


class AbstractSurfaceLayerModel(AbstractModel, Generic[SurfT]):
    """Abstract surface layer model class to define the interface for all surface layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> SurfT:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel, Generic[MixedT]):
    """Abstract mixed layer model class to define the interface for all mixed layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> MixedT:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, state: AbstractCoupledState, t: int) -> MixedT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: MixedT, dt: float) -> MixedT:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel, Generic[CloudT]):
    """Abstract cloud model class to define the interface for all cloud models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> CloudT:
        raise NotImplementedError
