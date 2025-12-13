"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod
from typing import Generic, TypeVar

from jax import Array
from simple_pytree import Pytree

from .utils import PhysicalConstants


class AbstractState(Pytree):
    """Abstract state class to define the interface for all states."""


class AbstractRadiationState(AbstractState):
    """Abstract radiation state."""

    net_rad: Array
    """Net surface radiation [W m-2]."""
    in_srad: Array
    """Incoming solar radiation [W m-2]."""
    out_srad: Array
    """Outgoing solar radiation [W m-2]."""
    in_lrad: Array
    """Incoming longwave radiation [W m-2]."""
    out_lrad: Array
    """Outgoing longwave radiation [W m-2]."""


class AbstractLandState(AbstractState):
    """Abstract land state."""

    alpha: Array
    """surface albedo [-], range 0 to 1."""
    surf_temp: Array
    """Surface temperature [K]."""
    rs: Array
    """Surface resistance [s m-1]."""
    wg: Array
    """No moisture content in the root zone [m3 m-3]."""
    wl: Array
    """No water content in the canopy [m]."""

    esat: Array
    """Saturation vapor pressure [Pa]."""
    qsat: Array
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array
    """Vapor pressure [Pa]."""
    qsatsurf: Array
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array
    """Kinematic heat flux [K m/s]."""
    wq: Array
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


class AbstractAtmosphereState(AbstractState):
    """Abstract atmosphere state."""


RadT = TypeVar("RadT", bound=AbstractRadiationState)
LandT = TypeVar("LandT", bound=AbstractLandState)
AtmosT = TypeVar("AtmosT", bound=AbstractAtmosphereState)


class AbstractCoupledState(AbstractState, Generic[RadT, LandT, AtmosT]):
    """Abstract coupled state, generic over radiation, land and atmosphere types."""

    rad: RadT
    land: LandT
    atmos: AtmosT

    @property
    def net_rad(self) -> Array:
        """Net surface radiation [W m-2]."""
        return self.rad.net_rad

    @property
    def in_srad(self) -> Array:
        """Incoming shortwave radiation [W m-2]."""
        return self.rad.in_srad


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel, Generic[RadT]):
    """Abstract radiation model class to define the interface for all radiation models."""

    tstart: Array
    """Start time of the model."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        dt: float,
        const: PhysicalConstants,
    ) -> RadT:
        raise NotImplementedError


class AbstractLandModel(AbstractModel, Generic[LandT]):
    """Abstract land model class to define the interface for all land models."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        const: PhysicalConstants,
    ) -> LandT:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: LandT, dt: float) -> LandT:
        raise NotImplementedError


class AbstractAtmosphereModel(AbstractModel, Generic[AtmosT]):
    """Abstract atmosphere model class to define the interface for all atmosphere models."""

    @abstractmethod
    def warmup(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        const: PhysicalConstants,
        land: AbstractLandModel[LandT],
    ) -> AbstractCoupledState[RadT, LandT, AtmosT]:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        const: PhysicalConstants,
    ) -> AtmosT:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self,
        state: AbstractCoupledState[RadT, LandT, AtmosT],
        t: int,
        const: PhysicalConstants,
    ) -> AbstractCoupledState[RadT, LandT, AtmosT]:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AtmosT, dt: float) -> AtmosT:
        raise NotImplementedError
