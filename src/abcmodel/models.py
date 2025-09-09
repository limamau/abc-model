from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from .utils import PhysicalConstants

# limamau: how to enforce that every model should take params, init_conds
# and diagnostics as input on the initialization method?
# limamau: it would be also nice to implement something like "check_dependencies"
# in order to verify if the concrete model assigned to one of the 5 components in
# the abc model has all variables needed from other models (minimal variables)...
# limamau: how to enforce that the minimal variables stated here are to be
# implemented by all stantiations of the abstract models described here?


class AbstractModel:
    """Abstract model class to define the interface for all models."""

    diagnostics: "AbstractDiagnostics"

    @abstractmethod
    def init_diagnostics(self, tsteps: int) -> None:
        raise NotImplementedError

    def store(self, t: int):
        self.diagnostics.store(t, self)


MT = TypeVar("MT", bound=AbstractModel)


@dataclass
class AbstractParams(Generic[MT]):
    """Abstract parameters class to define the interface for all models."""

    pass


@dataclass
class AbstractInitConds(Generic[MT]):
    """Abstract initial conditions class to define the interface for all models."""

    pass


class AbstractDiagnostics(Generic[MT]):
    """Abstract diagnostics class to define the interface for all models."""

    def __init__(self):
        pass

    @abstractmethod
    def post_init(self, tsteps: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def store(self, t: int, model: MT) -> None:
        raise NotImplementedError

    def get(self, name: str) -> np.ndarray:
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"Diagnostic variable '{name}' not found")


class AbstractRadiationModel(AbstractModel):
    tstart: float
    dFz: float
    net_rad: float
    # limamau: the following is used by AquaCrop, but not
    # implemeneted by constant radiation...
    in_srad: float

    @abstractmethod
    def run(
        self,
        t: float,
        dt: float,
        const: PhysicalConstants,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_f1(self) -> float:
        raise NotImplementedError


class AbstractLandSurfaceModel(AbstractModel):
    alpha: float
    surf_temp: float
    rs: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError


class AbstractSurfaceLayerModel(AbstractModel):
    ustar: float
    uw: float
    vw: float
    # limamau the following is currently not computed by minimal model,
    # so it will produce an error in land_surface.aquacrop model
    thetasurf: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_ra(self, u: float, v: float, wstar: float) -> float:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel):
    q: float
    dq: float
    esat: float
    e: float
    qsat: float
    qsatsurf: float
    dqsatdT: float
    theta: float
    wtheta: float
    wq: float
    wqe: float
    dtheta: float
    abl_height: float
    surf_pressure: float
    u: float
    v: float
    wstar: float
    thetav: float
    wthetav: float
    q2_h: float
    dz_h: float
    top_p: float
    top_T: float
    co2: float
    top_CO22: float
    wCO2e: float
    wCO2M: float
    dCO2: float
    wCO2: float
    wCO2A: float
    wCO2R: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        clouds: "AbstractCloudModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, t: float, const: PhysicalConstants) -> None:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel):
    cc_frac: float
    cc_mf: float
    cc_qf: float

    @abstractmethod
    def run(self, mixed_layer: "AbstractMixedLayerModel") -> None:
        raise NotImplementedError
