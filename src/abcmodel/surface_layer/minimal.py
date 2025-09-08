import numpy as np

from ..diagnostics import AbstractDiagnostics
from ..models import (
    AbstractInitConds,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractParams,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants


class MinimalSurfaceLayerParams(AbstractParams["MinimalSurfaceLayerModel"]):
    """Data class for minimal surface layer model parameters.

    It doesn't carry any parameters.
    """

    def __init__(self):
        pass


class MinimalSurfaceLayerInitConds(AbstractInitConds["MinimalSurfaceLayerModel"]):
    """Data class for minimal surface layer model initial conditions.

    Arguments
    ---------
    - ``ustar``: surface friction velocity [m/s].
    """

    def __init__(self, ustar: float):
        self.ustar = ustar


class MinimalSurfaceLayerDiagnostics(AbstractDiagnostics["MinimalSurfaceLayerModel"]):
    """Class for minimal surface layer model diagnostic variables.

    Variables
    -------
    - ``uw``: surface momentum flux u [m2 s-2].
    - ``vw``: surface momentum flux v [m2 s-2].
    - ``ustar``: surface friction velocity [m/s].
    """

    def post_init(self, tsteps: int):
        self.uw = np.zeros(tsteps)
        self.vw = np.zeros(tsteps)
        self.ustar = np.zeros(tsteps)

    def store(self, t: int, model: "MinimalSurfaceLayerModel"):
        self.uw[t] = model.uw
        self.vw[t] = model.vw
        self.ustar[t] = model.ustar


class MinimalSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Minimal surface layer model with constant friction velocity."""

    def __init__(
        self,
        params: MinimalSurfaceLayerParams,
        init_conds: MinimalSurfaceLayerInitConds,
        diagnostics: AbstractDiagnostics = MinimalSurfaceLayerDiagnostics(),
    ):
        self.ustar = init_conds.ustar
        self.diagnostics = diagnostics

    def run(
        self,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """Calculate momentum fluxes from wind components and friction velocity."""
        self.uw = -np.sign(mixed_layer.u) * (
            self.ustar**4.0 / (mixed_layer.v**2.0 / mixed_layer.u**2.0 + 1.0)
        ) ** (0.5)
        self.vw = -np.sign(mixed_layer.v) * (
            self.ustar**4.0 / (mixed_layer.u**2.0 / mixed_layer.v**2.0 + 1.0)
        ) ** (0.5)

    def compute_ra(self, u: float, v: float, wstar: float):
        """Calculate aerodynamic resistance from wind speed and friction velocity."""
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        return ueff / max(1.0e-3, self.ustar) ** 2.0
