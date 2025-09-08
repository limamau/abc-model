import numpy as np

from ..diagnostics import AbstractDiagnostics
from ..models import (
    AbstractInitConds,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractParams,
    AbstractRadiationModel,
)
from ..utils import PhysicalConstants


class ConstantRadiationParams(AbstractParams["ConstantRadiationModel"]):
    """Data class for constant radiation model parameters.

    Arguments
    ---------
    - ``dFz``: cloud top radiative divergence [W/m²].
    - ``tstart``: start time of day [hours UTC], range 0 to 24.
    """

    def __init__(self, dFz: float, tstart: float):
        self.dFz = dFz
        self.tstart = tstart


class ConstantRadiationInitConds(AbstractInitConds["ConstantRadiationModel"]):
    """Data class for constant radiation model initial conditions.

    Arguments
    ---------
    - ``net_rad``: net surface radiation [W/m²].
    """

    def __init__(self, net_rad: float):
        self.net_rad = net_rad


class ConstantRadiationDiagnostics(AbstractDiagnostics["ConstantRadiationModel"]):
    """Class for constant radiation model diagnostic variables.

    Variables
    ---------
    - ``net_rad``: net surface radiation [W/m²].
    """

    def post_init(self, tsteps: int):
        self.net_rad = np.zeros(tsteps)

    def store(self, t: int, model: "ConstantRadiationModel"):
        self.net_rad[t] = model.net_rad


class ConstantRadiationModel(AbstractRadiationModel):
    """Constant net radiation model.

    Simple radiation model that maintains fixed net radiation values without
    any atmospheric or solar calculations.

    **Processes:**
    1. Maintains constant net radiation.

    Updates
    --------
    * No updates - ``net_rad`` remains constant.
    """

    def __init__(
        self,
        params: ConstantRadiationParams,
        init_conds: ConstantRadiationInitConds,
        diagnostics: AbstractDiagnostics = ConstantRadiationDiagnostics(),
    ):
        self.dFz = params.dFz
        self.tstart = params.tstart
        self.net_rad: float = init_conds.net_rad
        self.diagnostics = diagnostics

    def run(
        self,
        t: float,
        dt: float,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """No calculations - maintains constant radiation values."""
        pass

    def get_f1(self):
        """No scaling factor for surface processes."""
        return 1.0
