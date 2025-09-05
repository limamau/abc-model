from ..components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
)
from ..utils import PhysicalConstants


class ConstantRadiationModel(AbstractRadiationModel):
    """Constant net radiation model.

    Simple radiation model that maintains fixed net radiation values without
    any atmospheric or solar calculations.

    **Processes:**
    1. Maintains constant net radiation.

    Arguments
    ----------
    * ``net_rad``: net surface radiation [W/m²].
    * ``dFz``: cloud top radiative divergence [W/m²].

    Updates
    --------
    * No updates - ``net_rad`` remains constant.
    """

    def __init__(
        self,
        net_rad: float,
        dFz: float,
    ):
        self.net_rad = net_rad
        self.dFz = dFz

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
