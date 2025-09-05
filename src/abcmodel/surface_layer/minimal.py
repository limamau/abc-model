import numpy as np

from ..components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants


class MinimalSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Minimal surface layer model with constant friction velocity.

    A surface layer model that uses a fixed friction velocity to calculate
    momentum fluxes and aerodynamic resistance without atmospheric stability effects.

    **Running processes:**

    Calculate momentum fluxes based on wind components and friction velocity.

    **Auxiliary processes:**

    Compute aerodynamic resistance using effective wind speed.

    Arguments
    ----------
    - ``ustar``: surface friction velocity [m/s].

    Updates
    --------
    - ``uw``: u-component momentum flux [m²/s²].
    - ``vw``: v-component momentum flux [m²/s²].
    - ``ra``: aerodynamic resistance [s/m] (aux).
    """

    def __init__(self, ustar: float):
        # surface friction velocity [m s-1]
        self.ustar = ustar

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
        self.ra = ueff / max(1.0e-3, self.ustar) ** 2.0
