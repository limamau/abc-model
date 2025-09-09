import numpy as np

from ..models import (
    AbstractDiagnostics,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants
from .standard import (
    AbstractStandardLandSurfaceModel,
    StandardLandSurfaceDiagnostics,
    StandardLandSurfaceInitConds,
    StandardLandSurfaceParams,
)


class JarvisStewartParams(StandardLandSurfaceParams):
    """Data class for Jarvis-Stewart model parameters.

    Arguments
    ---------
    - all arguments from StandardLandSurfaceParams.
    """

    pass


class JarvisStewartInitConds(StandardLandSurfaceInitConds):
    """Data class for Jarvis-Stewart model initial conditions.

    Arguments
    ---------
    - all arguments from StandardLandSurfaceInitConds.
    """

    pass


class JarvisStewartDiagnostics(StandardLandSurfaceDiagnostics["JarvisStewartModel"]):
    """Class for Jarvis-Stewart model diagnostics.

    Variables
    ---------
    - all variables from StandardLandSurfaceDiagnostics.
    """

    pass


class JarvisStewartModel(AbstractStandardLandSurfaceModel):
    """Jarvis-Stewart land surface model with empirical surface resistance.

    Implementation of the Jarvis-Stewart approach for calculating surface resistance
    based on environmental stress factors. Uses multiplicative stress functions
    for radiation, soil moisture, vapor pressure deficit, and temperature effects
    on stomatal conductance.

    Processes
    ---------
    1. Inherit all standard land surface processes from parent class.
    2. Calculate surface resistance using four environmental stress factors.
    3. Apply Jarvis-Stewart multiplicative stress function approach.
    4. No CO2 flux calculations (simple implementation).

    Updates
    --------
    - ``rs``: surface resistance for transpiration [s m-1].
    - all updates from ``AbstractStandardLandSurfaceModel``.
    """

    def __init__(
        self,
        params: JarvisStewartParams,
        init_conds: JarvisStewartInitConds,
        diagnostics: AbstractDiagnostics = JarvisStewartDiagnostics(),
    ):
        super().__init__(params, init_conds, diagnostics)

    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Compute surface resistance using Jarvis-Stewart approach.

        Parameters
        ----------
        - ``const``: physical constants (currently unused).
        - ``radiation``: radiation model. Uses ``get_f1()`` method.
        - ``surface_layer``: surface layer model (currently unused).
        - ``mixed_layer``: mixed layer model. Uses ``esat``, ``e``, and ``theta``.

        Updates
        -------
        Updates ``self.rs`` using multiplicative stress factors for radiation (f1),
        soil moisture (f2), vapor pressure deficit (f3), and temperature (f4).
        """
        # calculate surface resistances using Jarvis-Stewart model
        f1 = radiation.get_f1()

        if self.w2 > self.wwilt:
            f2 = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
        else:
            f2 = 1.0e8

        # limit f2 in case w2 > wfc, where f2 < 1
        f2 = max(f2, 1.0)
        f3 = 1.0 / np.exp(-self.gD * (mixed_layer.esat - mixed_layer.e) / 100.0)
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - mixed_layer.theta) ** 2.0)

        self.rs = self.rsmin / self.lai * f1 * f2 * f3 * f4

    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Pass.
        """
        pass
