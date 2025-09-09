import numpy as np

from ..models import (
    AbstractDiagnostics,
    AbstractInitConds,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractParams,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_esat, get_qsat


class MinimalLandSurfaceParams(AbstractParams["MinimalLandSurfaceModel"]):
    """Data class for minimal land surface model parameters.

    Arguments
    ---------
    None.
    """

    def __init__(self):
        pass


class MinimalLandSurfaceInitConds(AbstractInitConds["MinimalLandSurfaceModel"]):
    """Data class for minimal land surface model initial conditions.

    Arguments
    ---------
    - ``alpha``: surface albedo [-], range 0 to 1.
    - ``surf_temp``: surface temperature [K].
    - ``rs``: surface resistance [s m-1].
    """

    def __init__(self, alpha: float, surf_temp: float, rs: float):
        self.alpha = alpha
        self.surf_temp = surf_temp
        self.rs = rs


class MinimalLandSurfaceDiagnostics(AbstractDiagnostics["MinimalLandSurfaceModel"]):
    """Class for minimal land surface model diagnostics.

    Variables
    ---------
    - ``alpha``: surface albedo [-], range 0 to 1.
    - ``surf_temp``: surface temperature [K].
    - ``rs``: surface resistance [s m-1].
    """

    def post_init(self, tsteps: int):
        self.alpha = np.zeros(tsteps)
        self.surf_temp = np.zeros(tsteps)
        self.rs = np.zeros(tsteps)

    def store(self, t: int, model: "MinimalLandSurfaceModel"):
        self.alpha[t] = model.alpha
        self.surf_temp[t] = model.surf_temp
        self.rs[t] = model.rs


class MinimalLandSurfaceModel(AbstractLandSurfaceModel):
    """Minimal land surface model with fixed surface properties."""

    def __init__(
        self,
        params: MinimalLandSurfaceParams,
        init_conds: MinimalLandSurfaceInitConds,
        diagnostics: AbstractDiagnostics = MinimalLandSurfaceDiagnostics(),
    ):
        self.alpha = init_conds.alpha
        self.surf_temp = init_conds.surf_temp
        self.rs = init_conds.rs
        self.diagnostics = diagnostics

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Execute land surface model calculations.

        Parameters
        ----------
        - ``const``: physical constants (currently unused).
        - ``radiation``: radiation model (currently unused).
        - ``surface_layer``: surface layer model. Uses ``compute_ra`` method.
        - ``mixed_layer``: mixed layer model. Uses ``u``, ``v``, ``wstar``, ``theta``,
          ``surf_pressure`` and ``q``.

        Updates
        -------
        Updates thermodynamic variables in mixed layer model including saturation
        vapor pressure, saturation specific humidity, and related derivatives.
        Also computes aerodynamic resistance via surface layer model.
        """
        # limamau: the following two blocks are also computed by
        # the standard class - should we refactor some things here?
        # (1) compute aerodynamic resistance
        surface_layer.compute_ra(mixed_layer.u, mixed_layer.v, mixed_layer.wstar)

        # (2) calculate essential thermodynamic variables
        mixed_layer.esat = get_esat(mixed_layer.theta)
        mixed_layer.qsat = get_qsat(mixed_layer.theta, mixed_layer.surf_pressure)
        desatdT = mixed_layer.esat * (
            17.2694 / (mixed_layer.theta - 35.86)
            - 17.2694
            * (mixed_layer.theta - 273.16)
            / (mixed_layer.theta - 35.86) ** 2.0
        )
        mixed_layer.dqsatdT = 0.622 * desatdT / mixed_layer.surf_pressure
        mixed_layer.e = mixed_layer.q * mixed_layer.surf_pressure / 0.622

    def integrate(self, dt: float):
        """
        Integrate model forward in time.
        """
        pass
