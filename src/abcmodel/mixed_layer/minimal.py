from ..models import (
    AbstractCloudModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    """Minimal mixed layer model with constant properties.

    Simple mixed layer model that maintains fixed atmospheric properties without
    temporal evolution. Used for idealized simulations and testing.

    **Processes:**
    1. Initialize all mixed layer variables with constant values.
    2. No temporal integration - all values remain fixed.

    Arguments
    ----------
    * ``abl_height``: initial ABL height [m].
    * ``surf_pressure``: surface pressure [Pa].
    * ``theta``: initial mixed-layer potential temperature [K].
    * ``dtheta``: initial temperature jump at h [K].
    * ``wtheta``: surface kinematic heat flux [K m/s].
    * ``q``: initial mixed-layer specific humidity [kg/kg].
    * ``dq``: initial specific humidity jump at h [kg/kg].
    * ``wq``: surface kinematic moisture flux [kg/kg m/s].
    * ``co2``: initial mixed-layer CO2 [ppm].
    * ``dCO2``: initial CO2 jump at h [ppm].
    * ``wCO2``: surface kinematic CO2 flux [mgC/m²/s].
    * ``u``: initial mixed-layer u-wind speed [m/s].
    * ``v``: initial mixed-layer v-wind speed [m/s].
    * ``dz_h``: transition layer thickness [-].

    Updates
    --------
    * No updates - all values remain constant.
    """

    def __init__(
        self,
        abl_height: float,
        surf_pressure: float,
        theta: float,
        dtheta: float,
        wtheta: float,
        q: float,
        dq: float,
        wq: float,
        co2: float,
        dCO2: float,
        wCO2: float,
        u: float,
        v: float,
        dz_h: float,
    ):
        # initial ABL height [m]
        self.abl_height = abl_height
        # surface pressure [Pa]
        self.surf_pressure = surf_pressure
        # initial mixed-layer potential temperature [K]
        self.theta = theta
        # # initial temperature jump at h [K]
        self.dtheta = dtheta
        # surface kinematic heat flux [K m s-1]
        self.wtheta = wtheta
        # convective velocity scale [m s-1] (small in minimal)
        self.wstar = 1e-6
        # initial mixed-layer specific humidity [kg kg-1]
        self.q = q
        # initial specific humidity jump at h [kg kg-1]
        self.dq = dq
        # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wq = wq
        # transition layer thickness [-]
        self.dz_h = dz_h
        # conversion factor mgC m-2 s-1 to ppm m s-1
        const = PhysicalConstants()
        fac = const.mair / (const.rho * const.mco2)
        # initial mixed-layer CO2 [ppm]
        self.co2 = co2
        # initial CO2 jump at h [ppm]
        self.dCO2 = dCO2
        # surface kinematic CO2 flux [ppm m s-1]
        self.wCO2 = wCO2 * fac
        # surface assimulation CO2 flux [ppm m s-1]
        self.wCO2A = 0.0
        # surface respiration CO2 flux [ppm m s-1]
        self.wCO2R = 0.0
        # CO2 mass flux [ppm m s-1]
        self.wCO2M = 0.0
        # initial mixed-layer u-wind speed [m s-1]
        self.u = u
        # initial mixed-layer v-wind speed [m s-1]
        self.v = v
        # entrainment moisture flux [kg kg-1 m s-1]
        self.wqe = 0.0
        # entrainment CO2 flux [ppm m s-1]
        self.wCO2e = 0.0

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        clouds: AbstractCloudModel,
    ):
        """No calculations."""
        pass

    def integrate(self, dt: float):
        """No integration."""
        pass
