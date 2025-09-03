# limamau: this module contains the abstract classes for all main components of the ABC
# model. Another architecture possibility is to define each abstract class in their own
# files and extract kwargs from each run method - the coupler would manage integration...
# I'm leaving this idea for the furure in case we need/decide to go for it.
# limamau: in the future, it will be good to define parameter and initial conditions classes
# to simplifiy initializations. We should also somehow check whether all parameters of a component
# are being computed from one of the other components, otherwise throw a coupling error. I think
# bringing initalization out of the abstract classes is a good step in this direction.

from abc import abstractmethod

from .utils import PhysicalConstants


class AbstractRadiationModel:
    # require by minimal:
    # net radiation [W m-2]
    net_rad: float
    # cloud top radiative divergence [W m-2]
    dFz: float
    # used to output:
    # time of the day [h UTC]
    tstart: float
    # incoming short wave radiation [W m-2]
    in_srad: float
    # outgoing short wave radiation [W m-2]
    out_srad: float
    # incoming long wave radiation [W m-2]
    in_lrad: float
    # outgoing long wave radiation [W m-2]
    out_lrad: float

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
        pass


class AbstractLandSurfaceModel:
    # required by minimal:
    # surface albedo [-]
    alpha: float
    # initial surface temperature [K]
    surf_temp: float
    # used to output:
    # resistance transpiration [s m-1]
    rs: float
    # sensible heat flux [W m-2]
    hf: float
    # evapotranspiration [W m-2]
    le: float
    # open water evaporation [W m-2]
    le_liq: float
    # transpiration [W m-2]
    le_veg: float
    # soil evaporation [W m-2]
    le_soil: float
    # potential evaporation [W m-2]
    le_pot: float
    # reference evaporation at rs = rsmin / LAI [W m-2]
    le_ref: float
    # ground heat flux [W m-2]
    gf: float

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


class AbstractSurfaceLayerModel:
    # required by minimal:
    # surface friction velocity [m s-1]
    ustar: float
    # used to output:
    # surface momentum flux u [m2 s-2]
    uw: float
    # surface momentum flux v [m2 s-2]
    vw: float
    # 2m temperature [K]
    temp_2m: float
    # 2m specific humidity [kg kg-1]
    q2m: float
    # 2m u-wind [m s-1]
    u2m: float
    # 2m v-wind [m s-1]
    v2m: float
    # 2m vapor pressure [Pa]
    e2m: float
    # 2m saturated vapor pressure [Pa]
    esat2m: float
    # surface potential temperature [K]
    thetasurf: float
    # surface virtual potential temperature [K]
    thetavsurf: float
    # surface specific humidity [g kg-1]
    qsurf: float
    # drag coefficient for momentum [-]
    drag_m: float
    # drag coefficient for scalars [-]
    drag_s: float
    # Obukhov length [m]
    obukhov_length: float
    # bulk Richardson number [-]
    rib_number: float
    # aerodynamic resistance [s m-1]
    ra: float

    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        land_surface: "AbstractLandSurfaceModel",
        mixed_layer: "AbstractMixedLayerModel",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_ra(self, u: float, v: float, wstar: float) -> None:
        raise NotImplementedError


class AbstractMixedLayerModel:
    @abstractmethod
    def run(
        self,
        const: PhysicalConstants,
        radiation: "AbstractRadiationModel",
        surface_layer: "AbstractSurfaceLayerModel",
        clouds: "AbstractCloudModel",
    ):
        pass

    @abstractmethod
    def integrate(self, dt: float) -> None:
        raise NotImplementedError


class AbstractCloudModel:
    # required by no-clouds:
    # cloud core fraction [-]
    cc_frac: float
    # cloud core mass flux [m s-1]
    cc_mf: float
    # cloud core moisture flux [kg kg-1 m s-1]
    cc_qf: float

    @abstractmethod
    def run(self, mixed_layer: "AbstractMixedLayerModel") -> None:
        raise NotImplementedError
