from ..models import (
    AbstractCloudModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


class BulkMixedLayerModel(AbstractStandardStatsModel):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    **Processes:**
    1. Calculate large-scale vertical motions and compensating effects.
    2. Determine convective velocity scale and entrainment parameters.
    3. Compute all tendency terms for mixed layer variables.
    4. Integrate prognostic equations forward in time.

    Arguments
    ----------
    * ``sw_ml``: mixed-layer model switch.
    * ``sw_shearwe``: shear growth mixed-layer switch.
    * ``sw_fixft``: fix the free-troposphere switch.
    * ``abl_height``: initial ABL height [m].
    * ``surf_pressure``: surface pressure [Pa].
    * ``divU``: horizontal large-scale divergence of wind [s⁻¹].
    * ``coriolis_param``: Coriolis parameter [s⁻¹].
    * ``theta``: initial mixed-layer potential temperature [K].
    * ``dtheta``: initial temperature jump at h [K].
    * ``gammatheta``: free atmosphere potential temperature lapse rate [K/m].
    * ``advtheta``: advection of heat [K/s].
    * ``beta``: entrainment ratio for virtual heat [-].
    * ``wtheta``: surface kinematic heat flux [K m/s].
    * ``q``: initial mixed-layer specific humidity [kg/kg].
    * ``dq``: initial specific humidity jump at h [kg/kg].
    * ``gammaq``: free atmosphere specific humidity lapse rate [kg/kg/m].
    * ``advq``: advection of moisture [kg/kg/s].
    * ``wq``: surface kinematic moisture flux [kg/kg m/s].
    * ``co2``: initial mixed-layer CO2 [ppm].
    * ``dCO2``: initial CO2 jump at h [ppm].
    * ``gammaCO2``: free atmosphere CO2 lapse rate [ppm/m].
    * ``advCO2``: advection of CO2 [ppm/s].
    * ``wCO2``: surface kinematic CO2 flux [mgC/m²/s].
    * ``sw_wind``: prognostic wind switch.
    * ``u``: initial mixed-layer u-wind speed [m/s].
    * ``du``: initial u-wind jump at h [m/s].
    * ``gammau``: free atmosphere u-wind speed lapse rate [s⁻¹].
    * ``advu``: advection of u-wind [m/s²].
    * ``v``: initial mixed-layer v-wind speed [m/s].
    * ``dv``: initial v-wind jump at h [m/s].
    * ``gammav``: free atmosphere v-wind speed lapse rate [s⁻¹].
    * ``advv``: advection of v-wind [m/s²].
    * ``dz_h``: transition layer thickness [m].

    Updates
    --------
    * All mixed layer prognostic variables and their tendencies.
    * Entrainment parameters and vertical motion components.
    """

    # entrainment parameters:
    # large-scale vertical velocity [m s-1]
    ws: float
    # mixed-layer growth due to radiative divergence [m s-1]
    wf: float
    # mixed-layer top relavtive humidity [-]
    top_rh: float
    # lifting condensation level [m]
    lcl: float
    # virtual temperatures and fluxes:
    # initial virtual temperature jump at h [K]
    dthetav: float
    # entrainment kinematic heat flux [K m s-1]
    wthetae: float
    # entrainment kinematic virtual heat flux [K m s-1]
    wthetave: float
    # tendencies:
    # tendency of CBL [m s-1]
    htend: float
    # tendency of mixed-layer potential temperature [K s-1]
    thetatend: float
    # tendency of potential temperature jump at h [K s-1]
    dthetatend: float
    # tendency of mixed-layer specific humidity [kg kg-1 s-1]
    qtend: float
    # tendency of specific humidity jump at h [kg kg-1 s-1]
    dqtend: float
    # tendency of CO2 humidity [ppm]
    co2tend: float
    # tendency of CO2 jump at h [ppm s-1]
    dCO2tend: float
    # tendency of u-wind [m s-1 s-1]
    utend: float
    # tendency of u-wind jump at h [m s-1 s-1]
    dutend: float
    # tendency of v-wind [m s-1 s-1]
    vtend: float
    # tendency of v-wind jump at h [m s-1 s-1]
    dvtend: float
    # tendency of transition layer thickness [m s-1]
    dztend: float

    def __init__(
        self,
        sw_ml: bool,
        sw_shearwe: bool,
        sw_fixft: bool,
        abl_height: float,
        surf_pressure: float,
        divU: float,
        coriolis_param: float,
        theta: float,
        dtheta: float,
        gammatheta: float,
        advtheta: float,
        beta: float,
        wtheta: float,
        q: float,
        dq: float,
        gammaq: float,
        advq: float,
        wq: float,
        co2: float,
        dCO2: float,
        gammaCO2: float,
        advCO2: float,
        wCO2: float,
        sw_wind: bool,
        u: float,
        du: float,
        gammau: float,
        advu: float,
        v: float,
        dv: float,
        gammav: float,
        advv: float,
        dz_h: float,
    ):
        # mixed layer switches:
        # mixed-layer model switch
        self.sw_ml = sw_ml
        # shear growth mixed-layer switch
        self.sw_shearwe = sw_shearwe
        # fix the free-troposphere switch
        self.sw_fixft = sw_fixft
        # large scale parameters:
        # initial ABL height [m]
        self.abl_height = abl_height
        # surface pressure [Pa]
        self.surf_pressure = surf_pressure
        # horizontal large-scale divergence of wind [s-1]
        self.divU = divU
        # Coriolis parameter [m s-1]
        self.coriolis_param = coriolis_param
        # temperature parameters:
        # initial mixed-layer potential temperature [K]
        self.theta = theta
        # initial temperature jump at h [K]
        self.dtheta = dtheta
        # free atmosphere potential temperature lapse rate [K m-1]
        self.gammatheta = gammatheta
        # advection of heat [K s-1]
        self.advtheta = advtheta
        # entrainment ratio for virtual heat [-]
        self.beta = beta
        # surface kinematic heat flux [K m s-1]
        self.wtheta = wtheta
        # entrainment parameters:
        # convective velocity scale [m s-1]
        self.wstar = 0.0
        # entrainment velocity [m s-1]
        self.we = -1.0
        # 5. moisture parameters:
        # initial mixed-layer specific humidity [kg kg-1]
        self.q = q
        # initial specific humidity jump at h [kg kg-1]
        self.dq = dq
        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.gammaq = gammaq
        # advection of moisture [kg kg-1 s-1]
        self.advq = advq
        # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wq = wq
        # 8. mixed-layer top variables
        # transition layer thickness [-]
        self.dz_h = dz_h
        # CO2:
        # limamau: I had to initiate another const here
        # could be replaced by a post_init though...
        const = PhysicalConstants()
        # conversion factor mgC m-2 s-1 to ppm m s-1
        fac = const.mair / (const.rho * const.mco2)
        # initial mixed-layer CO2 [ppm]
        self.co2 = co2
        # initial CO2 jump at h [ppm]
        self.dCO2 = dCO2
        # free atmosphere CO2 lapse rate [ppm m-1]
        self.gammaco2 = gammaCO2
        # advection of CO2 [ppm s-1]
        self.advCO2 = advCO2
        # surface kinematic CO2 flux [ppm m s-1]
        self.wCO2 = wCO2 * fac
        # surface assimulation CO2 flux [ppm m s-1]
        self.wCO2A = 0.0
        # surface respiration CO2 flux [ppm m s-1]
        self.wCO2R = 0.0
        # CO2 mass flux [ppm m s-1]
        self.wCO2M = 0.0
        # 11. wind parameters
        # prognostic wind switch
        self.sw_wind = sw_wind
        # initial mixed-layer u-wind speed [m s-1]
        self.u = u
        # initial u-wind jump at h [m s-1]
        self.du = du
        # free atmosphere u-wind speed lapse rate [s-1]
        self.gammau = gammau
        # advection of u-wind [m s-2]
        self.advu = advu
        # initial mixed-layer v-wind speed [m s-1]
        self.v = v
        # initial v-wind jump at h [m s-1]
        self.dv = dv
        # free atmosphere v-wind speed lapse rate [s-1]
        self.gammav = gammav
        # advection of v-wind [m s-2]
        self.advv = advv

    def _calculate_vertical_motions(self, dFz: float, rho: float, cp: float):
        """Calculate large-scale subsidence and radiative divergence effects."""
        # calculate large-scale vertical velocity (subsidence)
        self.ws = -self.divU * self.abl_height

        # calculate mixed-layer growth due to cloud top radiative divergence
        radiative_denominator = rho * cp * self.dtheta
        self.wf = dFz / radiative_denominator

    def _calculate_free_troposphere_compensation(self):
        """Calculate compensation terms to fix free troposphere values."""
        if self.sw_fixft:
            w_th_ft = self.gammatheta * self.ws
            w_q_ft = self.gammaq * self.ws
            w_CO2_ft = self.gammaco2 * self.ws
        else:
            w_th_ft = 0.0
            w_q_ft = 0.0
            w_CO2_ft = 0.0
        return w_th_ft, w_q_ft, w_CO2_ft

    def _calculate_convective_velocity_scale(self, g: float):
        """Calculate convective velocity scale and entrainment parameters."""
        if self.wthetav > 0.0:
            buoyancy_term = g * self.abl_height * self.wthetav / self.thetav
            self.wstar = buoyancy_term ** (1.0 / 3.0)
        else:
            self.wstar = 1e-6

        # virtual heat entrainment flux
        self.wthetave = -self.beta * self.wthetav

    def _calculate_entrainment_velocity(self, ustar: float, g: float):
        """Calculate entrainment velocity with optional shear effects."""
        if self.sw_shearwe:
            shear_term = 5.0 * ustar**3.0 * self.thetav / (g * self.abl_height)
            numerator = -self.wthetave + shear_term
            self.we = numerator / self.dthetav
        else:
            self.we = -self.wthetave / self.dthetav

        # don't allow boundary layer shrinking if wtheta < 0
        if self.we < 0:
            self.we = 0.0

    def _calculate_entrainment_fluxes(self):
        """Calculate all entrainment fluxes."""
        self.wthetae = -self.we * self.dtheta
        self.wqe = -self.we * self.dq
        self.wCO2e = -self.we * self.dCO2

    def _calculate_mixed_layer_tendencies(
        self, cc_mf: float, cc_qf: float, w_th_ft: float, w_q_ft: float, w_CO2_ft: float
    ):
        """Calculate tendency terms for mixed layer variables."""
        # boundary layer height tendency
        self.htend = self.we + self.ws + self.wf - cc_mf

        # mixed layer scalar tendencies
        surface_heat_flux = (self.wtheta - self.wthetae) / self.abl_height
        self.thetatend = surface_heat_flux + self.advtheta

        surface_moisture_flux = (self.wq - self.wqe - cc_qf) / self.abl_height
        self.qtend = surface_moisture_flux + self.advq

        surface_co2_flux_term = (self.wCO2 - self.wCO2e - self.wCO2M) / self.abl_height
        self.co2tend = surface_co2_flux_term + self.advCO2

        # jump tendencies at boundary layer top
        # (entrainment growth term)
        egrowth = self.we + self.wf - cc_mf

        self.dthetatend = self.gammatheta * egrowth - self.thetatend + w_th_ft
        self.dqtend = self.gammaq * egrowth - self.qtend + w_q_ft
        self.dCO2tend = self.gammaco2 * egrowth - self.co2tend + w_CO2_ft

    def _calculate_wind_tendencies(self, uw: float, vw: float, cc_mf: float):
        """Calculate wind tendency terms if wind is prognostic."""
        # assume u + du = ug, so ug - u = du
        if self.sw_wind:
            coriolis_term_u = -self.coriolis_param * self.dv
            momentum_flux_term_u = (uw + self.we * self.du) / self.abl_height
            self.utend = coriolis_term_u + momentum_flux_term_u + self.advu

            coriolis_term_v = self.coriolis_param * self.du
            momentum_flux_term_v = (vw + self.we * self.dv) / self.abl_height
            self.vtend = coriolis_term_v + momentum_flux_term_v + self.advv

            entrainment_growth_term = self.we + self.wf - cc_mf
            self.dutend = self.gammau * entrainment_growth_term - self.utend
            self.dvtend = self.gammav * entrainment_growth_term - self.vtend

    def _calculate_transition_layer_tendency(self, cc_frac: float):
        """Calculate transition layer thickness tendency."""
        lcl_distance = self.lcl - self.abl_height

        if cc_frac > 0 or lcl_distance < 300:
            target_thickness = lcl_distance - self.dz_h
            self.dztend = target_thickness / 7200.0
        else:
            self.dztend = 0.0

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        clouds: AbstractCloudModel,
    ):
        """
        Calculate mixed layer tendencies and update diagnostic variables.

        Parameters
        ----------
        * ``const``: physical constants. Uses ``g``, ``rho``, and ``cp``.
        * ``radiation``: radiation model. Uses ``dFz``.
        * ``surface_layer``: surface layer model. Uses ``ustar``, ``uw``, and ``vw``.
        * ``clouds``: cloud model. Uses ``cc_frac``, ``cc_mf``, and ``cc_qf``.

        Updates
        -------
        Updates all tendency terms and diagnostic variables for the mixed layer
        evolution including entrainment, subsidence, and cloud effects.
        """
        self._calculate_vertical_motions(radiation.dFz, const.rho, const.cp)
        w_th_ft, w_q_ft, w_CO2_ft = self._calculate_free_troposphere_compensation()
        self._calculate_convective_velocity_scale(const.g)
        self._calculate_entrainment_velocity(surface_layer.ustar, const.g)
        self._calculate_entrainment_fluxes()
        self._calculate_mixed_layer_tendencies(
            clouds.cc_mf,
            clouds.cc_qf,
            w_th_ft,
            w_q_ft,
            w_CO2_ft,
        )
        self._calculate_wind_tendencies(
            surface_layer.uw,
            surface_layer.vw,
            clouds.cc_mf,
        )
        self._calculate_transition_layer_tendency(clouds.cc_frac)

    def integrate(self, dt: float):
        """Integrate mixed layer forward in time."""
        self.abl_height += dt * self.htend
        self.theta += dt * self.thetatend
        self.dtheta += dt * self.dthetatend
        self.q += dt * self.qtend
        self.dq += dt * self.dqtend
        self.co2 += dt * self.co2tend
        self.dCO2 += dt * self.dCO2tend
        self.dz_h += dt * self.dztend

        # limit dz to minimal value
        dz0 = 50
        if self.dz_h < dz0:
            self.dz_h = dz0

        if self.sw_wind:
            self.u += dt * self.utend
            self.du += dt * self.dutend
            self.v += dt * self.vtend
            self.dv += dt * self.dvtend
