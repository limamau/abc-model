from .components import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from .utils import PhysicalConstants, get_qsat


class AbstractStandardStatsModel(AbstractMixedLayerModel):
    def statistics(self, t: float, const: PhysicalConstants):
        # calculate virtual temperatures
        self.thetav = self.theta + 0.61 * self.theta * self.q
        self.wthetav = self.wtheta + 0.61 * self.theta * self.wq
        self.dthetav = (self.theta + self.dtheta) * (
            1.0 + 0.61 * (self.q + self.dq)
        ) - self.theta * (1.0 + 0.61 * self.q)

        # mixed-layer top properties
        self.top_p = self.surf_pressure - const.rho * const.g * self.abl_height
        self.top_T = self.theta - const.g / const.cp * self.abl_height
        self.top_rh = self.q / get_qsat(self.top_T, self.top_p)

        # find lifting condensation level iteratively
        if t == 0:
            self.lcl = self.abl_height
            rhlcl = 0.5
        else:
            rhlcl = 0.9998

        itmax = 30
        it = 0
        while ((rhlcl <= 0.9999) or (rhlcl >= 1.0001)) and it < itmax:
            self.lcl += (1.0 - rhlcl) * 1000.0
            p_lcl = self.surf_pressure - const.rho * const.g * self.lcl
            temp_lcl = self.theta - const.g / const.cp * self.lcl
            rhlcl = self.q / get_qsat(temp_lcl, p_lcl)
            it += 1

        if it == itmax:
            print("LCL calculation not converged!!")
            print("RHlcl = %f, zlcl=%f" % (rhlcl, self.lcl))


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    def __init__(
        self,
        # sw_ml: bool,
        # sw_shearwe: bool,
        # sw_fixft: bool,
        abl_height: float,
        surf_pressure: float,
        # divU: float,
        # coriolis_param: float,
        theta: float,
        dtheta: float,
        # gammatheta: float,
        # advtheta: float,
        # beta: float,
        wtheta: float,
        q: float,
        dq: float,
        # gammaq: float,
        # advq: float,
        wq: float,
        co2: float,
        dCO2: float,
        # gammaCO2: float,
        # advCO2: float,
        wCO2: float,
        # sw_wind: bool,
        u: float,
        # du: float,
        # gammau: float,
        # advu: float,
        v: float,
        # dv: float,
        # gammav: float,
        # advv: float,
        dz_h: float,
    ):
        pass
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
        pass

    def integrate(self, dt: float):
        pass


class StandardMixedLayerModel(AbstractStandardStatsModel):
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

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        clouds: AbstractCloudModel,
    ):
        # calculate large-scale vertical velocity (subsidence)
        self.ws = -self.divU * self.abl_height

        # calculate compensation to fix the free troposphere in case of subsidence
        if self.sw_fixft:
            w_th_ft = self.gammatheta * self.ws
            w_q_ft = self.gammaq * self.ws
            w_CO2_ft = self.gammaco2 * self.ws
        else:
            w_th_ft = 0.0
            w_q_ft = 0.0
            w_CO2_ft = 0.0

        # calculate mixed-layer growth due to cloud top radiative divergence
        self.wf = radiation.dFz / (const.rho * const.cp * self.dtheta)

        # calculate convective velocity scale w*
        if self.wthetav > 0.0:
            self.wstar = ((const.g * self.abl_height * self.wthetav) / self.thetav) ** (
                1.0 / 3.0
            )
        else:
            self.wstar = 1e-6

        # virtual heat entrainment flux
        self.wthetave = -self.beta * self.wthetav

        # compute mixed-layer tendencies
        if self.sw_shearwe:
            self.we = (
                -self.wthetave
                + 5.0
                * surface_layer.ustar**3.0
                * self.thetav
                / (const.g * self.abl_height)
            ) / self.dthetav
        else:
            self.we = -self.wthetave / self.dthetav

        # don't allow boundary layer shrinking if wtheta < 0
        if self.we < 0:
            self.we = 0.0

        # calculate entrainment fluxes
        self.wthetae = -self.we * self.dtheta
        self.wqe = -self.we * self.dq
        self.wCO2e = -self.we * self.dCO2

        self.htend = self.we + self.ws + self.wf - clouds.cc_mf

        self.thetatend = (self.wtheta - self.wthetae) / self.abl_height + self.advtheta
        self.qtend = (self.wq - self.wqe - clouds.cc_qf) / self.abl_height + self.advq
        self.co2tend = (
            self.wCO2 - self.wCO2e - self.wCO2M
        ) / self.abl_height + self.advCO2

        self.dthetatend = (
            self.gammatheta * (self.we + self.wf - clouds.cc_mf)
            - self.thetatend
            + w_th_ft
        )
        self.dqtend = (
            self.gammaq * (self.we + self.wf - clouds.cc_mf) - self.qtend + w_q_ft
        )
        self.dCO2tend = (
            self.gammaco2 * (self.we + self.wf - clouds.cc_mf) - self.co2tend + w_CO2_ft
        )

        # assume u + du = ug, so ug - u = du
        if self.sw_wind:
            self.utend = (
                -self.coriolis_param * self.dv
                + (surface_layer.uw + self.we * self.du) / self.abl_height
                + self.advu
            )
            self.vtend = (
                self.coriolis_param * self.du
                + (surface_layer.vw + self.we * self.dv) / self.abl_height
                + self.advv
            )

            self.dutend = self.gammau * (self.we + self.wf - clouds.cc_mf) - self.utend
            self.dvtend = self.gammav * (self.we + self.wf - clouds.cc_mf) - self.vtend

        # tendency of the transition layer thickness
        if clouds.cc_frac > 0 or self.lcl - self.abl_height < 300:
            self.dztend = ((self.lcl - self.abl_height) - self.dz_h) / 7200.0
        else:
            self.dztend = 0.0

    def integrate(self, dt: float):
        # set values previous time step
        h0 = self.abl_height

        theta0 = self.theta
        dtheta0 = self.dtheta
        q0 = self.q
        dq0 = self.dq
        CO20 = self.co2
        dCO20 = self.dCO2

        u0 = self.u
        du0 = self.du
        v0 = self.v
        dv0 = self.dv

        dz0 = self.dz_h

        # integrate mixed-layer equations
        self.abl_height = h0 + dt * self.htend
        self.theta = theta0 + dt * self.thetatend
        self.dtheta = dtheta0 + dt * self.dthetatend
        self.q = q0 + dt * self.qtend
        self.dq = dq0 + dt * self.dqtend
        self.co2 = CO20 + dt * self.co2tend
        self.dCO2 = dCO20 + dt * self.dCO2tend
        self.dz_h = dz0 + dt * self.dztend

        # limit dz to minimal value
        dz0 = 50
        if self.dz_h < dz0:
            self.dz_h = dz0

        if self.sw_wind:
            self.u = u0 + dt * self.utend
            self.du = du0 + dt * self.dutend
            self.v = v0 + dt * self.vtend
            self.dv = dv0 + dt * self.dvtend
