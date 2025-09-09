from abc import abstractmethod
from typing import Generic, TypeVar

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

ST = TypeVar("ST", bound="AbstractStandardLandSurfaceModel")


class StandardLandSurfaceParams(AbstractParams["AbstractStandardLandSurfaceModel"]):
    """Data class for standard land surface model parameters.

    Arguments
    ---------
    - ``a``: Clapp and Hornberger (1978) retention curve parameter.
    - ``b``: Clapp and Hornberger (1978) retention curve parameter.
    - ``p``: Clapp and Hornberger (1978) retention curve parameter.
    - ``cgsat``: Saturated soil heat capacity [J m-3 K-1].
    - ``wsat``: Saturated soil moisture content [m3 m-3].
    - ``wfc``: Soil moisture content at field capacity [m3 m-3].
    - ``wwilt``: Soil moisture content at wilting point [m3 m-3].
    - ``c1sat``: saturated soil conductivity parameter [-].
    - ``c2ref``: reference soil conductivity parameter [-].
    - ``lai``: Leaf area index [m2 m-2].
    - ``gD``: Canopy radiation extinction coefficient [-].
    - ``rsmin``: Minimum stomatal resistance [s m-1].
    - ``rssoilmin``: Minimum soil resistance [s m-1].
    - ``alpha``: Initial slope of the light response curve [mol J-1].
    - ``cveg``: Vegetation fraction [-].
    - ``wmax``: Maximum water storage capacity of the canopy [m].
    - ``lam``: Thermal diffusivity of the soil [W m-1 K-1].

    """

    def __init__(
        self,
        a: float,
        b: float,
        p: float,
        cgsat: float,
        wsat: float,
        wfc: float,
        wwilt: float,
        c1sat: float,
        c2ref: float,
        lai: float,
        gD: float,
        rsmin: float,
        rssoilmin: float,
        alpha: float,
        cveg: float,
        wmax: float,
        lam: float,
    ):
        self.a = a
        self.b = b
        self.p = p
        self.cgsat = cgsat
        self.wsat = wsat
        self.wfc = wfc
        self.wwilt = wwilt
        self.c1sat = c1sat
        self.c2ref = c2ref
        self.lai = lai
        self.gD = gD
        self.rsmin = rsmin
        self.rssoilmin = rssoilmin
        self.alpha = alpha
        self.cveg = cveg
        self.wmax = wmax
        self.lam = lam
        self.c_beta = 0.0


class StandardLandSurfaceInitConds(
    AbstractInitConds["AbstractStandardLandSurfaceModel"]
):
    """Data class for standard land surface model initial conditions.

    Arguments
    ---------
    - ``wg``: Soil moisture content in the root zone [m3 m-3].
    - ``w2``: Soil moisture content in the deep layer [m3 m-3].
    - ``temp_soil``: Soil temperature [K].
    - ``temp2``: Deep soil temperature [K].
    - ``surf_temp``: Surface temperature [K].
    - ``wl``: Liquid water storage on the canopy [m].

    """

    def __init__(
        self,
        wg: float,
        w2: float,
        temp_soil: float,
        temp2: float,
        surf_temp: float,
        wl: float,
    ):
        self.wg = wg
        self.w2 = w2
        self.temp_soil = temp_soil
        self.temp2 = temp2
        self.surf_temp = surf_temp
        self.wl = wl
        self.rs = 1.0e6
        self.rssoil = 1.0e6


class StandardLandSurfaceDiagnostics(AbstractDiagnostics[ST], Generic[ST]):
    """Class for standard land surface model diagnostics.

    Variables
    ---------
    - ``cliq``: Wet fraction of the canopy [-].
    - ``temp_soil_tend``: Soil temperature tendency [K s-1].
    - ``wgtend``: Soil moisture tendency [m3 m-3 s-1].
    - ``wltend``: Canopy water storage tendency [m s-1].
    - ``surf_temp``: Surface temperature [K].
    - ``le_veg``: Latent heat flux from vegetation [W m-2].
    - ``le_liq``: Latent heat flux from liquid water [W m-2].
    - ``le_soil``: Latent heat flux from soil [W m-2].
    - ``le``: Total latent heat flux [W m-2].
    - ``hf``: Sensible heat flux [W m-2].
    - ``gf``: Ground heat flux [W m-2].
    - ``le_pot``: Potential latent heat flux [W m-2].
    - ``le_ref``: Reference latent heat flux [W m-2].
    - ``ra``: Aerodynamic resistance [s m-1].
    """

    def post_init(self, tsteps: int):
        self.cliq = np.zeros(tsteps)
        self.temp_soil_tend = np.zeros(tsteps)
        self.wgtend = np.zeros(tsteps)
        self.wltend = np.zeros(tsteps)
        self.surf_temp = np.zeros(tsteps)
        self.le_veg = np.zeros(tsteps)
        self.le_liq = np.zeros(tsteps)
        self.le_soil = np.zeros(tsteps)
        self.le = np.zeros(tsteps)
        self.hf = np.zeros(tsteps)
        self.gf = np.zeros(tsteps)
        self.le_pot = np.zeros(tsteps)
        self.le_ref = np.zeros(tsteps)
        self.ra = np.zeros(tsteps)

    def store(self, t: int, model: ST):
        self.cliq[t] = model.cliq
        self.temp_soil_tend[t] = model.temp_soil_tend
        self.wgtend[t] = model.wgtend
        self.wltend[t] = model.wltend
        self.surf_temp[t] = model.surf_temp
        self.le_veg[t] = model.le_veg
        self.le_liq[t] = model.le_liq
        self.le_soil[t] = model.le_soil
        self.le[t] = model.le
        self.hf[t] = model.hf
        self.gf[t] = model.gf
        self.le_pot[t] = model.le_pot
        self.le_ref[t] = model.le_ref
        self.ra[t] = model.ra


class AbstractStandardLandSurfaceModel(AbstractLandSurfaceModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics."""

    # wet fraction [-]
    cliq: float
    # soil temperature tendency [K s-1]
    temp_soil_tend: float
    # soil moisture tendency [m3 m-3 s-1]
    wgtend: float
    # equivalent liquid water tendency [m s-1]
    wltend: float
    # aerodynamic resistance [s m-1]
    ra: float

    def __init__(
        self,
        params: StandardLandSurfaceParams,
        init_conds: StandardLandSurfaceInitConds,
        diagnostics: AbstractDiagnostics = StandardLandSurfaceDiagnostics(),
    ):
        # water content parameters
        self.wg = init_conds.wg
        self.w2 = init_conds.w2
        self.wsat = params.wsat
        self.wfc = params.wfc
        self.wwilt = params.wwilt

        # temperature params
        self.temp_soil = init_conds.temp_soil
        self.temp2 = init_conds.temp2
        self.surf_temp = init_conds.surf_temp

        # Clapp and Hornberger retention curve parameters
        self.a = params.a
        self.b = params.b
        self.p = params.p
        self.cgsat = params.cgsat

        # C parameters
        self.c1sat = params.c1sat
        self.c2ref = params.c2ref

        # vegetation parameters
        self.lai = params.lai
        self.gD = params.gD
        self.rsmin = params.rsmin
        self.rssoilmin = params.rssoilmin
        self.alpha = params.alpha

        # resistance parameters
        self.rs = init_conds.rs
        self.rssoil = init_conds.rssoil

        # vegetation and water layer parameters
        self.cveg = params.cveg
        self.wmax = params.wmax
        self.wl = init_conds.wl

        # thermal diffusivity
        self.lamb = params.lam

        # old: some sanity checks for valid input
        # limamau: I think this is supposed to be a parameter
        self.c_beta = 0.0  # zero curvature; linear response
        assert self.c_beta >= 0.0 or self.c_beta <= 1.0

        self.diagnostics = diagnostics

    @abstractmethod
    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ) -> None:
        raise NotImplementedError

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Execute complete land surface model calculations.

        Parameters
        ----------
        - ``const``: physical constants. Uses ``rho``, ``cp``, ``lv``, ``rhow``.
        - ``radiation``: radiation model. Uses ``net_rad``.
        - ``surface_layer``: surface layer model. ``compute_ra`` method.
        - ``mixed_layer``: mixed layer model. Uses ``u``, ``v``, ``wstar``, ``theta``,
          ``surf_pressure``, ``q``, and updates ``esat``, ``qsat``, ``dqsatdT``, ``e``,
          ``qsatsurf``, ``wtheta``, ``wq``.

        Updates
        -------
        Updates all surface fluxes, resistances, soil tendencies, and kinematic
        fluxes. Calculates surface temperature implicitly using energy balance
        equation including vegetation, soil, and liquid water components.
        """
        # compute aerodynamic resistance
        self.ra = surface_layer.compute_ra(
            mixed_layer.u, mixed_layer.v, mixed_layer.wstar
        )

        # first calculate essential thermodynamic variables
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

        # sub-model part
        self.compute_surface_resistance(const, radiation, surface_layer, mixed_layer)
        self.compute_co2_flux(const, surface_layer, mixed_layer)

        # recompute f2 using wg instead of w2
        if self.wg > self.wwilt:  # and self.w2 <= self.wfc):
            f2 = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
            f2 = 1.0e8
        self.rssoil = self.rssoilmin * f2

        wlmx = self.lai * self.wmax
        self.cliq = min(1.0, self.wl / wlmx)

        # calculate skin temperature implicitly
        self.surf_temp = (
            radiation.net_rad
            + const.rho * const.cp / self.ra * mixed_layer.theta
            + self.cveg
            * (1.0 - self.cliq)
            * const.rho
            * const.lv
            / (self.ra + self.rs)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (self.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.cveg
            * self.cliq
            * const.rho
            * const.lv
            / self.ra
            * (
                mixed_layer.dqsatdT * mixed_layer.theta
                - mixed_layer.qsat
                + mixed_layer.q
            )
            + self.lamb * self.temp_soil
        ) / (
            const.rho * const.cp / self.ra
            + self.cveg
            * (1.0 - self.cliq)
            * const.rho
            * const.lv
            / (self.ra + self.rs)
            * mixed_layer.dqsatdT
            + (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (self.ra + self.rssoil)
            * mixed_layer.dqsatdT
            + self.cveg
            * self.cliq
            * const.rho
            * const.lv
            / self.ra
            * mixed_layer.dqsatdT
            + self.lamb
        )

        # limamau: should eastsurf just be deleted here?
        # or should it rather be updated on mixed layer?
        # esatsurf = get_esat(self.surf_temp)
        mixed_layer.qsatsurf = get_qsat(self.surf_temp, mixed_layer.surf_pressure)

        self.le_veg = (
            (1.0 - self.cliq)
            * self.cveg
            * const.rho
            * const.lv
            / (self.ra + self.rs)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_liq = (
            self.cliq
            * self.cveg
            * const.rho
            * const.lv
            / self.ra
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )
        self.le_soil = (
            (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (self.ra + self.rssoil)
            * (
                mixed_layer.dqsatdT * (self.surf_temp - mixed_layer.theta)
                + mixed_layer.qsat
                - mixed_layer.q
            )
        )

        self.wltend = -self.le_liq / (const.rhow * const.lv)

        self.le = self.le_soil + self.le_veg + self.le_liq
        self.hf = const.rho * const.cp / self.ra * (self.surf_temp - mixed_layer.theta)
        self.gf = self.lamb * (self.surf_temp - self.temp_soil)
        self.le_pot = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + const.rho * const.cp / self.ra * (mixed_layer.qsat - mixed_layer.q)
        ) / (mixed_layer.dqsatdT + const.cp / const.lv)
        self.le_ref = (
            mixed_layer.dqsatdT * (radiation.net_rad - self.gf)
            + const.rho * const.cp / self.ra * (mixed_layer.qsat - mixed_layer.q)
        ) / (
            mixed_layer.dqsatdT
            + const.cp / const.lv * (1.0 + self.rsmin / self.lai / self.ra)
        )

        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * np.log(10.0)))

        self.temp_soil_tend = cg * self.gf - 2.0 * np.pi / 86400.0 * (
            self.temp_soil - self.temp2
        )

        d1 = 0.1
        c1 = self.c1sat * (self.wsat / self.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        self.wgtend = -c1 / (
            const.rhow * d1
        ) * self.le_soil / const.lv - c2 / 86400.0 * (self.wg - wgeq)

        # calculate kinematic heat fluxes
        mixed_layer.wtheta = self.hf / (const.rho * const.cp)
        mixed_layer.wq = self.le / (const.rho * const.lv)

    def integrate(self, dt: float):
        """
        Integrate model forward in time.
        """
        self.temp_soil += dt * self.temp_soil_tend
        self.wg += dt * self.wgtend
        self.wl += dt * self.wltend
