from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...utils import PhysicalConstants
from .stats import AbstractStandardStatsModel

# conversion factor mgC m-2 s-1 to ppm m s-1
# limamau: this conversion could be done in a post-processing
# function after jax.lax.scan just like in neuralgcm/dinosaur
# FAC = const.mair / (const.rho * const.mco2)


@dataclass
class BulkMixedLayerInitConds:
    """Data class for bulk mixed layer model initial state."""

    # initialized by the user
    h_abl: float
    """Initial atmospheric boundary layer (ABL) height [m]."""
    θ: float
    """Initial mixed-layer potential temperature [K]."""
    Δθ: float
    """Initial temperature jump at the top of the ABL [K]."""
    wθ: float
    """Surface kinematic heat flux [K m/s]."""
    q: float
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: float
    """Initial specific humidity jump at h [kg/kg]."""
    wq: float
    """Surface kinematic moisture flux [kg/kg m/s]."""
    co2: float
    """Initial mixed-layer CO2 [ppm]."""
    dCO2: float
    """Initial CO2 jump at the top of the ABL [ppm]."""
    wCO2: float
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: float
    """Initial mixed-layer u-wind speed [m/s]."""
    du: float
    """Initial u-wind jump at the top of the ABL [m/s]."""
    v: float
    """Initial mixed-layer v-wind speed [m/s]."""
    dv: float
    """Initial v-wind jump at the top of the ABL [m/s]."""
    dz_h: float
    """Transition layer thickness [m]."""
    surf_pressure: float
    """Surface pressure, which is actually not updated (not a state), it's only here for simplicity [Pa]."""

    # initialized to zero by default
    wstar: float = 0.0
    """Convective velocity scale [m s-1]."""
    we: float = -1.0
    """Entrainment velocity [m s-1]."""
    wCO2A: float = 0.0
    """Surface assimulation CO2 flux [mgC/m²/s]."""
    wCO2R: float = 0.0
    """Surface respiration CO2 flux [mgC/m²/s]."""
    wCO2M: float = 0.0
    """CO2 mass flux [mgC/m²/s]."""

    # should be initialized during warmup
    θv: float = jnp.nan
    """Mixed-layer potential temperature [K]."""
    Δθv: float = jnp.nan
    """Virtual temperature jump at the top of the ABL [K]."""
    wθv: float = jnp.nan
    """Surface kinematic virtual heat flux [K m s-1]."""
    wqe: float = jnp.nan
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    qsat: float = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    e: float = jnp.nan
    """Vapor pressure [Pa]."""
    esat: float = jnp.nan
    """Saturation vapor pressure [Pa]."""
    wCO2e: float = jnp.nan
    """Entrainment CO2 flux [mgC/m²/s]."""
    wθe: float = jnp.nan
    """Entrainment potential temperature flux [K m s-1]."""
    wθve: float = jnp.nan
    """Entrainment virtual heat flux [K m s-1]."""
    lcl: float = jnp.nan
    """Lifting condensation level [m]."""
    top_rh: float = jnp.nan
    """Top of mixed layer relative humidity [%]."""
    top_p: float = jnp.nan
    """Pressure at top of mixed layer [Pa]."""
    top_T: float = jnp.nan
    """Temperature at top of mixed layer [K]."""
    utend: float = jnp.nan
    """Zonal wind velocity tendency [m s-2]."""
    dutend: float = jnp.nan
    """Zonal wind velocity tendency at the ABL height [m s-2]."""
    vtend: float = jnp.nan
    """Meridional wind velocity tendency [m s-2]."""
    dvtend: float = jnp.nan
    """Meridional wind velocity tendency at the ABL height [m/s²]."""
    h_abl_tend: float = jnp.nan
    """Tendency of CBL [m s-1]."""
    θtend: float = jnp.nan
    """Tendency of mixed-layer potential temperature [K s-1]."""
    Δθtend: float = jnp.nan
    """Tendency of mixed-layer potential temperature at the ABL height [K s-1]."""
    qtend: float = jnp.nan
    """Tendency of mixed-layer specific humidity [kg/kg s-1]."""
    dqtend: float = jnp.nan
    """Tendency of mixed-layer specific humidity at the ABL height [kg/kg s-1]."""
    co2tend: float = jnp.nan
    """Tendency of CO2 concentration [ppm s-1]."""
    dCO2tend: float = jnp.nan
    """Tendency of CO2 concentration at the ABL height [ppm s-1]."""
    dztend: float = jnp.nan
    """Tendency of transition layer thickness [m s-1]."""
    ws: float = jnp.nan
    """Large-scale vertical velocity (subsidence) [m s-1]."""
    wf: float = jnp.nan
    """Mixed-layer growth due to cloud top radiative divergence [m s-1]."""


class BulkMixedLayerModel(AbstractStandardStatsModel):
    """Bulk mixed layer model with full atmospheric boundary layer dynamics.

    Complete mixed layer model that simulates atmospheric boundary layer evolution
    including entrainment, subsidence, cloud effects, and wind dynamics.

    Args:
        divU: horizontal large-scale divergence of wind [s-1].
        coriolis_param: Coriolis parameter [s-1].
        γθ: free atmosphere potential temperature lapse rate [K m-1].
        advθ: advection of heat [K s-1].
        beta: entrainment ratio for virtual heat [-].
        γq: free atmosphere specific humidity lapse rate [kg/kg m-1].
        advq: advection of moisture [kg/kg s-1].
        γCO2: free atmosphere CO2 lapse rate [ppm m-1].
        advCO2: advection of CO2 [ppm s-1].
        γu: free atmosphere u-wind speed lapse rate [s-1].
        advu: advection of u-wind [m s-2].
        γv: free atmosphere v-wind speed lapse rate [s-1].
        advv: advection of v-wind [m s-2].
        dFz: cloud top radiative divergence [W m-2].
        is_shear_growing: shear growth mixed-layer switch.
        is_fix_free_trop: fix the free-troposphere switch.
        is_wind_prog: prognostic wind switch.
    """

    def __init__(
        self,
        divU: float,
        coriolis_param: float,
        γθ: float,
        advθ: float,
        beta: float,
        γq: float,
        advq: float,
        γCO2: float,
        advCO2: float,
        γu: float,
        advu: float,
        γv: float,
        advv: float,
        dFz: float,
        is_shear_growing: bool = True,
        is_fix_free_trop: bool = True,
        is_wind_prog: bool = True,
    ):
        self.divU = divU
        self.coriolis_param = coriolis_param
        self.γ_θ = γθ
        self.advθ = advθ
        self.beta = beta
        self.γ_q = γq
        self.advq = advq
        self.γCO2 = γCO2
        self.advCO2 = advCO2
        self.γu = γu
        self.advu = advu
        self.γv = γv
        self.advv = advv
        self.ΔFz = dFz
        self.is_shear_growing = is_shear_growing
        self.is_fix_free_trop = is_fix_free_trop
        self.is_wind_prog = is_wind_prog

    def run(self, state: PyTree, const: PhysicalConstants):
        """Run the model."""
        state.ws = self.compute_ws(state.h_abl)
        state.wf = self.compute_wf(state.Δθ, const)
        w_th_ft = self.compute_w_th_ft(state.ws)
        w_q_ft = self.compute_w_q_ft(state.ws)
        w_CO2_ft = self.compute_w_CO2_ft(state.ws)
        state.wstar = self.compute_wstar(
            state.h_abl,
            state.wθv,
            state.θv,
            const.g,
        )
        state.wθve = self.compute_wθve(state.wθv)
        state.we = self.compute_we(
            state.h_abl,
            state.wθve,
            state.Δθv,
            state.θv,
            state.ustar,
            const.g,
        )
        state.wθe = self.compute_wθe(state.we, state.Δθ)
        state.wqe = self.compute_wqe(state.we, state.dq)
        state.wCO2e = self.compute_wCO2e(state.we, state.dCO2)
        state.h_abl_tend = self.compute_h_abl_tend(
            state.we, state.ws, state.wf, state.cc_mf
        )
        state.θtend = self.compute_θtend(state.h_abl, state.wθ, state.wθe)
        state.Δθtend = self.compute_Δθtend(
            state.we, state.wf, state.cc_mf, state.θtend, w_th_ft
        )
        state.qtend = self.compute_qtend(state.h_abl, state.wq, state.wqe, state.cc_qf)
        state.dqtend = self.compute_dqtend(
            state.we, state.wf, state.cc_mf, state.qtend, w_q_ft
        )
        state.co2tend = self.compute_co2tend(
            state.h_abl, state.wCO2, state.wCO2e, state.wCO2M
        )
        state.dCO2tend = self.compute_dCO2tend(
            state.we, state.wf, state.cc_mf, state.co2tend, w_CO2_ft
        )
        state.utend = self.compute_utend(
            state.h_abl, state.we, state.uw, state.du, state.dv
        )
        state.vtend = self.compute_vtend(
            state.h_abl, state.we, state.vw, state.du, state.dv
        )
        state.dutend = self.compute_dutend(state.we, state.wf, state.cc_mf, state.utend)
        state.dvtend = self.compute_dvtend(state.we, state.wf, state.cc_mf, state.vtend)
        state.dztend = self.compute_dztend(
            state.lcl,
            state.h_abl,
            state.cc_frac,
            state.dz_h,
        )
        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        """Integrate mixed layer forward in time."""
        state.h_abl += dt * state.h_abl_tend
        state.θ += dt * state.θtend
        state.Δθ += dt * state.Δθtend
        state.q += dt * state.qtend
        state.dq += dt * state.dqtend
        state.co2 += dt * state.co2tend
        state.dCO2 += dt * state.dCO2tend
        state.dz_h += dt * state.dztend

        # limit dz to minimal value
        state.dz_h = jnp.maximum(state.dz_h, 50.0)

        state.u = jnp.where(self.is_wind_prog, state.u + dt * state.utend, state.u)
        state.du = jnp.where(self.is_wind_prog, state.du + dt * state.dutend, state.du)
        state.v = jnp.where(self.is_wind_prog, state.v + dt * state.vtend, state.v)
        state.dv = jnp.where(self.is_wind_prog, state.dv + dt * state.dvtend, state.dv)

        return state

    def compute_ws(self, h_abl: Array) -> Array:
        """Compute the large-scale subsidence velocity as

        .. math::
            w_s = -\\text{div}U \\cdot h,

        where :math:`\\text{div}U` is the horizontal large-scale divergence of wind and :math:`h` is the ABL height.
        """
        return -self.divU * h_abl

    def compute_wf(self, Δθ: Array, const: PhysicalConstants) -> Array:
        """Compute the mixed-layer growth due to cloud top radiative divergence as

        .. math::
            w_f = \\frac{\\Delta F_z}{\\rho c_p \\Delta \\theta},

        where :math:`\\Delta F_z` is the cloud top radiative divergence, :math:`\\rho` is air density,
        :math:`c_p` is specific heat capacity, and :math:`\\Delta \\theta` is the temperature jump at the top of the ABL.
        """
        radiative_denominator = const.rho * const.cp * Δθ
        return self.ΔFz / radiative_denominator

    def compute_w_th_ft(self, ws: Array) -> Array:
        """Compute the potential temperature compensation term to fix free troposphere values as

        .. math::
            w_{\\theta,ft} = \\gamma_\\theta w_s,

        where :math:`\\gamma_\\theta` is the potential temperature compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_th_ft_active = self.γ_θ * ws
        return jnp.where(self.is_fix_free_trop, w_th_ft_active, 0.0)

    def compute_w_q_ft(self, ws: Array) -> Array:
        """Compute humidity compensation term to fix free troposphere values as

        .. math::
            w_{q,ft} = \\gamma_q w_s,

        where :math:`\\gamma_q` is the humidity compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_q_ft_active = self.γ_q * ws
        return jnp.where(self.is_fix_free_trop, w_q_ft_active, 0.0)

    def compute_w_CO2_ft(self, ws: Array) -> Array:
        """Compute CO2 compensation term to fix free troposphere values as

        .. math::
            w_{CO2,ft} = \\gamma_{CO2} w_s,

        where :math:`\\gamma_{CO2}` is the CO2 compensation factor and :math:`w_s` comes from :meth:`compute_ws`.
        This is used in case we are fixing the free troposhere.
        """
        w_CO2_ft_active = self.γCO2 * ws
        return jnp.where(self.is_fix_free_trop, w_CO2_ft_active, 0.0)

    def compute_wstar(
        self,
        h_abl: Array,
        wθv: Array,
        θv: Array,
        g: float,
    ) -> Array:
        """Compute the convective velocity scale, defined by

        .. math::
            w_* = \\left( \\frac{g h (\\overline{w'\\theta_v'})_s}{\\theta_v} \\right)^{1/3},

        where :math:`g` is the gravity acceleration, :math:`h` is the height of the atmospheric boundary layer,
        :math:`(\\overline{w'\\theta_v'})_s` is the virtual heat flux at the surface and :math:`\\theta_v` is
        the virtual potential temperature.
        """
        buoyancy_term = g * h_abl * wθv / θv
        wstar_positive = buoyancy_term ** (1.0 / 3.0)
        # clip to 1e-6 in case wθv is negative
        return jnp.where(wθv > 0.0, wstar_positive, 1e-6)

    def compute_wθve(self, wθv: Array) -> Array:
        """Compute the entrainment virtual heat flux as

        .. math::
            (\\overline{w'\\theta_v'})_e = -\\beta (\\overline{w'\\theta_v'})_s,

        where :math:`\\beta` is the entrainment coefficient and
        :math:`(\\overline{w'\\theta_v'})_s` is the virtual heat flux at the surface.
        """
        return -self.beta * wθv

    def compute_we(
        self,
        h_abl: Array,
        wθve: Array,
        Δθv: Array,
        θv: Array,
        ustar: Array,
        g: float,
    ):
        """Compute the entrainment velocity as

        .. math::
            w_e = -\\frac{(\\overline{w'\\theta_v'})_e}{\\Delta \\theta_v},

        where :math:`(\\overline{w'\\theta_v'})_e` is the entrainment virtual heat flux
        and :math:`\\Delta \\theta_v` is the virtual potential temperature jump at the top of the ABL.

        If shear effects are included (``is_shear_growing=True``), an additional term is added

        .. math::
            w_e = \\frac{-\\overline{w'\\theta_v'}_e + 5 u_*^3 \\theta_v / (g h)}{\\Delta \\theta_v},

        where :math:`u_*` is the friction velocity, and :math:`\\theta_v` is the virtual potential temperature,
        :math:`g` is gravity acceleration, and :math:`h` is the height of the ABL.
        """
        # entrainment velocity with shear effects
        shear_term = 5.0 * ustar**3.0 * θv / (g * h_abl)
        numerator = -wθve + shear_term
        we_with_shear = numerator / Δθv

        # entrainment velocity without shear effects
        we_no_shear = -wθve / Δθv

        # select based on is_shear_growing flag
        we_calculated = jnp.where(self.is_shear_growing, we_with_shear, we_no_shear)

        # don't allow boundary layer shrinking if wθ < 0
        assert isinstance(we_calculated, jnp.ndarray)  # limmau: this is not good
        we_final = jnp.where(we_calculated < 0.0, 0.0, we_calculated)

        return we_final

    def compute_wθe(self, we: Array, Δθ: Array) -> Array:
        """Compute the entrainment heat flux as

        .. math::
            (\\overline{w'\\theta'})_e = -w_e \\Delta \\theta,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta \\theta` is the potential temperature jump at the top of the ABL.
        """
        return -we * Δθ

    def compute_wqe(self, we: Array, dq: Array) -> Array:
        """Compute the entrainment moisture flux as

        .. math::
            (\\overline{w'q'})_e = -w_e \\Delta q,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta q` is the moisture jump at the top of the ABL.
        """
        return -we * dq

    def compute_wCO2e(self, we: Array, dCO2: Array) -> Array:
        """Compute the entrainment CO2 flux as

        .. math::
            (\\overline{w'CO_2'})_e = -w_e \\Delta CO_2,

        where :math:`w_e` is the entrainment velocity
        and :math:`\\Delta CO_2` is the CO2 jump at the top of the ABL.
        """
        return -we * dCO2

    def compute_h_abl_tend(
        self, we: Array, ws: Array, wf: Array, cc_mf: Array
    ) -> Array:
        """Compute the boundary layer height tendency as

        .. math::
            \\frac{dh}{dt} = w_e + w_s + w_f - \\text{cc}_{mf},

        where

        - :math:`w_e` comes from :meth:`compute_we`,

        - :math:`w_s` comes from :meth:`compute_ws`,

        - :math:`w_f` comes from :meth:`compute_wf`,

        and :math:`\\text{cc}_{mf}` is the cloud core mass flux from the cloud model.
        """
        return we + ws + wf - cc_mf

    def compute_θtend(self, h_abl: Array, wθ: Array, wθe: Array) -> Array:
        """Compute the mixed-layer potential temperature tendency :math:`\\frac{d\\theta}{dt}` as

        .. math::
            \\frac{d\\theta}{dt} = \\frac{(\\overline{w'\\theta'})_s - (\\overline{w'\\theta'})_e}{h} + \\text{adv}_\\theta,

        where :math:`\\overline{w'\\theta'}_s` is the surface potential temperature flux,
        :math:`\\overline{w'\\theta'}_e` is the entrainment potential temperature flux,
        and :math:`\\text{adv}_\\theta` is the potential temperature advection.
        """
        surface_heat_flux = (wθ - wθe) / h_abl
        return surface_heat_flux + self.advθ

    def compute_Δθtend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        θtend: Array,
        w_th_ft: Array,
    ) -> Array:
        """Compute the potential temperature jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta \\theta}{\\text{d}t}
            = \\gamma_\\theta (w_e + w_f - \\text{cc}_{mf})
            - \\frac{\\text{d}\\theta}{\\text{d}t} + w_{\\theta,ft},

        where
        """
        egrowth = we + wf - cc_mf
        return self.γ_θ * egrowth - θtend + w_th_ft

    def compute_qtend(self, h_abl: Array, wq: Array, wqe: Array, cc_qf: Array) -> Array:
        """Compute the mixed-layer specific humidity tendency as

        .. math::
            \\frac{\\text{d}q}{\\text{d}t} = \\frac{\\overline{w'q'}_s - \\overline{w'q'}_e - \\text{cc}_{qf}}{h} + \\text{adv}_q
        """
        surface_moisture_flux = (wq - wqe - cc_qf) / h_abl
        return surface_moisture_flux + self.advq

    def compute_dqtend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        qtend: Array,
        w_q_ft: Array,
    ) -> Array:
        """Compute the specific humidity jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta q}{\\text{d}t} = \\gamma_q (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}q}{\\text{d}t} + w_{q,ft}
        """
        egrowth = we + wf - cc_mf
        return self.γ_q * egrowth - qtend + w_q_ft

    def compute_co2tend(
        self,
        h_abl: Array,
        wCO2: Array,
        wCO2e: Array,
        wCO2M: Array,
    ) -> Array:
        """Compute the mixed-layer CO2 tendency as

        .. math::
            \\frac{\\text{d}CO_2}{\\text{d}t}
            = \\frac{\\overline{w'CO_2'}_s - \\overline{w'CO_2'}_e - \\text{cc}_{CO2f}}{h} + \\text{adv}_{CO2}
        """
        surface_co2_flux_term = (wCO2 - wCO2e - wCO2M) / h_abl
        return surface_co2_flux_term + self.advCO2

    def compute_dCO2tend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        co2tend: Array,
        w_CO2_ft: Array,
    ) -> Array:
        """Compute the CO2 jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta CO_2}{\\text{d}t}
            = \\gamma_{CO2} (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}CO_2}{\\text{d}t} + w_{CO2,ft}
        """
        egrowth = we + wf - cc_mf
        return self.γCO2 * egrowth - co2tend + w_CO2_ft

    def compute_utend(
        self,
        h_abl: Array,
        we: Array,
        uw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute the zonal wind tendency as

        .. math::
            \\frac{\\text{d}u}{\\text{d}t}
            = -f_c \\Delta v + \\frac{\\overline{u'w'}_s + w_e \\Delta u}{h} + \\text{adv}_u
        """
        coriolis_term_u = -self.coriolis_param * dv
        momentum_flux_term_u = (uw + we * du) / h_abl
        utend_active = coriolis_term_u + momentum_flux_term_u + self.advu
        return jnp.where(self.is_wind_prog, utend_active, 0.0)

    def compute_vtend(
        self,
        h_abl: Array,
        we: Array,
        vw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute the meridional wind tendency as

        .. math::
            \\frac{\\text{d}v}{\\text{d}t} = f_c \\Delta u + \\frac{\\overline{v'w'}_s + w_e \\Delta v}{h} + \\text{adv}_v
        """
        coriolis_term_v = self.coriolis_param * du
        momentum_flux_term_v = (vw + we * dv) / h_abl
        vtend_active = coriolis_term_v + momentum_flux_term_v + self.advv
        return jnp.where(self.is_wind_prog, vtend_active, 0.0)

    def compute_dutend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        utend: Array,
    ) -> Array:
        """Compute zonal wind jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta u}{\\text{d}t} = \\gamma_u (w_e + w_f - \\text{cc}_{mf}) - \\frac{\\text{d}u}{\\text{d}t}
        """
        entrainment_growth_term = we + wf - cc_mf
        dutend_active = self.γu * entrainment_growth_term - utend
        return jnp.where(self.is_wind_prog, dutend_active, 0.0)

    def compute_dvtend(
        self,
        we: Array,
        wf: Array,
        cc_mf: Array,
        vtend: Array,
    ) -> Array:
        """Compute meridional wind jump at the top of the ABL tendency as

        .. math::
            \\frac{\\text{d}\\Delta v}{\\text{d}t}
            = \\gamma_v (w_e + w_f - \\text{cc}_{mf})
            - \\frac{\\text{d}v}{\\text{d}t}
        """
        entrainment_growth_term = we + wf - cc_mf
        dvtend_active = self.γv * entrainment_growth_term - vtend
        return jnp.where(self.is_wind_prog, dvtend_active, 0.0)

    def compute_dztend(
        self,
        lcl: Array,
        h_abl: Array,
        cc_frac: Array,
        dz_h: Array,
    ):
        """Compute the transition layer thickness tendency as

        .. math::
            \\frac{\\text{d}\\delta z_h}{\\text{d}t} = \\frac{(LCL - h) - \\delta z_h}{\\tau}

        where :math:`\\tau = 7200` s. This is basically a relaxation term.
        """
        lcl_distance = lcl - h_abl

        # tendency for active case
        target_thickness = lcl_distance - dz_h
        dztend_active = target_thickness / 7200.0
        condition = (cc_frac > 0) | (lcl_distance < 300)
        dztend = jnp.where(condition, dztend_active, 0.0)

        return dztend
