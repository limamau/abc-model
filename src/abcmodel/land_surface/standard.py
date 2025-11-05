from abc import abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..models import (
    AbstractLandSurfaceModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_esat, get_qsat


@dataclass
class StandardLandSurfaceInitConds:
    """Standard land surface model initial state."""

    # the following variables are supposed to be initialized by the user
    alpha: float
    """Slope of the light response curve [mol J-1]."""
    wg: float
    """Soil moisture content in the root zone [m3 m-3]."""
    w2: float
    """Soil moisture content in the deep layer [m3 m-3]."""
    temp_soil: float
    """Soil temperature [K]."""
    temp2: float
    """Deep soil temperature [K]."""
    surf_temp: float
    """Surface temperature [K]."""
    wl: float
    """Liquid water storage on the canopy [m]."""

    # the following variables are initialized to high values and
    # are expected to converge during warmup
    rs: float = 1.0e6
    """Surface resistance [m s-1]."""
    rssoil: float = 1.0e6
    """Soil resistance [m s-1]."""

    # the following variables are expected to be assigned during warmup
    cliq: float = jnp.nan
    """Wet fraction of the canopy [-]."""
    temp_soil_tend: float = jnp.nan
    """Soil temperature tendency [K s-1]."""
    wgtend: float = jnp.nan
    """Soil moisture tendency [m3 m-3 s-1]."""
    wltend: float = jnp.nan
    """Canopy water storage tendency [m s-1]."""
    le_veg: float = jnp.nan
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: float = jnp.nan
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: float = jnp.nan
    """Latent heat flux from soil [W m-2]."""
    le: float = jnp.nan
    """Total latent heat flux [W m-2]."""
    hf: float = jnp.nan
    """Sensible heat flux [W m-2]."""
    gf: float = jnp.nan
    """Ground heat flux [W m-2]."""
    le_pot: float = jnp.nan
    """Potential latent heat flux [W m-2]."""
    le_ref: float = jnp.nan
    """Reference latent heat flux [W m-2]."""
    ra: float = jnp.nan
    """Aerodynamic resistance [s m-1]."""


class AbstractStandardLandSurfaceModel(AbstractLandSurfaceModel):
    """Abstract standard land surface model with comprehensive soil-vegetation dynamics.

    This class serves as a base, defining the parameters, core logic, and
    integration steps for a standard land surface model. It handles the
    surface energy balance, soil physics, and prognostic updates.

    It requires concrete implementations for the `compute_surface_resistance`
    and `compute_co2_flux` methods, which are specific to the chosen
    vegetation/carbon model.
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
        cveg: float,
        wmax: float,
        lam: float,
    ):
        """
        Args:
            a: Clapp and Hornberger (1978) retention curve parameter.
            b: Clapp and Hornberger (1978) retention curve parameter.
            p: Clapp and Hornberger (1978) retention curve parameter.
            cgsat: Saturated soil heat capacity [J m-3 K-1].
            wsat: Saturated soil moisture content [m3 m-3].
            wfc: Soil moisture content at field capacity [m3 m-3].
            wwilt: Soil moisture content at wilting point [m3 m-3].
            c1sat: saturated soil conductivity parameter [-].
            c2ref: reference soil conductivity parameter [-].
            lai: Leaf area index [m2 m-2].
            gD: Canopy radiation extinction coefficient [-].
            rsmin: Minimum stomatal resistance [s m-1].
            rssoilmin: Minimum soil resistance [s m-1].
            cveg: Vegetation fraction [-].
            wmax: Maximum water storage capacity of the canopy [m].
            lam: Thermal diffusivity of the soil [W m-1 K-1].
        """
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
        self.cveg = cveg
        self.wmax = wmax
        self.lam = lam
        self.c_beta = 0.0

    # limamau: this looks cleaner (state as argument rather than explicit state variables)
    # and should be implemented for the other models in the future.
    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
    ):
        """Run the full land surface model for one time step.

        Args:
            state:
            const:
            surface_layer:

        Returns:
            The updated state object.

        Notes:
            The order of execution is:

            1. :meth:`~~abcmodel.models.AbstractSurfaceLayerModel.compute_ra`
            2. :meth:`~compute_thermodynamics`
            3. :meth:`~compute_surface_resistance`
            4. :meth:`~compute_co2_flux`
            5. :meth:`~compute_soil_resistance`
            6. :meth:`~compute_wet_canopy_fraction`
            7. :meth:`~compute_skin_temperature`
            8. :meth:`~compute_surface_fluxes`
            9. :meth:`~compute_reference_fluxes`
            10. :meth:`~compute_prognostic_tendencies`
            11. :meth:`~compute_kinematic_fluxes`

        """
        state.ra = surface_layer.compute_ra(state)
        state = self.compute_thermodynamics(state)
        state = self.compute_surface_resistance(state, const)
        state = self.compute_co2_flux(state, const)
        state = self.compute_soil_resistance(state)
        state = self.compute_wet_canopy_fraction(state)
        state = self.compute_skin_temperature(state, const)
        state = self.compute_surface_fluxes(state, const)
        state = self.compute_reference_fluxes(state, const)
        state = self.compute_prognostic_tendencies(state, const)
        state = self.compute_kinematic_fluxes(state, const)
        return state

    def integrate(self, state: PyTree, dt: float):
        """Integrate the model forward one time step (Euler forward).

        Updates the prognostic variables (soil temperature, soil moisture,
        canopy water) using the tendencies calculated in the `run` step.

        Args:
            state:
            dt: time step [s].

        Returns:
            The state with updated prognostic variables.
        """
        state.temp_soil += dt * state.temp_soil_tend
        state.wg += dt * state.wgtend
        state.wl += dt * state.wltend

        return state

    @abstractmethod
    def compute_surface_resistance(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Calculate the surface resistance (stomatal) for vegetation.

        Args:
            state:
            const:

        Returns:
            The state with updated surface resistance.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_co2_flux(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        """Calculate the CO2 flux (e.g., GPP, NEE) for the surface.

        Args:
            state:
            const:

        Returns:
            The state with updated CO2 flux values.
        """
        raise NotImplementedError

    def compute_thermodynamics(self, state: PyTree) -> PyTree:
        """
        Calculate saturated vapor pressure ``esat``, saturated
        specific humidity ``qsat``, the derivative ``dqsatdT``, and
        vapor pressure ``e``.

        Args:
            state:

        Returns:
            The state with updated thermodynamic properties.

        Notes:
            The derivative of saturated vapor pressure, :math:`e_{\\text{sat}}`,
            with respect to temperature :math:`T` (in Kelvin) is derived from the
            August-Roche-Magnus formula

            .. math::
                e_{\\text{sat}} = 611 \\cdot \\exp(f(T_C)),

            where :math:`T_C = T - 273.16`:

            .. math::
                f(T_C) = \\frac{17.2694 \\cdot T_C}{T_C + 237.3}
                     = \\frac{17.2694 \\cdot (T - 273.16)}{T - 35.86}

            Using the quotient rule, the derivative :math:`f'(T)` is:

            .. math::
                f'(T) = \\frac
                { 17.2694 \\cdot (T - 35.86) - 17.2694 \\cdot (T - 273.16) }
                { (T - 35.86)^2 }

            This implementation simplifies the numerator, giving:

            .. math::
                \\frac{de_{\\text{sat}}}{dT} = e_{\\text{sat}} \\cdot f'(T)

            The derivative of specific humidity, :math:`q_{\\text{sat}}`, is then
            found using :math:`q_{\\text{sat}} \\approx 0.622 \\cdot e_{\\text{sat}} / P`:

            .. math::
                \\frac{dq_{\\text{sat}}}{dT} \\approx \\frac{0.622}{P} \\frac{de_{\\text{sat}}}{dT}
        """
        state.esat = get_esat(state.theta)
        state.qsat = get_qsat(state.theta, state.surf_pressure)

        temp_celsius = state.theta - 273.16
        temp_denom = state.theta - 35.86

        # This is f'(T) from the notes above
        f_prime = (17.2694 / temp_denom) - (17.2694 * temp_celsius / temp_denom**2.0)

        desatdT = state.esat * f_prime
        state.dqsatdT = 0.622 * desatdT / state.surf_pressure
        state.e = state.q * state.surf_pressure / 0.622
        return state

    def compute_soil_resistance(self, state: PyTree) -> PyTree:
        """Calculates the soil resistance for evaporation.

        Args:
            state:

        Returns:
            The state with updated soil resistance.

        Notes:
            A soil moisture stress factor, :math:`f_2`, is calculated based on
            the root-zone soil moisture :math:`w_g` relative to the wilting
            point :math:`w_{\\text{wilt}}` and field capacity :math:`w_{\\text{fc}}`:

            .. math::
                f_2 = \\frac{w_{\\text{fc}} - w_{\\text{wilt}}}{w_g - w_{\\text{wilt}}}

            The soil resistance :math:`r_{s,soil}` is then the minimum
            resistance scaled by this factor:

            .. math::
                r_{s,soil} = r_{s,soil,min} \\cdot f_2
        """
        f2 = jnp.where(
            state.wg > self.wwilt,
            (self.wfc - self.wwilt) / (state.wg - self.wwilt),
            1.0e8,  # Use a very high resistance if below wilting point
        )
        assert isinstance(f2, Array)
        state.rssoil = self.rssoilmin * f2
        return state

    def compute_wet_canopy_fraction(self, state: PyTree) -> PyTree:
        """Calculates the fraction of the canopy that is wet.

        Args:
            state:

        Return:
            The state updated with the wet canopy fraction.
        """
        wlmx = self.lai * self.wmax
        state.cliq = jnp.minimum(1.0, state.wl / wlmx)
        return state

    def compute_skin_temperature(
        self, state: PyTree, const: PhysicalConstants
    ) -> PyTree:
        """Calculates the surface "skin" temperature implicitly.

        Args:
            state:
            const:

        Returns
            The state updated with the skin temperature.

        Notes:
            This function solves the surface energy balance for the
            surface temperature :math:`T_{\\text{surf}}`. The energy balance is:

            .. math::
                R_{\\text{net}} - G = H + LE

            where :math:`G`, :math:`H`, and :math:`LE` are all functions of
            :math:`T_{\\text{surf}}`. The latent heat :math:`LE` is partitioned
            into vegetation, soil, and wet canopy fractions.

            By linearizing the equations for :math:`H` and :math:`LE` with
            respect to :math:`T_{\\text{surf}}`, we can rearrange the balance
            into the form :math:`T_{\\text{surf}} = \\text{Numerator} / \\text{Denominator}`.
            This is an implicit solution that avoids iterative solving.
        """
        # --- Pre-calculate common terms for readability ---
        rho_cp_ra = const.rho * const.cp / state.ra
        rho_lv_ra = const.rho * const.lv / state.ra

        # Latent heat term for dry vegetation
        le_veg_coeff = self.cveg * (1.0 - state.cliq) * const.rho * const.lv
        le_veg_denom = state.ra + state.rs
        le_veg_term = le_veg_coeff / le_veg_denom

        # Latent heat term for bare soil
        le_soil_coeff = (1.0 - self.cveg) * const.rho * const.lv
        le_soil_denom = state.ra + state.rssoil
        le_soil_term = le_soil_coeff / le_soil_denom

        # Latent heat term for wet canopy (interception)
        le_liq_term = self.cveg * state.cliq * rho_lv_ra

        # --- Build the Numerator ---
        # (Terms not dependent on surf_temp)
        numerator = (
            state.net_rad
            + rho_cp_ra * state.theta
            + self.lam * state.temp_soil
            + (le_veg_term + le_soil_term + le_liq_term)
            * (state.dqsatdT * state.theta - state.qsat + state.q)
        )

        # --- Build the Denominator ---
        # (Terms that multiply surf_temp)
        denominator = (
            rho_cp_ra
            + self.lam
            + (le_veg_term + le_soil_term + le_liq_term) * state.dqsatdT
        )

        state.surf_temp = numerator / denominator
        return state

    def compute_surface_fluxes(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        """Calculates all surface energy fluxes using the new skin temperature.

        Args:
            state:
            const:

        Returns
            The state updated with surface fluxes.

        Notes:
            The latent heat fluxes (:math:`LE`) and sensible heat flux
            (:math:`H`) are calculated using bulk-aerodynamic formulas based
            on the difference between the surface and the air, linearized
            using :math:`dq_{\\text{sat}}/dT`:

            .. math::
                LE = \\frac{\\rho \\cdot L_v}{r_a + r_s}
                     \\left[
                         \\frac{dq_{\\text{sat}}}{dT}(T_{\\text{surf}} - \\theta) +
                         (q_{\\text{sat}}(\\theta) - q)
                     \\right]

            Similar forms are used for :math:`LE_{\\text{veg}}`,
            :math:`LE_{\\text{liq}}`, and :math:`LE_{\\text{soil}}`
            using their respective resistances.

            Sensible heat flux :math:`H`:
            .. math::
                H = \\frac{\\rho \\cdot c_p}{r_a} (T_{\\text{surf}} - \\theta)

            Ground heat flux :math:`G`:
            .. math::
                G = \\lambda (T_{\\text{surf}} - T_{\\text{soil}})
        """
        # Saturated specific humidity at the new surface temperature
        state.qsatsurf = get_qsat(state.surf_temp, state.surf_pressure)

        # Common term: (dq/dT * (T_surf - T_air) + (q_sat_air - q_air))
        le_base_term = (
            state.dqsatdT * (state.surf_temp - state.theta) + state.qsat - state.q
        )

        # Latent heat from vegetation (transpiration)
        state.le_veg = (
            (1.0 - state.cliq)
            * self.cveg
            * const.rho
            * const.lv
            / (state.ra + state.rs)
            * le_base_term
        )
        # Latent heat from canopy water (evaporation)
        state.le_liq = (
            state.cliq * self.cveg * const.rho * const.lv / state.ra * le_base_term
        )
        # Latent heat from soil (evaporation)
        state.le_soil = (
            (1.0 - self.cveg)
            * const.rho
            * const.lv
            / (state.ra + state.rssoil)
            * le_base_term
        )

        # Total Latent Heat
        state.le = state.le_soil + state.le_veg + state.le_liq

        # Canopy water tendency
        state.wltend = -state.le_liq / (const.rhow * const.lv)

        # Sensible Heat Flux
        state.hf = const.rho * const.cp / state.ra * (state.surf_temp - state.theta)

        # Ground Heat Flux
        state.gf = self.lam * (state.surf_temp - state.temp_soil)
        return state

    def compute_reference_fluxes(
        self, state: PyTree, const: PhysicalConstants
    ) -> PyTree:
        """Calculates potential and reference evaporation (Penman-Monteith).

        Args:
            state: Current state of the model.
            const: Physical constants.

        Returns:
            Updated state with reference fluxes.

        """
        # common numerator for Penman-Monteith equations
        numerator = state.dqsatdT * (
            state.net_rad - state.gf
        ) + const.rho * const.cp / state.ra * (state.qsat - state.q)

        # potential evaporation (Priestley-Taylor)
        state.le_pot = numerator / (state.dqsatdT + const.cp / const.lv)

        # reference evaporation (Penman-Monteith)
        denominator_ref = state.dqsatdT + const.cp / const.lv * (
            1.0 + self.rsmin / self.lai / state.ra
        )
        state.le_ref = numerator / denominator_ref
        return state

    def compute_prognostic_tendencies(
        self, state: PyTree, const: PhysicalConstants
    ) -> PyTree:
        """Calculates the time-tendencies for soil moisture and temperature.

        Args:
            state:
            const:

        Returns:
            Updated state with prognostic tendencies.

        """
        # soil temperature tendency
        cg = self.cgsat * (self.wsat / state.w2) ** (self.b / (2.0 * jnp.log(10.0)))

        state.temp_soil_tend = cg * state.gf - 2.0 * jnp.pi / 86400.0 * (
            state.temp_soil - state.temp2
        )

        # soil moisture tendency
        d1 = 0.1  # Depth of the top soil layer (assumed 10cm)
        c1 = self.c1sat * (self.wsat / state.wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (state.w2 / (self.wsat - state.w2))

        # equilibrium soil moisture
        wgeq = state.w2 - self.wsat * self.a * (
            (state.w2 / self.wsat) ** self.p
            * (1.0 - (state.w2 / self.wsat) ** (8.0 * self.p))
        )

        # evaporation flux from soil + hydraulic redistribution
        state.wgtend = -c1 / (
            const.rhow * d1
        ) * state.le_soil / const.lv - c2 / 86400.0 * (state.wg - wgeq)

        return state

    def compute_kinematic_fluxes(
        self, state: PyTree, const: PhysicalConstants
    ) -> PyTree:
        """Converts energy fluxes (W/m^2) to kinematic fluxes (K m/s, kg/kg m/s).

        Args:
            state:
            const:

        Returns:
            Updated state with kinematic fluxes.

        """
        state.wtheta = state.hf / (const.rho * const.cp)
        state.wq = state.le / (const.rho * const.lv)
        return state
