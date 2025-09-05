import numpy as np

from ..components import (
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_psih, get_psim, get_qsat


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Standard surface layer model with atmospheric stability corrections.

    Calculates surface-atmosphere exchange using Monin-Obukhov similarity theory
    with stability functions and iterative solution for Obukhov length.

    **Processes:**
    1. Calculate effective wind speed and surface properties.
    2. Determine bulk Richardson number and solve for Obukhov length.
    3. Compute drag coefficients with stability corrections.
    4. Calculate momentum fluxes and 2m diagnostic variables.

    Arguments
    ----------
    - ``ustar``: surface friction velocity [m/s].
    - ``z0m``: roughness length for momentum [m].
    - ``z0h``: roughness length for scalars [m].
    - ``theta``: surface potential temperature [K].

    Updates
    --------
    - ``uw``, ``vw``: momentum fluxes [m²/s²].
    - ``temp_2m``, ``q2m``, ``u2m``, ``v2m``: 2m diagnostic variables.
    - ``thetavsurf``, ``qsurf``: surface properties.
    - ``obukhov_length``, ``rib_number``: stability parameters.
    - ``ra``: aerodynamic resistance [s/m].
    """

    # 2m diagnostic variables:
    # 2m temperature [K]
    temp_2m: float
    # 2m specific humidity [kg kg-1]
    q2m: float
    # 2m vapor pressure [Pa]
    e2m: float
    # 2m saturated vapor pressure [Pa]
    esat2m: float
    # 2m u-wind [m s-1]
    u2m: float
    # 2m v-wind [m s-1]
    v2m: float
    # surface momentum fluxes:
    # surface momentum flux in u-direction [m2 s-2]
    uw: float
    # surface momentum flux in v-direction [m2 s-2]
    vw: float
    # surface variables:
    # surface virtual potential temperature [K]
    thetavsurf: float
    # surface specific humidity [g kg-1]
    qsurf: float
    # turbulence:
    # Obukhov length [m]
    obukhov_length: float
    # bulk Richardson number [-]
    rib_number: float
    # aerodynamic resistance [s m-1]
    ra: float

    def __init__(
        self,
        ustar: float,
        z0m: float,
        z0h: float,
        theta: float,
    ):
        # surface friction velocity [m s-1]
        self.ustar = ustar
        # roughness length for momentum [m]
        self.z0m = z0m
        # roughness length for scalars [m]
        self.z0h = z0h
        # drag coefficient for momentum [-]
        self.drag_m = 1e12
        # drag coefficient for scalars [-]
        self.drag_s = 1e12
        # surface potential temperature [K]
        self.thetasurf = theta

    def _calculate_effective_wind_speed(
        self,
        u: float,
        v: float,
        wstar: float,
    ) -> float:
        """Calculate effective wind speed including convective effects."""
        return max(0.01, np.sqrt(u**2.0 + v**2.0 + wstar**2.0))

    def _calculate_surface_properties(
        self,
        ueff: float,
        theta: float,
        wtheta: float,
        q: float,
        surf_pressure: float,
        rs: float,
    ):
        """Calculate surface temperature and humidity."""
        self.thetasurf = theta + wtheta / (self.drag_s * ueff)
        qsatsurf = get_qsat(self.thetasurf, surf_pressure)
        cq = (1.0 + self.drag_s * ueff * rs) ** -1.0
        self.qsurf = (1.0 - cq) * q + cq * qsatsurf
        self.thetavsurf = self.thetasurf * (1.0 + 0.61 * self.qsurf)

    def _calculate_richardson_number(
        self, ueff: float, zsl: float, g: float, thetav: float
    ):
        """Calculate bulk Richardson number."""
        self.rib_number = g / thetav * zsl * (thetav - self.thetavsurf) / ueff**2.0
        self.rib_number = min(self.rib_number, 0.2)

    def _calculate_scalar_correction_term(self, zsl: float, oblen: float) -> float:
        """Calculate scalar stability correction term."""
        log_term = np.log(zsl / self.z0h)
        upper_stability = get_psih(zsl / oblen)
        surface_stability = get_psih(self.z0h / oblen)

        return log_term - upper_stability + surface_stability

    def _calculate_momentum_correction_term(self, zsl: float, oblen: float) -> float:
        """Calculate momentum stability correction term."""
        log_term = np.log(zsl / self.z0m)
        upper_stability = get_psim(zsl / oblen)
        surface_stability = get_psim(self.z0m / oblen)

        return log_term - upper_stability + surface_stability

    def _calculate_rib_function(self, zsl: float, oblen: float) -> float:
        """Calculate Richardson number function for iteration."""
        scalar_term = self._calculate_scalar_correction_term(zsl, oblen)
        momentum_term = self._calculate_momentum_correction_term(zsl, oblen)

        return self.rib_number - zsl / oblen * scalar_term / momentum_term**2.0

    def _calculate_rib_function_term(self, zsl: float, oblen: float) -> float:
        """Calculate function term for derivative calculation."""
        scalar_term = self._calculate_scalar_correction_term(zsl, oblen)
        momentum_term = self._calculate_momentum_correction_term(zsl, oblen)

        return -zsl / oblen * scalar_term / momentum_term**2.0

    def _ribtol(self, zsl: float):
        """Iterative solution for Obukhov length from Richardson number."""
        # initial guess based on stability
        oblen = 1.0 if self.rib_number > 0.0 else -1.0
        oblen0 = 2.0 if self.rib_number > 0.0 else -2.0

        convergence_threshold = 0.001
        perturbation = 0.001

        while abs(oblen - oblen0) > convergence_threshold:
            oblen0 = oblen

            # calculate function value at current estimate
            fx = self._calculate_rib_function(zsl, oblen)

            # calculate derivative using finite differences
            oblen_start = oblen - perturbation * oblen
            oblen_end = oblen + perturbation * oblen

            fx_start = self._calculate_rib_function_term(zsl, oblen_start)
            fx_end = self._calculate_rib_function_term(zsl, oblen_end)

            fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

            # Newton-Raphson update
            oblen = oblen - fx / fxdif

            # prevent runaway solutions
            if abs(oblen) > 1e15:
                break

        return oblen

    def _calculate_drag_coefficients(self, zsl: float, k: float):
        """Calculate drag coefficients with stability corrections."""
        # momentum stability correction
        momentum_correction = self._calculate_momentum_correction_term(
            zsl, self.obukhov_length
        )

        # scalar stability correction
        scalar_correction = self._calculate_scalar_correction_term(
            zsl, self.obukhov_length
        )

        # drag coefficients
        self.drag_m = k**2.0 / momentum_correction**2.0
        self.drag_s = k**2.0 / (momentum_correction * scalar_correction)

    def _calculate_momentum_fluxes(self, ueff: float, u: float, v: float):
        """Calculate momentum fluxes and friction velocity."""
        self.ustar = np.sqrt(self.drag_m) * ueff
        self.uw = -self.drag_m * ueff * u
        self.vw = -self.drag_m * ueff * v

    def _calculate_2m_variables(
        self,
        wtheta: float,
        wq: float,
        q: float,
        surf_pressure: float,
        k: float,
    ):
        """Calculate 2m diagnostic meteorological variables."""
        # stability correction terms
        scalar_correction = (
            np.log(2.0 / self.z0h)
            - get_psih(2.0 / self.obukhov_length)
            + get_psih(self.z0h / self.obukhov_length)
        )
        momentum_correction = (
            np.log(2.0 / self.z0m)
            - get_psim(2.0 / self.obukhov_length)
            + get_psim(self.z0m / self.obukhov_length)
        )

        # scaling factor for scalar fluxes
        scalar_scale = 1.0 / (self.ustar * k)
        momentum_scale = 1.0 / (self.ustar * k)

        # temperature and humidity at 2m
        self.temp_2m = self.thetasurf - wtheta * scalar_scale * scalar_correction
        self.q2m = self.qsurf - wq * scalar_scale * scalar_correction

        # wind components at 2m
        self.u2m = -self.uw * momentum_scale * momentum_correction
        self.v2m = -self.vw * momentum_scale * momentum_correction

        # vapor pressures at 2m
        # limamau: name these constants
        self.esat2m = 0.611e3 * np.exp(
            17.2694 * (self.temp_2m - 273.16) / (self.temp_2m - 35.86)
        )
        self.e2m = self.q2m * surf_pressure / 0.622

    def run(
        self,
        const: PhysicalConstants,
        land_surface: AbstractLandSurfaceModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Calculate surface layer turbulent exchange and diagnostic variables.

        Parameters
        ----------
        - ``const``: physical constants. Uses ``g`` and ``k``.
        - ``land_surface``: land surface model. Uses ``rs``.
        - ``mixed_layer``: mixed layer model. Uses wind components, fluxes, and thermodynamic variables.

        Updates
        -------
        Updates all surface layer variables including momentum fluxes, drag coefficients,
        Obukhov length, and 2m diagnostic meteorological variables.
        """
        ueff = self._calculate_effective_wind_speed(
            mixed_layer.u, mixed_layer.v, mixed_layer.wstar
        )
        self._calculate_surface_properties(
            ueff,
            mixed_layer.theta,
            mixed_layer.wtheta,
            mixed_layer.q,
            mixed_layer.surf_pressure,
            land_surface.rs,
        )

        zsl = 0.1 * mixed_layer.abl_height
        self._calculate_richardson_number(ueff, zsl, const.g, mixed_layer.thetav)

        # limamau: the following is rather slow
        # we can probably use a scan when JAX is on
        # before they had the option:
        # Fast C++ iteration
        # self.L    = ribtol.ribtol(self.Rib, zsl, self.z0m, self.z0h)
        # we could make this faster with a scan or something using jax
        self.obukhov_length = self._ribtol(zsl)

        self._calculate_drag_coefficients(zsl, const.k)
        self._calculate_momentum_fluxes(ueff, mixed_layer.u, mixed_layer.v)
        self._calculate_2m_variables(
            mixed_layer.wtheta,
            mixed_layer.wq,
            mixed_layer.q,
            mixed_layer.surf_pressure,
            const.k,
        )

    def compute_ra(self, u: float, v: float, wstar: float):
        """Calculate aerodynamic resistance from wind speed and drag coefficient."""
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        self.ra = (self.drag_s * ueff) ** -1.0
