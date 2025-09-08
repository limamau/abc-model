import numpy as np

from ..diagnostics import AbstractDiagnostics
from ..models import (
    AbstractInitConds,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractParams,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_psih, get_psim, get_qsat


class StandardSurfaceLayerParams(AbstractParams["StandardSurfaceLayerModel"]):
    """Data class for standard surface layer model parameters.

    Arguments
    ---------
    - ``ustar``: surface friction velocity [m/s].
    - ``z0m``: roughness length for momentum [m].
    - ``z0h``: roughness length for scalars [m].
    - ``theta``: surface potential temperature [K].
    """

    def __init__(self):
        pass


class StandardSurfaceLayerInitConds(AbstractInitConds["StandardSurfaceLayerModel"]):
    """Data class for standard surface layer model initial conditions.

    Arguments
    ---------
    - ``ustar``: surface friction velocity [m/s].
    - ``z0m``: roughness length for momentum [m].
    - ``z0h``: roughness length for scalars [m].
    - ``theta``: surface potential temperature [K].

    Extra
    -----
    The following parameters are initialized to high values
    and are expected to converge to realistic values during warmup.
    - ``drag_m``: drag coefficient for momentum [-].
    - ``drag_s``: drag coefficient for scalars [-].
    """

    def __init__(
        self,
        ustar: float,
        z0m: float,
        z0h: float,
        theta: float,
    ):
        self.ustar = ustar
        self.z0m = z0m
        self.z0h = z0h
        self.theta = theta
        self.drag_m = 1e12
        self.drag_s = 1e12


class StandardSurfaceLayerDiagnostics(AbstractDiagnostics["StandardSurfaceLayerModel"]):
    """Class for standard surface layer model diagnostic variables.

    Variables
    ---------
    - ``uw``: # surface momentum flux u [m2 s-2].
    - ``vw``: # surface momentum flux v [m2 s-2].
    - ``temp_2m``: # 2m temperature [K].
    - ``q2m``: # 2m specific humidity [kg kg-1].
    - ``u2m``: 2m u-wind [m s-1].
    - ``v2m``: 2m v-wind [m s-1].
    - ``e2m``: 2m vapor pressure [Pa].
    - ``esat2m``: 2m saturated vapor pressure [Pa].
    - ``thetasurf``: surface potential temperature [K].
    - ``thetavsurf``: surface virtual potential temperature [K].
    - ``qsurf``: surface specific humidity [kg kg-1].
    - ``ustar``: surface friction velocity [m s-1].
    - ``z0m``: roughness length for momentum [m]
    - ``z0h``: roughness length for scalars [m].
    - ``drag_m``: drag coefficient for momentum [-].
    - ``drag_s``: drag coefficient for scalars [-].
    - ``obukhov_length``: Obukhov length [m].
    - ``rib_number``: bulk Richardson number [-].
    """

    def post_init(self, tsteps: int):
        self.uw = np.zeros(tsteps)
        self.vw = np.zeros(tsteps)
        self.temp_2m = np.zeros(tsteps)
        self.q2m = np.zeros(tsteps)
        self.u2m = np.zeros(tsteps)
        self.v2m = np.zeros(tsteps)
        self.e2m = np.zeros(tsteps)
        self.esat2m = np.zeros(tsteps)
        self.thetasurf = np.zeros(tsteps)
        self.thetavsurf = np.zeros(tsteps)
        self.qsurf = np.zeros(tsteps)
        self.ustar = np.zeros(tsteps)
        self.drag_m = np.zeros(tsteps)
        self.drag_s = np.zeros(tsteps)
        self.obukhov_length = np.zeros(tsteps)
        self.rib_number = np.zeros(tsteps)

    def store(self, t: int, model: "StandardSurfaceLayerModel"):
        self.uw[t] = model.uw
        self.vw[t] = model.vw
        self.temp_2m[t] = model.temp_2m
        self.q2m[t] = model.q2m
        self.u2m[t] = model.u2m
        self.v2m[t] = model.v2m
        self.e2m[t] = model.e2m
        self.esat2m[t] = model.esat2m
        self.thetasurf[t] = model.thetasurf
        self.thetavsurf[t] = model.thetavsurf
        self.qsurf[t] = model.qsurf
        self.ustar[t] = model.ustar
        self.drag_m[t] = model.drag_m
        self.drag_s[t] = model.drag_s
        self.obukhov_length[t] = model.obukhov_length
        self.rib_number[t] = model.rib_number


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    """Standard surface layer model with atmospheric stability corrections.

    Calculates surface-atmosphere exchange using Monin-Obukhov similarity theory
    with stability functions and iterative solution for Obukhov length.

    Processes
    ---------
    1. Calculate effective wind speed and surface properties.
    2. Determine bulk Richardson number and solve for Obukhov length.
    3. Compute drag coefficients with stability corrections.
    4. Calculate momentum fluxes and 2m diagnostic variables.
    """

    def __init__(
        self,
        params: StandardSurfaceLayerParams,
        init_conds: StandardSurfaceLayerInitConds,
        diagnostics: AbstractDiagnostics = StandardSurfaceLayerDiagnostics(),
    ):
        self.ustar = init_conds.ustar
        self.z0m = init_conds.z0m
        self.z0h = init_conds.z0h
        self.theta = init_conds.theta
        self.drag_m = init_conds.drag_m
        self.drag_s = init_conds.drag_s
        self.diagnostics = diagnostics

    def calculate_effective_wind_speed(
        self,
        u: float,
        v: float,
        wstar: float,
    ) -> float:
        """Calculate effective wind speed including convective effects."""
        return max(0.01, np.sqrt(u**2.0 + v**2.0 + wstar**2.0))

    def calculate_surface_properties(
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

    def calculate_richardson_number(
        self, ueff: float, zsl: float, g: float, thetav: float
    ):
        """Calculate bulk Richardson number."""
        self.rib_number = g / thetav * zsl * (thetav - self.thetavsurf) / ueff**2.0
        self.rib_number = min(self.rib_number, 0.2)

    def calculate_scalar_correction_term(self, zsl: float, oblen: float) -> float:
        """Calculate scalar stability correction term."""
        log_term = np.log(zsl / self.z0h)
        upper_stability = get_psih(zsl / oblen)
        surface_stability = get_psih(self.z0h / oblen)

        return log_term - upper_stability + surface_stability

    def calculate_momentum_correction_term(self, zsl: float, oblen: float) -> float:
        """Calculate momentum stability correction term."""
        log_term = np.log(zsl / self.z0m)
        upper_stability = get_psim(zsl / oblen)
        surface_stability = get_psim(self.z0m / oblen)

        return log_term - upper_stability + surface_stability

    def calculate_rib_function(self, zsl: float, oblen: float) -> float:
        """Calculate Richardson number function for iteration."""
        scalar_term = self.calculate_scalar_correction_term(zsl, oblen)
        momentum_term = self.calculate_momentum_correction_term(zsl, oblen)

        return self.rib_number - zsl / oblen * scalar_term / momentum_term**2.0

    def calculate_rib_function_term(self, zsl: float, oblen: float) -> float:
        """Calculate function term for derivative calculation."""
        scalar_term = self.calculate_scalar_correction_term(zsl, oblen)
        momentum_term = self.calculate_momentum_correction_term(zsl, oblen)

        return -zsl / oblen * scalar_term / momentum_term**2.0

    def ribtol(self, zsl: float):
        """Iterative solution for Obukhov length from Richardson number."""
        # initial guess based on stability
        oblen = 1.0 if self.rib_number > 0.0 else -1.0
        oblen0 = 2.0 if self.rib_number > 0.0 else -2.0

        convergence_threshold = 0.001
        perturbation = 0.001

        while abs(oblen - oblen0) > convergence_threshold:
            oblen0 = oblen

            # calculate function value at current estimate
            fx = self.calculate_rib_function(zsl, oblen)

            # calculate derivative using finite differences
            oblen_start = oblen - perturbation * oblen
            oblen_end = oblen + perturbation * oblen

            fx_start = self.calculate_rib_function_term(zsl, oblen_start)
            fx_end = self.calculate_rib_function_term(zsl, oblen_end)

            fxdif = (fx_start - fx_end) / (oblen_start - oblen_end)

            # Newton-Raphson update
            oblen = oblen - fx / fxdif

            # prevent runaway solutions
            if abs(oblen) > 1e15:
                break

        return oblen

    def calculate_drag_coefficients(self, zsl: float, k: float):
        """Calculate drag coefficients with stability corrections."""
        # momentum stability correction
        momentum_correction = self.calculate_momentum_correction_term(
            zsl, self.obukhov_length
        )

        # scalar stability correction
        scalar_correction = self.calculate_scalar_correction_term(
            zsl, self.obukhov_length
        )

        # drag coefficients
        self.drag_m = k**2.0 / momentum_correction**2.0
        self.drag_s = k**2.0 / (momentum_correction * scalar_correction)

    def calculate_momentum_fluxes(self, ueff: float, u: float, v: float):
        """Calculate momentum fluxes and friction velocity."""
        self.ustar = np.sqrt(self.drag_m) * ueff
        self.uw = -self.drag_m * ueff * u
        self.vw = -self.drag_m * ueff * v

    def calculate_2m_variables(
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
        ueff = self.calculate_effective_wind_speed(
            mixed_layer.u, mixed_layer.v, mixed_layer.wstar
        )
        self.calculate_surface_properties(
            ueff,
            mixed_layer.theta,
            mixed_layer.wtheta,
            mixed_layer.q,
            mixed_layer.surf_pressure,
            land_surface.rs,
        )

        zsl = 0.1 * mixed_layer.abl_height
        self.calculate_richardson_number(ueff, zsl, const.g, mixed_layer.thetav)

        # limamau: the following is rather slow
        # we can probably use a scan when JAX is on
        # before they had the option:
        # Fast C++ iteration
        # self.L    = ribtol.ribtol(self.Rib, zsl, self.z0m, self.z0h)
        # we could make this faster with a scan or something using jax
        self.obukhov_length = self.ribtol(zsl)

        self.calculate_drag_coefficients(zsl, const.k)
        self.calculate_momentum_fluxes(ueff, mixed_layer.u, mixed_layer.v)
        self.calculate_2m_variables(
            mixed_layer.wtheta,
            mixed_layer.wq,
            mixed_layer.q,
            mixed_layer.surf_pressure,
            const.k,
        )

    def compute_ra(self, u: float, v: float, wstar: float):
        """Calculate aerodynamic resistance from wind speed and drag coefficient."""
        # limamau: we should probably move this variable to land surface model class
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        return (self.drag_s * ueff) ** -1.0
