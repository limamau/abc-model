from abc import abstractmethod

import numpy as np

from .utils import PhysicalConstants, get_psih, get_psim, get_qsat, get_ribtol


class AbstractSurfaceLayerModel:
    def __init__(
        self,
        ustar: float,
        z0m: float,
        z0h: float,
    ):
        # constants
        self.const = PhysicalConstants()
        # surface friction velocity [m s-1]
        self.ustar = ustar
        # roughness length for momentum [m]
        self.z0m = z0m
        # roughness length for scalars [m]
        self.z0h = z0h
        # surface momentum flux in u-direction [m2 s-2]
        self.uw = None
        # surface momentum flux in v-direction [m2 s-2]
        self.vw = None
        # drag coefficient for momentum [-]
        self.drag_m = 1e12
        # drag coefficient for scalars [-]
        self.drag_s = 1e12
        # Obukhov length [m]
        self.obukhov_length = None
        # bulk Richardson number [-]
        self.rib_number = None
        # aerodynamic resistance [s m-1]
        self.ra = None

    @abstractmethod
    def run(
        self,
        u: float,
        v: float,
        theta: float,
        thetav: float,
        wstar: float,
        wtheta: float,
        wq: float,
        surf_pressure: float,
        rs: float,
        q: float,
        abl_height: float,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_ra(self, u: float, v: float, wstar: float):
        raise NotImplementedError


class InertSurfaceLayerModel(AbstractSurfaceLayerModel):
    def run(
        self,
        u: float,
        v: float,
        theta: float,
        thetav: float,
        wstar: float,
        wtheta: float,
        wq: float,
        surf_pressure: float,
        rs: float,
        q: float,
        abl_height: float,
    ):
        self.uw = -np.sign(u) * (self.ustar**4.0 / (v**2.0 / u**2.0 + 1.0)) ** (0.5)
        self.vw = -np.sign(v) * (self.ustar**4.0 / (u**2.0 / v**2.0 + 1.0)) ** (0.5)

    def compute_ra(self, u: float, v: float, wstar: float):
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        self.ra = ueff / max(1.0e-3, self.ustar) ** 2.0


class StandardSurfaceLayerModel(AbstractSurfaceLayerModel):
    def run(
        self,
        u: float,
        v: float,
        theta: float,
        thetav: float,
        wstar: float,
        wtheta: float,
        wq: float,
        surf_pressure: float,
        rs: float,
        q: float,
        abl_height: float,
    ):
        ueff = max(0.01, np.sqrt(u**2.0 + v**2.0 + wstar**2.0))
        self.thetasurf = theta + wtheta / (self.drag_s * ueff)
        qsatsurf = get_qsat(self.thetasurf, surf_pressure)
        cq = (1.0 + self.drag_s * ueff * rs) ** -1.0
        self.qsurf = (1.0 - cq) * q + cq * qsatsurf

        self.thetavsurf = self.thetasurf * (1.0 + 0.61 * self.qsurf)

        zsl = 0.1 * abl_height
        self.rib_number = (
            self.const.g / thetav * zsl * (thetav - self.thetavsurf) / ueff**2.0
        )
        self.rib_number = min(self.rib_number, 0.2)

        self.obukhov_length = get_ribtol(
            self.rib_number,
            zsl,
            self.z0m,
            self.z0h,
        )  # Slow python iteration
        # self.L    = ribtol.ribtol(self.Rib, zsl, self.z0m, self.z0h) # Fast C++ iteration

        self.drag_m = (
            self.const.k**2.0
            / (
                np.log(zsl / self.z0m)
                - get_psim(zsl / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
            ** 2.0
        )
        self.drag_s = (
            self.const.k**2.0
            / (
                np.log(zsl / self.z0m)
                - get_psim(zsl / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
            / (
                np.log(zsl / self.z0h)
                - get_psih(zsl / self.obukhov_length)
                + get_psih(self.z0h / self.obukhov_length)
            )
        )

        self.ustar = np.sqrt(self.drag_m) * ueff
        self.uw = -self.drag_m * ueff * u
        self.vw = -self.drag_m * ueff * v

        # diagnostic meteorological variables
        self.temp_2m = self.thetasurf - wtheta / self.ustar / self.const.k * (
            np.log(2.0 / self.z0h)
            - get_psih(2.0 / self.obukhov_length)
            + get_psih(self.z0h / self.obukhov_length)
        )
        self.q2m = self.qsurf - wq / self.ustar / self.const.k * (
            np.log(2.0 / self.z0h)
            - get_psih(2.0 / self.obukhov_length)
            + get_psih(self.z0h / self.obukhov_length)
        )
        self.u2m = (
            -self.uw
            / self.ustar
            / self.const.k
            * (
                np.log(2.0 / self.z0m)
                - get_psim(2.0 / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
        )
        self.v2m = (
            -self.vw
            / self.ustar
            / self.const.k
            * (
                np.log(2.0 / self.z0m)
                - get_psim(2.0 / self.obukhov_length)
                + get_psim(self.z0m / self.obukhov_length)
            )
        )
        self.esat2m = 0.611e3 * np.exp(
            17.2694 * (self.temp_2m - 273.16) / (self.temp_2m - 35.86)
        )
        self.e2m = self.q2m * surf_pressure / 0.622

    def compute_ra(self, u: float, v: float, wstar: float):
        ueff = np.sqrt(u**2.0 + v**2.0 + wstar**2.0)
        self.ra = (self.drag_s * ueff) ** -1.0
