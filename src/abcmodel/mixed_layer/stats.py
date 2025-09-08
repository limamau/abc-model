from ..models import (
    AbstractMixedLayerModel,
)
from ..utils import PhysicalConstants, get_qsat


class AbstractStandardStatsModel(AbstractMixedLayerModel):
    """Abstract base class for mixed layer models with standard meteorological statistics.

    Provides a common calculation method for virtual temperature, mixed-layer top
    properties, and lifting condensation level determination.

    Updates
    --------
    - ``thetav``, ``wthetav``, ``dthetav``: virtual temperature variables.
    - ``top_p``, ``top_T``, ``top_rh``: mixed-layer top properties.
    - ``lcl``: lifting condensation level height [m].
    """

    def statistics(self, t: float, const: PhysicalConstants):
        """Calculate standard meteorological statistics and diagnostics.

        Parameters
        ----------
        - ``t``: current time step.
        - ``const``: physical constants. Uses ``rho``, ``g``, and ``cp``.

        Updates
        -------
        Updates virtual temperatures, mixed-layer top properties, and lifting
        condensation level based on current thermodynamic state.
        """
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
