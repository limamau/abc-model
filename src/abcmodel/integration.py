import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .abstracts import AbstractCoupledState, AtmosT, LandT, RadT
from .coupling import ABCoupler


def warmup(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
    tstart: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmos.warmup(coupler.rad, coupler.land, state, t, dt, tstart)
    return state


def timestep(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
    tstart: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Run a single timestep of the model."""
    atmos = coupler.atmos.statistics(state, t)
    state = state.replace(atmos=atmos)
    rad = coupler.rad.run(state, t, dt, tstart)
    state = state.replace(rad=rad)
    land = coupler.land.run(state)
    state = state.replace(land=land)
    atmos = coupler.atmos.run(state)
    state = state.replace(atmos=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = state.replace(land=land)
    atmos = coupler.atmos.integrate(state.atmos, dt)
    state = state.replace(atmos=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    inner_dt: float,
    outter_dt: float,
    runtime: float,
    tstart: float,
) -> tuple[Array, AbstractCoupledState[RadT, LandT, AtmosT]]:
    """Integrate the coupler forward in time.

    Args:
        state: Initial coupled state.
        coupler: ABCoupler instance.
        inner_dt: Inner time step (the one used to run the model) [s].
        outter_dt: Outer time step (the one used to save diagnostics) [s].
        runtime: Total runtime [s].

    Returns:
        times: Array of time values [h].
        trajectory: CoupledState containing the full state trajectory.
    """
    inner_tsteps = int(np.floor(outter_dt / inner_dt))
    outter_tsteps = int(np.floor(runtime / outter_dt))

    state = warmup(state, coupler, 0, inner_dt, tstart)
    state = coupler.compute_diagnostics(state)

    def inner_step_fn(state, t):
        state = timestep(state, coupler, t, inner_dt, tstart)
        return state, state

    def outter_step_fn(state, t):
        timesteps = t + jnp.arange(inner_tsteps)
        state, inner_traj = jax.lax.scan(
            inner_step_fn, state, timesteps, length=inner_tsteps
        )
        avg_traj = jax.tree.map(lambda x: jnp.mean(x, axis=0), inner_traj)
        return state, avg_traj

    timesteps = jnp.arange(outter_tsteps) * inner_tsteps
    state, trajectory = jax.lax.scan(
        outter_step_fn, state, timesteps, length=outter_tsteps
    )

    times = jnp.arange(outter_tsteps) * outter_dt / 3600.0 + tstart

    return times, trajectory
