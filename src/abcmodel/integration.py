from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from .coupling import ABCoupler, CoupledState


def warmup(state: CoupledState, coupler: ABCoupler, t: int, dt: float) -> CoupledState:
    """Warmup the model by running it for a few timesteps."""
    state = replace(
        state,
        atmosphere=coupler.atmosphere.statistics(state.atmosphere, t, coupler.const),
    )
    state = replace(state, radiation=coupler.radiation.run(state, t, dt, coupler.const))
    state = coupler.atmosphere.warmup(state, coupler.const, coupler.land)
    return state


def timestep(
    state: CoupledState, coupler: ABCoupler, t: int, dt: float
) -> CoupledState:
    """Run a single timestep of the model."""
    atmos = coupler.atmosphere.statistics(state.atmosphere, t, coupler.const)
    state = replace(state, atmosphere=atmos)
    rad = coupler.radiation.run(state, t, dt, coupler.const)
    state = replace(state, radiation=rad)
    land = coupler.land.run(state, coupler.const)
    state = replace(state, land=land)
    atmos = coupler.atmosphere.run(state, coupler.const)
    state = replace(state, atmosphere=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = replace(state, land=land)
    atmos = coupler.atmosphere.integrate(state.atmosphere, dt)
    state = replace(state, atmosphere=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(state: CoupledState, coupler: ABCoupler, dt: float, runtime: float):
    """Integrate the coupler forward in time.

    Args:
        state: Initial coupled state.
        coupler: ABCoupler instance.
        dt: Time step [s].
        runtime: Total runtime [s].

    Returns:
        times: Array of time values [h].
        trajectory: CoupledState containing the full state trajectory.
    """
    tsteps = int(np.floor(runtime / dt))

    # warmup
    state = warmup(state, coupler, 0, dt)
    state = coupler.compute_diagnostics(state)

    def iter_fn(state, t):
        state = timestep(state, coupler, t, dt)
        return state, state

    timesteps = jnp.arange(tsteps)
    state, trajectory = jax.lax.scan(iter_fn, state, timesteps, length=tsteps)

    times = jnp.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart

    return times, trajectory
