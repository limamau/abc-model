import jax.numpy as jnp

import abcconfigs.class_model as cm
import abcmodel


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # define radiation model
    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    land_surface_init_conds = abcmodel.land.MinimalLandSurfaceInitConds(
        alpha=jnp.array(0.25),
        surf_temp=jnp.array(288.8),
        rs=jnp.array(1.0),
    )
    land_surface_model = abcmodel.land.MinimalLandSurfaceModel()

    # surface layer
    surface_layer_init_conds = (
        abcmodel.atmosphere.surface_layer.ObukhovSurfaceLayerInitConds(
            **cm.obukhov_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmosphere.surface_layer.ObukhovSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.atmosphere.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.atmosphere.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.atmosphere.clouds.CumulusInitConds()
    cloud_model = abcmodel.atmosphere.clouds.CumulusModel()

    # define atmosphere model
    atmosphere_model = abcmodel.atmosphere.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        radiation=radiation_model,
        land=land_surface_model,
        atmosphere=atmosphere_model,
    )
    atmosphere_state = abcmodel.atmosphere.DayOnlyAtmosphereState(
        surface_layer=surface_layer_init_conds,
        mixed_layer=mixed_layer_init_conds,
        clouds=cloud_init_conds,
    )
    state = abcoupler.init_state(
        radiation_init_conds,
        land_surface_init_conds,
        atmosphere_state,
    )

    # run run run
    abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)


if __name__ == "__main__":
    main()
