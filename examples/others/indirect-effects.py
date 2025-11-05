import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import get_esat

N_PERTURB = 10
SEED = 42
WG = 0.1
Q = 0.008
SURF_TEMP = 290.0
MULT_STD = 10.0


def run_wrapper(wg, surf_temp):
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # radiation with clouds
    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    ls_kwargs = cm.aquacrop.init_conds_kwargs
    ls_kwargs["wg"] = wg
    ls_kwargs["surf_temp"] = surf_temp
    land_surface_init_conds = abcmodel.land_surface.AquaCropInitConds(
        **ls_kwargs,
    )
    land_surface_model = abcmodel.land_surface.AquaCropModel(
        **cm.aquacrop.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = abcmodel.surface_layer.StandardSurfaceLayerInitConds(
        **cm.standard_surface_layer.init_conds_kwargs
    )
    surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel()

    # mixed layer
    ml_kwargs = cm.bulk_mixed_layer.init_conds_kwargs
    # ml_kwargs["q"] = q
    mixed_layer_init_conds = abcmodel.mixed_layer.BulkMixedLayerInitConds(
        **ml_kwargs,
    )
    mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.clouds.StandardCumulusInitConds()
    cloud_model = abcmodel.clouds.StandardCumulusModel()

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        radiation=radiation_model,
        land_surface=land_surface_model,
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    state = abcoupler.init_state(
        radiation_init_conds,
        land_surface_init_conds,
        surface_layer_init_conds,
        mixed_layer_init_conds,
        cloud_init_conds,
    )

    # run run run
    return abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)


def main():
    # control run
    time, control_traj = run_wrapper(WG, SURF_TEMP)

    # plot output
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, control_traj.wg, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("wg [m3 m-3]")

    plt.subplot(234)
    plt.plot(time, control_traj.w2, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("w2 [m3 m-3]")

    plt.subplot(232)
    plt.plot(time, control_traj.surf_temp, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("surf_temp [K]")

    plt.subplot(235)
    plt.plot(time, control_traj.thetasurf, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("thetasurf [K]")

    plt.subplot(233)
    plt.plot(time, get_esat(control_traj.surf_temp) - control_traj.e, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("VPD [Pa]")

    plt.subplot(236)
    plt.plot(time, control_traj.wCO2, color="black")
    plt.xlabel("time [h]")
    plt.ylabel("wCO2 [mgC m-2 s-1]")

    plt.tight_layout()
    plt.show()
    plt.close()

    # wg perturbed run
    key = jr.PRNGKey(SEED)
    key, wg_perturb = jr.split(key)
    wg_std = control_traj.wg.std()
    print("wg std:", wg_std)
    wg_perturb = jr.normal(shape=(N_PERTURB,), key=key) * wg_std * MULT_STD
    _, wg_traj = jax.block_until_ready(
        jax.vmap(run_wrapper)(WG + wg_perturb, jnp.repeat(Q, N_PERTURB))
    )

    # surf_temp perturbed run
    key = jr.PRNGKey(SEED)
    key, surf_temp_perturb = jr.split(key)
    surf_temp_std = control_traj.surf_temp.std()
    print("surf_temp std:", surf_temp_std)
    surf_temp_perturb = (
        jr.normal(shape=(N_PERTURB,), key=key) * surf_temp_std * MULT_STD
    )
    _, surf_temp_traj = jax.block_until_ready(
        jax.vmap(run_wrapper)(jnp.repeat(WG, N_PERTURB), SURF_TEMP + surf_temp_perturb)
    )

    # # q perturbed run
    # key = jr.PRNGKey(SEED)
    # key, q_perturb = jr.split(key)
    # q_std = control_traj.q.std()
    # print("q std:", q_std)
    # q_perturb = jr.normal(shape=(N_PERTURB,), key=key) * q_std * 10
    # _, q_traj = jax.vmap(run_wrapper)(jnp.repeat(WG, N_PERTURB), Q + q_perturb)

    # plot output
    print("plotting...")
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, control_traj.wg, color="black")
    for i in range(N_PERTURB):
        plt.plot(time, wg_traj.wg[i], color="blue", alpha=0.1)
        plt.plot(time, surf_temp_traj.wg[i], color="red", alpha=0.1)
    plt.xlabel("time [h]")
    plt.ylabel("wg [m3 m-3]")

    plt.subplot(234)
    plt.plot(time, control_traj.w2, color="black")
    for i in range(N_PERTURB):
        plt.plot(time, wg_traj.w2[i], color="blue", alpha=0.1)
        plt.plot(time, surf_temp_traj.w2[i], color="red", alpha=0.1)
    plt.xlabel("time [h]")
    plt.ylabel("w2 [m3 m-3]")

    plt.subplot(232)
    plt.plot(time, control_traj.surf_temp, color="black")
    for i in range(N_PERTURB):
        plt.plot(time, wg_traj.surf_temp[i], color="blue", alpha=0.1)
        plt.plot(time, surf_temp_traj.surf_temp[i], color="red", alpha=0.1)
    plt.xlabel("time [h]")
    plt.ylabel("surf_temp [K]")

    plt.subplot(235)
    plt.plot(time, control_traj.thetasurf, color="black", label="control run")
    for i in range(N_PERTURB):
        plt.plot(
            time, wg_traj.thetasurf[i], color="blue", alpha=0.1, label="wg perturb. run"
        )
        plt.plot(
            time,
            surf_temp_traj.thetasurf[i],
            color="red",
            alpha=0.1,
            label="surf_temp perturb. run",
        )
    plt.xlabel("time [h]")
    plt.ylabel("thetasurf [K]")

    plt.subplot(233)
    plt.plot(time, get_esat(control_traj.surf_temp) - control_traj.e, color="black")
    for i in range(N_PERTURB):
        plt.plot(
            time, get_esat(wg_traj.surf_temp[i]) - wg_traj.e[i], color="blue", alpha=0.1
        )
        plt.plot(
            time,
            get_esat(surf_temp_traj.surf_temp[i]) - surf_temp_traj.e[i],
            color="red",
            alpha=0.1,
        )
    plt.xlabel("time [h]")
    plt.ylabel("VPD [Pa]")

    plt.subplot(236)
    plt.plot(time, control_traj.wCO2, color="black")
    for i in range(N_PERTURB):
        plt.plot(time, wg_traj.wCO2[i], color="blue", alpha=0.1)
        plt.plot(time, surf_temp_traj.wCO2[i], color="red", alpha=0.1)
    plt.xlabel("time [h]")
    plt.ylabel("wCO2 [mgC m-2 s-1]")

    plt.tight_layout()
    plt.show()

    # effect timeseires
    effect_wg = wg_traj.wCO2 - control_traj.wCO2
    rmsd_wg = jnp.sqrt(jnp.mean(effect_wg**2))
    effect_surf_temp = surf_temp_traj.wCO2 - control_traj.wCO2
    rmsd_surf_temp = jnp.sqrt(jnp.mean(effect_surf_temp**2))

    print(f"RMSD of wg effect: {rmsd_wg:.4f}")
    print(f"RMSD of surf_temp effect: {rmsd_surf_temp:.4f}")

    # relative importances
    total_effect_rmsd = rmsd_wg + rmsd_surf_temp
    if total_effect_rmsd == 0:
        print("\nRelative Importances:")
        print("wg: 0% (no effect detected)")
        print("surf_temp: 0% (no effect detected)")
    else:
        rel_importance_wg = rmsd_wg / total_effect_rmsd
        rel_importance_surf_temp = rmsd_surf_temp / total_effect_rmsd

        print("\nRelative Importances (based on RMSD):")
        print(f"wg: {rel_importance_wg:.2%}")
        print(f"surf_temp: {rel_importance_surf_temp:.2%}")


if __name__ == "__main__":
    main()
