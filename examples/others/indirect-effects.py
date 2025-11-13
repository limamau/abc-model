import jax.numpy as jnp
import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import compute_esat


def run_wrapper(wg: float, q: float):
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
    ml_kwargs["q"] = q
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

    return abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)


def make_fancy_plot(
    axes,
    time: jnp.ndarray,
    traj,
    color: str,
    marker,
    label: str,
    factor: int = 60,
):
    # mixed layer
    axes[0, 0].plot(
        time[::factor],
        traj.abl_height[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 0].set_title("h [m]")
    axes[0, 1].plot(
        time[::factor],
        traj.wCO2A[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 1].set_title("wCO2A [mgC/m²/s]")
    axes[0, 2].plot(
        time[::factor],
        traj.wCO2R[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 2].set_title("wCO2R [mgC/m²/s]")
    axes[0, 3].plot(
        time[::factor],
        traj.wCO2[::factor],
        label=label,
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[0, 3].set_title("wCO2 [mgC/m²/s]")
    axes[0, 3].legend()

    # temperature
    axes[1, 0].plot(
        time[::factor],
        traj.temp_soil[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 0].set_title("temp soil [K]")
    axes[1, 1].plot(
        time[::factor],
        traj.surf_temp[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 1].set_title("surf temp [K]")
    axes[1, 2].plot(
        time[::factor],
        traj.theta[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 2].set_title("theta [K]")
    axes[1, 3].plot(
        time[::factor],
        traj.temp_2m[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[1, 3].set_title("temp 2m [K]")

    # water
    # this is the core of this example! #
    vpd = (compute_esat(traj.surf_temp) - traj.e) / 1000.0  # kPa
    axes[2, 0].plot(
        time[::factor], vpd[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[2, 0].set_title("VPD [kPa]")
    axes[2, 1].plot(
        time[::factor], traj.wg[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[2, 1].set_title("wg [kg/kg]")
    # - - - - - - - - - - - - - - - - - #
    axes[2, 2].plot(
        time[::factor], traj.q[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[2, 2].set_title("q [kg/kg]")
    axes[2, 3].plot(
        time[::factor], traj.w2[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[2, 3].set_title("w2 [kg/kg]")

    # energy fluxes
    axes[3, 0].plot(
        time[::factor], traj.hf[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[3, 0].set_title("H [W/m²]")
    axes[3, 1].plot(
        time[::factor], traj.le[::factor], color=color, marker=marker, linestyle="None"
    )
    axes[3, 1].set_title("LE [W/m²]")
    axes[3, 2].plot(
        time[::factor],
        traj.le_veg[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 2].set_title("LEveg [W/m²]")
    axes[3, 3].plot(
        time[::factor],
        traj.le_liq[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[3, 3].set_title("LEliq [W/m²]")

    # radiation
    axes[4, 0].plot(
        time[::factor],
        traj.in_srad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 0].set_title("SWin [W/m²]")
    axes[4, 1].plot(
        time[::factor],
        traj.out_srad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 1].set_title("SWout [W/m²]")
    axes[4, 2].plot(
        time[::factor],
        traj.in_lrad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 2].set_title("LWin [W/m²]")
    axes[4, 3].plot(
        time[::factor],
        traj.out_lrad[::factor],
        color=color,
        marker=marker,
        linestyle="None",
    )
    axes[4, 3].set_title("LWout [W/m²]")


def main():
    wg_control = 0.21
    delta_wg = 0.1
    q_control = 0.008
    delta_q = 0.004

    # plot
    _, axes = plt.subplots(5, 4, figsize=(18, 9))

    # add experiments:
    # control
    time, control_traj = run_wrapper(wg_control, q_control)
    make_fancy_plot(axes, time, control_traj, "gray", "o", "control")
    # positive soil moisture anomaly
    time, pos_soil_moist_traj = run_wrapper(wg_control + delta_wg, q_control)
    make_fancy_plot(axes, time, pos_soil_moist_traj, "dodgerblue", "+", "+|ΔSM|")
    # negative soil moisture anomaly
    time, neg_soil_moist_traj = run_wrapper(wg_control - delta_wg, q_control)
    make_fancy_plot(axes, time, neg_soil_moist_traj, "dodgerblue", "_", "-|ΔSM|")
    # positive specific humidity anomaly
    time, pos_surf_temp_traj = run_wrapper(wg_control, q_control + delta_q)
    make_fancy_plot(axes, time, pos_surf_temp_traj, "orangered", "+", "+|Δq|")
    # negative specific humidity anomaly
    time, neg_surf_temp_traj = run_wrapper(wg_control, q_control - delta_q)
    make_fancy_plot(axes, time, neg_surf_temp_traj, "orangered", "_", "-|Δq|")

    # names on each row
    row_names = [
        "Mixed Layer",
        "Temperature",
        "Water",
        "Energy Fluxes",
        "Radiation",
    ]
    for i, name in enumerate(row_names):
        ax = axes[i, 0]
        ax.set_ylabel(
            name,
            fontweight="bold",
        )

    # only put x-axis labels and ticks on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [h]")
    for ax in axes[:-1, :].flatten():
        ax.set_xticks([])

    # leave space for row labels
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
