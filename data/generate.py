import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import abcconfigs.class_model as cm
import abcmodel


def get_perturbed_configs(seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Perturb initial conditions and parameters
    h_abl = np.random.uniform(50, 400)
    theta = np.random.uniform(280, 295)
    # Correlation between soil temp and air temp
    temp_soil = theta + np.random.uniform(-2, 2)
    
    q = np.random.uniform(0.004, 0.012)
    deltatheta = np.random.uniform(0.5, 2.0)
    
    # Winds
    u = np.random.uniform(2, 12)
    v = np.random.uniform(-5, 5)
    
    # Soil
    wg = np.random.uniform(0.171, 0.35)
    d1 = np.random.uniform(0.05, 0.2)
    
    # Cloud cover for radiation
    cc = np.random.uniform(0.0, 0.5)

    # Prepare kwargs
    rad_model_kwargs = cm.standard_radiation.model_kwargs.copy()
    rad_model_kwargs["cc"] = cc
    
    ags_state_kwargs = cm.ags.state_kwargs.copy()
    ags_state_kwargs["wg"] = wg
    ags_state_kwargs["temp_soil"] = temp_soil
    
    ags_model_kwargs = cm.ags.model_kwargs.copy()
    ags_model_kwargs["d1"] = d1

    mixed_state_kwargs = cm.bulk_mixed_layer.state_kwargs.copy()
    mixed_state_kwargs.update({
        "h_abl": h_abl,
        "theta": theta,
        "deltatheta": deltatheta,
        "q": q,
        "u": u,
        "v": v,
    })

    return {
        "rad_model_kwargs": rad_model_kwargs,
        "ags_state_kwargs": ags_state_kwargs,
        "ags_model_kwargs": ags_model_kwargs,
        "mixed_state_kwargs": mixed_state_kwargs,
    }


def main():
    NUM_TRAJS = 1000
    # time step [s]
    dt = 15.0
    # total run time [s]
    runtime = 12 * 3600.0
    # start time of the day [h]
    tstart = 6.8

    # Store results
    # Time dimension will be determined after first run
    times = None
    
    results = {
        "h_abl": [],
        "theta": [],
        "q": [],
        "cc_frac": [],
        "gf": [],
        "le_veg": [],
    }

    print(f"Generating {NUM_TRAJS} trajectories...")
    
    for i in range(NUM_TRAJS):
        if i % 100 == 0:
            print(f"  Simulating trajectory {i+1}/{NUM_TRAJS}...")
            
        configs = get_perturbed_configs(seed=i) # deterministic seed for reproducibility per traj
        
        # rad with clouds
        rad_model = abcmodel.rad.StandardRadiationModel(
            **configs["rad_model_kwargs"],
        )
        rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

        # land surface
        land_model = abcmodel.land.AgsModel(
            **configs["ags_model_kwargs"],
        )
        land_state = land_model.init_state(
            **configs["ags_state_kwargs"],
        )

        # surface layer
        surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()
        surface_layer_state = surface_layer_model.init_state(
            **cm.obukhov_surface_layer.state_kwargs
        )

        # mixed layer
        mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
            **cm.bulk_mixed_layer.model_kwargs,
        )
        mixed_layer_state = mixed_layer_model.init_state(
            **configs["mixed_state_kwargs"],
        )

        # clouds
        cloud_model = abcmodel.atmos.clouds.CumulusModel()
        cloud_state = cloud_model.init_state()

        # define atmos model
        atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
            surface_layer=surface_layer_model,
            mixed_layer=mixed_layer_model,
            clouds=cloud_model,
        )
        atmos_state = atmos_model.init_state(
            surface=surface_layer_state,
            mixed=mixed_layer_state,
            clouds=cloud_state,
        )

        # define coupler and coupled state
        abcoupler = abcmodel.ABCoupler(
            rad=rad_model,
            land=land_model,
            atmos=atmos_model,
        )
        state = abcoupler.init_state(
            rad_state,
            land_state,
            atmos_state,
        )

        # run run run
        t_arr, trajectory = abcmodel.integrate(state, abcoupler, dt, runtime, tstart)
        
        if times is None:
            times = np.array(t_arr)

        # Collect data (convert to numpy)
        results["h_abl"].append(np.array(trajectory.atmos.mixed.h_abl))
        results["theta"].append(np.array(trajectory.atmos.mixed.theta))
        results["q"].append(np.array(trajectory.atmos.mixed.q))
        results["cc_frac"].append(np.array(trajectory.atmos.clouds.cc_frac))
        results["gf"].append(np.array(trajectory.land.gf))
        results["le_veg"].append(np.array(trajectory.land.le_veg))

    # Convert to arrays: shape (NUM_TRAJS, time)
    for key in results:
        results[key] = np.array(results[key])

    print("Computing statistics...")
    stats = {}
    for key in results:
        data = results[key]
        stats[key] = {
            "mean": np.mean(data, axis=0),
            "var": np.var(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
        }

    # Save to HDF5
    output_file = "data/generated_dataset.h5"
    print(f"Saving to {output_file}...")
    with h5py.File(output_file, "w") as f:
        f.create_dataset("time", data=times)
        
        # Save raw data
        grp_raw = f.create_group("raw")
        for key, val in results.items():
            grp_raw.create_dataset(key, data=val)
            
        # Save statistics
        grp_stats = f.create_group("statistics")
        for key, val in stats.items():
            subgrp = grp_stats.create_group(key)
            for stat_name, stat_val in val.items():
                subgrp.create_dataset(stat_name, data=stat_val)

    # Plot statistics
    print("Plotting results...")
    plt.figure(figsize=(15, 10))
    
    plot_vars = [
        ("h_abl", "h [m]", 231),
        ("q", "q [g kg-1]", 232), # Scale q by 1000 later if needed, but here raw check
        ("gf", "ground heat flux [W m-2]", 233),
        ("theta", "theta [K]", 234),
        ("cc_frac", "cloud fraction [-]", 235),
        ("le_veg", "latent heat flux vegetation [W m-2]", 236),
    ]

    for var_name, ylabel, plot_id in plot_vars:
        plt.subplot(plot_id)
        
        # Get stats
        mean = stats[var_name]["mean"]
        min_val = stats[var_name]["min"]
        max_val = stats[var_name]["max"]
        
        # Scaling for q
        scale = 1000.0 if var_name == "q" else 1.0
        
        plt.plot(times, mean * scale, 'b-', label='Mean')
        plt.fill_between(times, min_val * scale, max_val * scale, color='b', alpha=0.2, label='Range (Min-Max)')
        
        plt.xlabel("time [h]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(var_name)

    plt.tight_layout()
    plt.savefig("figs/generated_statistics.png") # Save instead of show for non-interactive
    print("Plot saved to figs/generated_statistics.png")
    # plt.show()


if __name__ == "__main__":
    main()
