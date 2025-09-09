import matplotlib.pyplot as plt

import configs.class_model as cm
from abcmodel import ABCModel
from abcmodel.clouds import StandardCumulusModel
from abcmodel.land_surface import JarvisStewartModel
from abcmodel.mixed_layer import BulkMixedLayerModel
from abcmodel.radiation import StandardRadiationModel
from abcmodel.surface_layer import StandardSurfaceLayerModel


def main():
    # 0. running configurations:
    dt = 60.0  # time step [s]
    runtime = 96 * 3600.0  # total run time [s]

    # define mixed layer model
    mixed_layer_model = BulkMixedLayerModel(
        cm.mixed_layer.params,
        cm.mixed_layer.init_conds,
    )

    # 2. define surface layer model
    surface_layer_model = StandardSurfaceLayerModel(
        cm.surface_layer.params,
        cm.surface_layer.init_conds,
    )

    # 3. define radiation model
    radiation_model = StandardRadiationModel(
        cm.radiation.params,
        cm.radiation.init_conds,
    )

    # 4. define land surface model
    land_surface_model = JarvisStewartModel(
        cm.land_surface.jarvis_stewart_params,
        cm.land_surface.jarvis_stewart_init_conds,
    )

    # 5. clouds
    cloud_model = StandardCumulusModel(
        cm.clouds.params,
        cm.clouds.init_conds,
    )

    # init and run the model
    abc = ABCModel(
        dt=dt,
        runtime=runtime,
        mixed_layer=mixed_layer_model,
        surface_layer=surface_layer_model,
        radiation=radiation_model,
        land_surface=land_surface_model,
        clouds=cloud_model,
    )
    abc.run()

    # plot output
    time = abc.get_t()
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(time, abc.mixed_layer.diagnostics.get("abl_height"))
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(234)
    plt.plot(time, abc.mixed_layer.diagnostics.get("theta"))
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(time, abc.mixed_layer.diagnostics.get("q") * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(time, abc.clouds.diagnostics.get("cc_frac"))
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(time, abc.mixed_layer.diagnostics.get("wCO2"))
    plt.xlabel("time [h]")
    plt.ylabel("surface kinematic CO2 flux [ppm m s-1]")

    plt.subplot(236)
    plt.plot(time, abc.land_surface.diagnostics.get("le_veg"))
    plt.xlabel("time [h]")
    plt.ylabel("transpiration [W m-2]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
