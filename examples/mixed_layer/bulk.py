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
        cm.params.mixed_layer,
        cm.init_conds.mixed_layer,
    )

    # 2. define surface layer model
    surface_layer_model = StandardSurfaceLayerModel(
        cm.params.surface_layer,
        cm.init_conds.surface_layer,
    )

    # 3. define radiation model
    radiation_model = StandardRadiationModel(
        cm.params.radiation,
        cm.init_conds.radiation,
    )

    # 4. define land surface model
    land_surface_model = JarvisStewartModel(
        cm.params.jarvis_stewart,
        cm.init_conds.jarvis_stewart,
    )

    # 5. clouds
    cloud_model = StandardCumulusModel(
        cm.params.clouds,
        cm.init_conds.clouds,
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
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.plot(abc.out.t, abc.mixed_layer.diagnostics.get("abl_height"))
    plt.xlabel("time [h]")
    plt.ylabel("ABL height [m]")

    plt.subplot(234)
    plt.plot(abc.out.t, abc.mixed_layer.diagnostics.get("theta"))
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(232)
    plt.plot(abc.out.t, abc.mixed_layer.diagnostics.get("q") * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.subplot(235)
    plt.plot(abc.out.t, abc.clouds.diagnostics.get("cc_frac"))
    plt.xlabel("time [h]")
    plt.ylabel("cloud fraction [-]")

    plt.subplot(233)
    plt.plot(abc.out.t, abc.mixed_layer.diagnostics.get("co2"))
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer CO2 [ppm]")

    plt.subplot(236)
    plt.plot(abc.out.t, abc.mixed_layer.diagnostics.get("u"))
    plt.xlabel("time [h]")
    plt.ylabel("mixed-layer u-wind speed [m s-1]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
