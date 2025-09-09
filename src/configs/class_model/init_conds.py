from abcmodel.clouds import StandardCumulusInitConds
from abcmodel.land_surface import AquaCropInitConds, JarvisStewartInitConds
from abcmodel.mixed_layer import BulkMixedLayerInitConds
from abcmodel.radiation import StandardRadiationInitConds
from abcmodel.surface_layer import StandardSurfaceLayerInitConds

THETA = 288.0

radiation = StandardRadiationInitConds(
    net_rad=400,
)

surface_layer = StandardSurfaceLayerInitConds(
    ustar=0.3,
    z0m=0.02,
    z0h=0.002,
    theta=THETA,
)

clouds = StandardCumulusInitConds()

mixed_layer = BulkMixedLayerInitConds(
    abl_height=200.0,
    theta=THETA,
    dtheta=1.0,
    wtheta=0.1,
    q=0.008,
    dq=-0.001,
    wq=1e-4,
    co2=422.0,
    dCO2=-44.0,
    wCO2=0.0,
    u=6.0,
    du=4.0,
    v=-4.0,
    dv=4.0,
    dz_h=150.0,
)

jarvis_stewart = JarvisStewartInitConds(
    wg=0.21,
    w2=0.21,
    temp_soil=285.0,
    temp2=286.0,
    surf_temp=290.0,
    wl=0.0000,
)

aquacrop = AquaCropInitConds(
    wg=0.21,
    w2=0.21,
    temp_soil=285.0,
    temp2=286.0,
    surf_temp=290.0,
    wl=0.0000,
)
