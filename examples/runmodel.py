import matplotlib.pyplot as plt

from abcmodel.abcmodel import LandSurfaceInput, Model


def main():
    # create empty model_input and set up case
    run1input = LandSurfaceInput()

    # 0. running configurations:
    dt = 60.0  # time step [s]
    runtime = 12 * 3600.0  # total run time [s]

    # 1. mixed-layer input:
    # 1.1. mixed layer switches
    # mixed-layer model switch
    sw_ml = True
    # shear growth mixed-layer switch
    sw_shearwe = False
    # fix the free-troposphere switch
    sw_fixft = False
    # 1.2. large scale parameters
    # initial ABL height [m]
    abl_height = 200.0
    # surface pressure [Pa]
    surf_pressure = 101300.0
    # horizontal large-scale divergence of wind [s-1]
    divU = 0.0
    # Coriolis parameter [m s-1]
    coriolis_param = 1.0e-4
    # 1.3 temperature parameters
    # initial mixed-layer potential temperature [K]
    theta = 288.0
    # initial temperature jump at h [K]
    dtheta = 1.0
    # free atmosphere potential temperature lapse rate [K m-1]
    gammatheta = 0.006
    # advection of heat [K s-1]
    advtheta = 0.0
    # entrainment ratio for virtual heat [-]
    beta = 0.2
    # surface kinematic heat flux [K m s-1]
    wtheta = 0.1
    # 1.4 moisture parameters
    # initial mixed-layer specific humidity [kg kg-1]
    q = 0.008
    # initial specific humidity jump at h [kg kg-1]
    dq = -0.001
    # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
    gammaq = 0.0
    # advection of moisture [kg kg-1 s-1]
    advq = 0.0
    # surface kinematic moisture flux [kg kg-1 m s-1]
    wq = 1e-4
    # 1.5. CO2 parameters
    # initial mixed-layer CO2 [ppm]
    co2 = 422.0
    # initial CO2 jump at h [ppm]
    dCO2 = -44.0
    # free atmosphere CO2 lapse rate [ppm m-1]
    gammaCO2 = 0.0
    # advection of CO2 [ppm s-1]
    advCO2 = 0.0
    # surface kinematic CO2 flux [ppm m s-1]
    wCO2 = 0.0
    # 1.6. wind parameters
    # prognostic wind switch
    sw_wind = True
    # initial mixed-layer u-wind speed [m s-1]
    u = 6.0
    # initial u-wind jump at h [m s-1]
    du = 4.0
    # free atmosphere u-wind speed lapse rate [s-1]
    gammau = 0.0
    # advection of u-wind [m s-2]
    advu = 0.0
    # initial mixed-layer v-wind speed [m s-1]
    v = -4.0
    # initial v-wind jump at h [m s-1]
    dv = 4.0
    # free atmosphere v-wind speed lapse rate [s-1]
    gammav = 0.0
    # advection of v-wind [m s-2]
    advv = 0.0

    # 2. surface layer input:
    # surface layer switch
    sw_sl = False
    # surface friction velocity [m s-1]
    ustar = 0.3
    # roughness length for momentum [m]
    z0m = 0.02
    # roughness length for scalars [m]
    z0h = 0.002

    # 3. surface radiation input:
    # radiation switch
    sw_rad = True
    # latitude [deg]
    lat = 51.97
    # longitude [deg]
    lon = -4.93
    # day of the year [-]
    doy = 268.0
    # time of the day [h UTC]
    tstart = 6.8
    # cloud cover fraction [-]
    cc = 0.0
    # net radiation [W m-2]
    net_rad = 400.0
    # cloud top radiative divergence [W m-2]
    dFz = 0.0

    # 4. land surface switch
    # land surface switch
    run1input.sw_ls = False
    # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
    run1input.ls_type = "js"
    # volumetric water content top soil layer [m3 m-3]
    run1input.wg = 0.21
    # volumetric water content deeper soil layer [m3 m-3]
    run1input.w2 = 0.21
    # vegetation fraction [-]
    run1input.cveg = 0.85
    # temperature top soil layer [K]
    run1input.Tsoil = 285.0
    # temperature deeper soil layer [K]
    run1input.T2 = 286.0
    # Clapp and Hornberger retention curve parameter a
    run1input.a = 0.219
    # Clapp and Hornberger retention curve parameter b
    run1input.b = 4.90
    # Clapp and Hornberger retention curve parameter c
    run1input.p = 4.0
    run1input.CGsat = 3.56e-6  # saturated soil conductivity for heat
    # saturated volumetric water content ECMWF config [-]
    run1input.wsat = 0.472
    # volumetric water content field capacity [-]
    run1input.wfc = 0.323
    # volumetric water content wilting point [-]
    run1input.wwilt = 0.171
    # C1 sat?
    run1input.C1sat = 0.132
    # C2 sat?
    run1input.C2ref = 1.8
    # leaf area index [-]
    run1input.LAI = 2.0
    # correction factor transpiration for VPD [-]
    run1input.gD = 0.0
    # minimum resistance transpiration [s m-1]
    run1input.rsmin = 110.0
    # minimun resistance soil evaporation [s m-1]
    run1input.rssoilmin = 50.0
    # surface albedo [-]
    run1input.alpha = 0.25
    # initial surface temperature [K]
    run1input.Ts = 290.0
    # thickness of water layer on wet vegetation [m]
    run1input.Wmax = 0.0002
    # equivalent water layer depth for wet vegetation [m]
    run1input.Wl = 0.0000
    # thermal diffusivity skin layer [-]
    run1input.Lambda = 5.9
    # Plant type ('c3' or 'c4')
    run1input.c3c4 = "c3"

    # 5. cumulus parameterization
    # cumulus parameterization switch
    sw_cu = True
    # Transition layer thickness [m]
    dz_h = 150.0

    # init and run the model
    r1 = Model(
        # 0. running configuration
        dt=dt,
        runtime=runtime,
        # 1. mixed layer
        sw_ml=sw_ml,
        sw_shearwe=sw_shearwe,
        sw_fixft=sw_fixft,
        abl_height=abl_height,
        surf_pressure=surf_pressure,
        divU=divU,
        coriolis_param=coriolis_param,
        theta=theta,
        dtheta=dtheta,
        gammatheta=gammatheta,
        advtheta=advtheta,
        beta=beta,
        wtheta=wtheta,
        q=q,
        dq=dq,
        gammaq=gammaq,
        advq=advq,
        wq=wq,
        co2=co2,
        dCO2=dCO2,
        gammaCO2=gammaCO2,
        advCO2=advCO2,
        wCO2=wCO2,
        sw_wind=sw_wind,
        u=u,
        du=du,
        gammau=gammau,
        advu=advu,
        v=v,
        dv=dv,
        gammav=gammav,
        advv=advv,
        sw_sl=sw_sl,
        ustar=ustar,
        z0m=z0m,
        z0h=z0h,
        sw_rad=sw_rad,
        lat=lat,
        lon=lon,
        doy=doy,
        tstart=tstart,
        cc=cc,
        net_rad=net_rad,
        dFz=dFz,
        sw_cu=sw_cu,
        dz_h=dz_h,
        model_input=run1input,
    )
    r1.run()

    # plot output
    plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.plot(r1.out.t, r1.out.h)
    plt.xlabel("time [h]")
    plt.ylabel("h [m]")

    plt.subplot(132)
    plt.plot(r1.out.t, r1.out.theta)
    plt.xlabel("time [h]")
    plt.ylabel("theta [K]")

    plt.subplot(133)
    plt.plot(r1.out.t, r1.out.q * 1000.0)
    plt.xlabel("time [h]")
    plt.ylabel("q [g kg-1]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
