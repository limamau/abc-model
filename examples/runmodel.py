#
# Example of how to run the Python code, and access the output
# This case is identical to the default setup of CLASS (the version with interface)
#


import matplotlib.pyplot as plt

from abcmodel.abcmodel import *

# from pylab import *

"""
Create empty model_input and set up case
"""
run1input = model_input()

run1input.dt = 60.0  # time step [s]
run1input.runtime = 12 * 3600  # total run time [s]

# mixed-layer input
run1input.sw_ml = True  # mixed-layer model switch
run1input.sw_shearwe = False  # shear growth mixed-layer switch
run1input.sw_fixft = False  # Fix the free-troposphere switch
run1input.h = 200.0  # initial ABL height [m]
run1input.Ps = 101300.0  # surface pressure [Pa]
run1input.divU = 0.0  # horizontal large-scale divergence of wind [s-1]
run1input.fc = 1.0e-4  # Coriolis parameter [m s-1]

run1input.theta = 288.0  # initial mixed-layer potential temperature [K]
run1input.dtheta = 1.0  # initial temperature jump at h [K]
run1input.gammatheta = 0.006  # free atmosphere potential temperature lapse rate [K m-1]
run1input.advtheta = 0.0  # advection of heat [K s-1]
run1input.beta = 0.2  # entrainment ratio for virtual heat [-]
run1input.wtheta = 0.1  # surface kinematic heat flux [K m s-1]

run1input.q = 0.008  # initial mixed-layer specific humidity [kg kg-1]
run1input.dq = -0.001  # initial specific humidity jump at h [kg kg-1]
run1input.gammaq = 0.0  # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
run1input.advq = 0.0  # advection of moisture [kg kg-1 s-1]
run1input.wq = 0.1e-3  # surface kinematic moisture flux [kg kg-1 m s-1]

run1input.CO2 = 422.0  # initial mixed-layer CO2 [ppm]
run1input.dCO2 = -44.0  # initial CO2 jump at h [ppm]
run1input.gammaCO2 = 0.0  # free atmosphere CO2 lapse rate [ppm m-1]
run1input.advCO2 = 0.0  # advection of CO2 [ppm s-1]
run1input.wCO2 = 0.0  # surface kinematic CO2 flux [ppm m s-1]

run1input.sw_wind = False  # prognostic wind switch
run1input.u = 6.0  # initial mixed-layer u-wind speed [m s-1]
run1input.du = 4.0  # initial u-wind jump at h [m s-1]
run1input.gammau = 0.0  # free atmosphere u-wind speed lapse rate [s-1]
run1input.advu = 0.0  # advection of u-wind [m s-2]

run1input.v = -4.0  # initial mixed-layer u-wind speed [m s-1]
run1input.dv = 4.0  # initial u-wind jump at h [m s-1]
run1input.gammav = 0.0  # free atmosphere v-wind speed lapse rate [s-1]
run1input.advv = 0.0  # advection of v-wind [m s-2]

run1input.sw_sl = False  # surface layer switch
run1input.ustar = 0.3  # surface friction velocity [m s-1]
run1input.z0m = 0.02  # roughness length for momentum [m]
run1input.z0h = 0.002  # roughness length for scalars [m]

run1input.sw_rad = False  # radiation switch
run1input.lat = 51.97  # latitude [deg]
run1input.lon = -4.93  # longitude [deg]
run1input.doy = 268.0  # day of the year [-]
run1input.tstart = 6.8  # time of the day [h UTC]
run1input.cc = 0.0  # cloud cover fraction [-]
run1input.Q = 400.0  # net radiation [W m-2]
run1input.dFz = 0.0  # cloud top radiative divergence [W m-2]

run1input.sw_ls = False  # land surface switch
run1input.ls_type = (
    "js"  # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
)
run1input.wg = 0.21  # volumetric water content top soil layer [m3 m-3]
run1input.w2 = 0.21  # volumetric water content deeper soil layer [m3 m-3]
run1input.cveg = 0.85  # vegetation fraction [-]
run1input.Tsoil = 285.0  # temperature top soil layer [K]
run1input.T2 = 286.0  # temperature deeper soil layer [K]
run1input.a = 0.219  # Clapp and Hornberger retention curve parameter a
run1input.b = 4.90  # Clapp and Hornberger retention curve parameter b
run1input.p = 4.0  # Clapp and Hornberger retention curve parameter c
run1input.CGsat = 3.56e-6  # saturated soil conductivity for heat

run1input.wsat = 0.472  # saturated volumetric water content ECMWF config [-]
run1input.wfc = 0.323  # volumetric water content field capacity [-]
run1input.wwilt = 0.171  # volumetric water content wilting point [-]

run1input.C1sat = 0.132
run1input.C2ref = 1.8

run1input.LAI = 2.0  # leaf area index [-]
run1input.gD = 0.0  # correction factor transpiration for VPD [-]
run1input.rsmin = 110.0  # minimum resistance transpiration [s m-1]
run1input.rssoilmin = 50.0  # minimun resistance soil evaporation [s m-1]
run1input.alpha = 0.25  # surface albedo [-]

run1input.Ts = 290.0  # initial surface temperature [K]

run1input.Wmax = 0.0002  # thickness of water layer on wet vegetation [m]
run1input.Wl = 0.0000  # equivalent water layer depth for wet vegetation [m]

run1input.Lambda = 5.9  # thermal diffusivity skin layer [-]

run1input.c3c4 = "c3"  # Plant type ('c3' or 'c4')

run1input.sw_cu = False  # Cumulus parameterization switch
run1input.dz_h = 150.0  # Transition layer thickness [m]

"""
Init and run the model
"""
r1 = model(run1input)
r1.run()

"""
Plot output
"""
plt.figure()
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
plt.show()
