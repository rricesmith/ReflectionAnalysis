from NuRadioMC.SignalGen import askaryan
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt




rad_data = [0, 300, 400, 500, 600, 936.3636364]
#amp_data = [1, 0.02366431913, 0.007071067812, 0.00632455532, 0.004264014327, 0.004242640687, 0.0005]
amp_data = [1, 0.02366431913, 0.007071067812, 0.00632455532, 0.004264014327, 0.004242640687]
#amp_data = np.sqrt(amp_data)

#fit4 = np.poly1d(np.polyfit(rad_data, amp_data, 4))
#fit5 = np.poly1d(np.polyfit(rad_data, amp_data, 5))
#fit6 = np.poly1d(np.polyfit(rad_data, amp_data, 6))
#fit7 = np.poly1d(np.polyfit(rad_data, amp_data, 7))
xfit = np.linspace(10, 2000, 2000)
power_fit = 44*xfit**-1.4
power_fit_2k = 424*xfit**-1.77
r_data = 1/xfit

#plt.plot(rad_data, amp_data, '.', xfit, fit5(xfit), '-', xfit, fit7(xfit), '--', xfit, fit4(xfit), 'x', xfit, power_fit, 'o')
data = plt.scatter(rad_data, amp_data)
lower_fit, = plt.plot(xfit, power_fit, '-')
higher_fit, = plt.plot(xfit, power_fit_2k, '--')
r, = plt.plot(xfit, r_data)
extra_data = plt.scatter(2000, 0.0005)
plt.legend((data, extra_data, lower_fit, higher_fit, r), ('Original Data', 'Extrapolated data at 2k','Fit to Data, y=44x^-1.4', 'Fit with extra data, y=424x^-1.77', '1/r scaling'))
plt.yscale('log')
plt.title('Power Law Fit to Data')
plt.ylabel('Sqrt(Surface Air Pulse/d^-2 extrap. from 100m)')
plt.xlabel('Distance (m)')
# plt.show()
plt.savefig('plots/SurfacePulseDistanceFit.png')
plt.close()

data = plt.scatter(rad_data, amp_data)
lower_fit, = plt.plot(xfit, power_fit, '-')
r, = plt.plot(xfit, r_data)
plt.legend((data, lower_fit, r), ('Original Data', 'Fit to Data, y=44x^-1.4', '1/r scaling'))
plt.yscale('log')
plt.title('Power Law Fit to Data')
plt.ylabel('Sqrt(Surface Air Pulse / d^2)')
plt.xlabel('Distance (m)')
# plt.show()
plt.savefig('plots/SurfacePulseDistanceFit_NoExtra.png')
plt.close()



# quit()

n_samples = 512 #define number of sample in the trace
dt = 0.5 * units.ns #definte time resolution (bin width) 0.5ns

trace = askaryan.get_time_trace(1e17 * units.eV, 55 * units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=10*units.m, model='Alvarez2009')

#print(trace)

max_trace = max(trace)

max_trace = max_trace  * units.V * units.m
print(max_trace)

tt = np.arange(0, dt * n_samples, dt)

#plt.scatter(tt, trace)
#plt.show()

print(len(trace))
print(len(tt))

max_amp = max(trace)
print(max_amp)

rad = np.arange(10, 1000, 10)
amps = np.zeros(len(rad))
amps_55_8 = np.zeros(len(rad))
scaling = np.zeros(len(rad))

#amps = max_amp * 0.01 * 1/rad * np.exp(-rad/400)

for a in range(len(rad)):
    scaling_exp = np.exp(-rad[a]/400)
    scaling[a] = scaling_exp
    amps[a] = 0.01 * max(askaryan.get_time_trace(2e18*units.eV, 55*units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=rad[a]*units.m, model='Alvarez2009')) * np.exp(-rad[a]/400)
    amps_55_8[a] =  0.01 * max(askaryan.get_time_trace(2e18*units.eV, 55.8*units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=rad[a]*units.m, model='Alvarez2009')) * np.exp(-rad[a]/400)

lpda_threshold = 1.3662817627550142e-05  #LPDA threshold at sigma of 3.9498194908011524
#dipole_threshold = 1.689172534356015e-05

plt_55 = plt.scatter(rad, amps)
plt_558 = plt.scatter(rad, amps_55_8)
plt.legend((plt_55, plt_558), ('55 deg viewangle', '55.8 deg viewangle'))
plt.title('Max Amplitude Askaryan E-field for 2e18eV shower *0.01 coupling with 400m attenuation')
plt.tick_params(which='both', top=True, right=True)
plt.ylabel('V/m')
plt.xlabel('radius (m)')
plt.yscale('log')
# plt.show()
plt.savefig('plots/SurfaceAskaryanAmplitudeVsRadius.png')
plt.close()


energies = np.arange(1, 100, 1)
max_amps = np.zeros(len(energies))
max_rad = np.zeros(len(energies))
rad = np.arange(10, 1000, 10)

for i_E in range(len(energies)):
    trace = askaryan.get_time_trace(energies[i_E] * 1e17 * units.eV, 55 * units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=1*units.km, model='Alvarez2009')

    max_trace = max(trace)

    for r in rad:
        trace = 0.01 * max(askaryan.get_time_trace(energies[i_E] * 1e17 * units.eV, 55 * units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=r*units.m, model='Alvarez2009'))
#        V_r = max_trace * 0.01 * 1/r * np.exp(-r/400)
        if trace > lpda_threshold:
            max_rad[i_E] = r

plt.scatter(energies, max_rad)
plt.title('Max Radius of Voltage past LPDA Threshold for Surface Propogated Signals')
plt.ylabel('Radius (m)')
plt.xlabel('Energy (10^17 eV)')
# plt.show()
plt.savefig('plots/SurfaceAskaryanMaxRadiusVsEnergy.png')
plt.close()
