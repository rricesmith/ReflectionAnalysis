from NuRadioMC.SignalGen import askaryan
import NuRadioReco.detector.antennapattern
import NuRadioReco.detector.ARIANNA.analog_components as ac
from NuRadioReco.utilities import units, fft
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Askaryan Time Traces for two angles ---
n_samples = 512
dt = 0.5 * units.ns
sampling_rate = 1.0 / dt  # samples per second

# Shower parameters
energy = 1e17 * units.eV
shower_type = 'had'
n_index = 1.78
R = 10 * units.m
model = 'Alvarez2009'

theta_55 = 55 * units.deg
theta_57 = 57 * units.deg

# Get the electric field traces for both angles
trace_55 = askaryan.get_time_trace(energy, theta_55, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)
trace_57 = askaryan.get_time_trace(energy, theta_57, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)
times = np.arange(0, n_samples * dt, dt)

# --- 2. Calculate Antenna Response ---
provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
LPDA_antenna = provider.load_antenna_pattern("createLPDA_100MHz_InfFirn")

frequencies = np.fft.rfftfreq(n_samples, dt)

inc_zen = 0 * units.deg
inc_azi = 0 * units.deg
orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

VELs = LPDA_antenna.get_antenna_response_vectorized(frequencies, inc_zen, inc_azi,
                                            orientation_theta_phi[0], orientation_theta_phi[1],
                                            rotation_theta_phi[0], rotation_theta_phi[1])
vel_theta = VELs['theta']

# --- 3. E-field → Voltage via VEL (multiply in frequency domain) ---
trace_spec_55 = fft.time2freq(trace_55, sampling_rate)
trace_spec_57 = fft.time2freq(trace_57, sampling_rate)

voltage_spec_55 = trace_spec_55 * vel_theta
voltage_spec_57 = trace_spec_57 * vel_theta

voltage_trace_55 = fft.freq2time(voltage_spec_55, sampling_rate)
voltage_trace_57 = fft.freq2time(voltage_spec_57, sampling_rate)

# --- 4. Apply '200'-series amplifier response ---
gain_200 = ac.amplifier_response['200']['gain']
amp_spec_55 = voltage_spec_55 * gain_200(frequencies / units.GHz)
amp_spec_57 = voltage_spec_57 * gain_200(frequencies / units.GHz)

amp_trace_55 = fft.freq2time(amp_spec_55, sampling_rate)
amp_trace_57 = fft.freq2time(amp_spec_57, sampling_rate)

# --- 5. Windowing ---
# E-field: 50 ns window, starting 15 ns before the peak of the 55-deg e-field
n_efield_win = int(50 * units.ns / dt)    # 100 samples
n_efield_before = int(15 * units.ns / dt) # 30 samples
efield_max_idx = np.argmax(np.abs(trace_55))
efield_start = max(0, efield_max_idx - n_efield_before)
efield_end = efield_start + n_efield_win

# Voltage / post-amp: 128 ns window, starting 40 ns before the peak of the 55-deg voltage trace
n_volt_win = int(128 * units.ns / dt)   # 256 samples
n_volt_before = int(40 * units.ns / dt) # 80 samples
volt_max_idx = np.argmax(np.abs(voltage_trace_55))
volt_start = max(0, volt_max_idx - n_volt_before)
volt_end = volt_start + n_volt_win

times_efield = times[efield_start:efield_end]
times_volt = times[volt_start:volt_end]

# --- 6. Plotting (2x2) ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Top left: E-field in time
ax1.plot(times_efield / units.ns, trace_55[efield_start:efield_end], label=r'$\theta = 55°$')
ax1.plot(times_efield / units.ns, trace_57[efield_start:efield_end], label=r'$\theta = 57°$')
ax1.set_title("Askaryan E-field")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Electric Field (V/m)")
ax1.legend()
ax1.grid(True)

# Top right: Raw voltage in time
ax2.plot(times_volt / units.ns, voltage_trace_55[volt_start:volt_end], label=r'$\theta = 55°$')
ax2.plot(times_volt / units.ns, voltage_trace_57[volt_start:volt_end], label=r'$\theta = 57°$')
ax2.set_title("Raw Voltage (E-field * LPDA VEL_theta)")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Voltage (V)")
ax2.legend()
ax2.grid(True)

# Bottom left: Post-amp time trace
ax3.plot(times_volt / units.ns, amp_trace_55[volt_start:volt_end], label=r'$\theta = 55°$')
ax3.plot(times_volt / units.ns, amp_trace_57[volt_start:volt_end], label=r'$\theta = 57°$')
ax3.set_title("Post-Amp Voltage (200-series)")
ax3.set_xlabel("Time (ns)")
ax3.set_ylabel("Voltage (V)")
ax3.legend()
ax3.grid(True)

# Bottom right: Post-amp frequency spectrum (log y-scale, 0-500 MHz)
ax4.plot(frequencies / units.MHz, np.abs(amp_spec_55), label=r'$\theta = 55°$')
ax4.plot(frequencies / units.MHz, np.abs(amp_spec_57), label=r'$\theta = 57°$')
ax4.set_xlim(0, 500)
ax4.set_title("Post-Amp Frequency Spectrum (200-series)")
ax4.set_xlabel("Frequency (MHz)")
ax4.set_ylabel("|V(f)|")
ax4.legend()
ax4.grid(True)

outfile = 'plots/AskaryanPulseAndVoltage.png'
plt.tight_layout()
plt.savefig(outfile)
print(f"Plot saved to {outfile}")
