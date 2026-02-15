from NuRadioMC.SignalGen import askaryan
import NuRadioReco.detector.antennapattern
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Askaryan Time Traces for two angles ---
n_samples = 512 # define number of samples in the trace
dt = 0.5 * units.ns # define time resolution (bin width)

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

# Calculate frequencies corresponding to the time trace
frequencies = np.fft.rfftfreq(n_samples, dt)

# Define geometry for antenna response (Using Frontlobe configuration)
inc_zen = 0 * units.deg
inc_azi = 0 * units.deg
orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

# Get Vector Effective Length (VEL)
VELs = LPDA_antenna.get_antenna_response_vectorized(frequencies, inc_zen, inc_azi,
                                            orientation_theta_phi[0], orientation_theta_phi[1],
                                            rotation_theta_phi[0], rotation_theta_phi[1])
vel_theta = VELs['theta']

# --- 3. Convolve / Multiply in Frequency Domain for both angles ---
trace_spec_55 = askaryan.get_frequency_spectrum(energy, theta_55, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)
trace_spec_57 = askaryan.get_frequency_spectrum(energy, theta_57, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)

voltage_spec_55 = trace_spec_55 * vel_theta
voltage_spec_57 = trace_spec_57 * vel_theta

voltage_trace_55 = np.fft.irfft(voltage_spec_55, n=n_samples)
voltage_trace_57 = np.fft.irfft(voltage_spec_57, n=n_samples)

# --- 4. Window: 128 ns starting 40 ns before the peak of the 55-deg voltage trace ---
n_window = int(128 * units.ns / dt)  # 256 samples
n_before = int(40 * units.ns / dt)   # 80 samples

max_idx = np.argmax(np.abs(voltage_trace_55))
start_idx = max(0, max_idx - n_before)
end_idx = start_idx + n_window

# Same window applied to both e-field and voltage plots
times_win = times[start_idx:end_idx]
trace_55_win = trace_55[start_idx:end_idx]
trace_57_win = trace_57[start_idx:end_idx]
voltage_55_win = voltage_trace_55[start_idx:end_idx]
voltage_57_win = voltage_trace_57[start_idx:end_idx]

# --- 5. Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Original Askaryan Electric Field
ax1.plot(times_win / units.ns, trace_55_win, label=r'$\theta = 55°$')
ax1.plot(times_win / units.ns, trace_57_win, label=r'$\theta = 57°$')
ax1.set_title("Original Askaryan Pulse (E-field)")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Electric Field (V/m)")
ax1.legend()
ax1.grid(True)

# Plot 2: Resulting Voltage Trace
ax2.plot(times_win / units.ns, voltage_55_win, label=r'$\theta = 55°$')
ax2.plot(times_win / units.ns, voltage_57_win, label=r'$\theta = 57°$')
ax2.set_title("Voltage Trace (E-field * LPDA VEL_theta)")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Voltage (V)")
ax2.legend()
ax2.grid(True)

# Plot 3: Frequency Spectrum of Voltage Trace (log y-scale, 0-500 MHz)
ax3.plot(frequencies / units.MHz, np.abs(voltage_spec_55), label=r'$\theta = 55°$')
ax3.plot(frequencies / units.MHz, np.abs(voltage_spec_57), label=r'$\theta = 57°$')
ax3.set_xlim(0, 500)
ax3.set_title("Voltage Frequency Spectrum")
ax3.set_xlabel("Frequency (MHz)")
ax3.set_ylabel("|V(f)|")
ax3.legend()
ax3.grid(True)

outfile = 'plots/AskaryanPulseAndVoltage.png'
plt.tight_layout()
plt.savefig(outfile)
print(f"Plot saved to {outfile}")
