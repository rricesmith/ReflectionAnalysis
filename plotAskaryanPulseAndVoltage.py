from NuRadioMC.SignalGen import askaryan
import NuRadioReco.detector.antennapattern
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Askaryan Time Trace (from surface_shower_propagation.py) ---
n_samples = 512 # define number of samples in the trace
dt = 0.5 * units.ns # define time resolution (bin width)

# Shower parameters
energy = 1e17 * units.eV
theta = 55 * units.deg
shower_type = 'had'
n_index = 1.78
R = 10 * units.m
model = 'Alvarez2009'

# Get the electric field trace
trace = askaryan.get_time_trace(energy, theta, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)
times = np.arange(0, n_samples * dt, dt)

# --- 2. Calculate Antenna Response (from plotAntennaResponse.py) ---
provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
LPDA_antenna = provider.load_antenna_pattern("createLPDA_100MHz_InfFirn")

# Calculate frequencies corresponding to the time trace
frequencies = np.fft.rfftfreq(n_samples, dt)

# Define geometry for antenna response (Using Frontlobe configuration)
# Using Zenith 0 deg, Azimuth 0 deg as representative of the front lobe boresight
inc_zen = 0 * units.deg
inc_azi = 0 * units.deg
orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

# Get Vector Effective Length (VEL)
VELs = LPDA_antenna.get_antenna_response_vectorized(frequencies, inc_zen, inc_azi,
                                            orientation_theta_phi[0], orientation_theta_phi[1],
                                            rotation_theta_phi[0], rotation_theta_phi[1])

# Extract the Theta component of the VEL
vel_theta = VELs['theta']

# --- 3. Convolve / Multiply in Frequency Domain ---
# Get frequency spectrum directly
trace_spec = askaryan.get_frequency_spectrum(energy, theta, n_samples, dt, shower_type=shower_type, n_index=n_index, R=R, model=model)

# Apply Antenna Response (E-field * VEL = Voltage)
# Note: This is an element-wise multiplication in the frequency domain
voltage_spec = trace_spec * vel_theta

# Convert back to time domain to get the voltage trace
voltage_trace = np.fft.irfft(voltage_spec, n=n_samples)


# --- 4. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Original Askaryan Electric Field
ax1.plot(times / units.ns, trace)
ax1.set_title("Original Askaryan Pulse (E-field)")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Electric Field (V/m)")
ax1.grid(True)

# Plot 2: Resulting Voltage Trace
ax2.plot(times / units.ns, voltage_trace)
ax2.set_title("Voltage Trace (E-field * LPDA VEL_theta)")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Voltage (V)")
ax2.grid(True)

outfile = 'plots/AskaryanPulseAndVoltage.png'
plt.tight_layout()
plt.savefig(outfile)
print(f"Plot saved to {outfile}")
