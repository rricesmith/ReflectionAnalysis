import numpy as np
import matplotlib.pyplot  as plt
import NuRadioReco.detector.ARIANNA.analog_components as ac
import os


fig, ax = plt.subplots(1,1,figsize=(12, 8),sharex='col',sharey='row')
ff = np.linspace(0.01,1.2,1000)

amp_response = '100'
ax.plot(ff,20*np.log10(np.abs(ac.amplifier_response[amp_response]['gain'](ff))),linewidth=3,linestyle='solid',label='100-series',color='darkblue')

amp_response = '200'
ax.plot(ff,20*np.log10(np.abs(ac.amplifier_response[amp_response]['gain'](ff))),linewidth=3,linestyle='dotted',label='200-series',color='darkorange')

amp_response = '300'
ax.plot(ff,20*np.log10(np.abs(ac.amplifier_response[amp_response]['gain'](ff))),linewidth=3,linestyle='dashed',label='300-series',color='darkgreen')

ax.set_ylim(0,75)
ax.set_xlim(0.0,1.2)
ax.legend()
ax.set_ylabel('gain [dB]')
ax.set_xlabel('frequency [GHz]')

# fig.tight_layout()
fig.savefig('plots/ampResponse.png')
#fig.savefig(PathToARIANNAanalysis + '/plots/ampResponse.pdf')

# --- Noise Analysis ---

# Parameters
n_samples = 256
sampling_rate = 2.0 # GHz
dt = 1.0 / sampling_rate # ns
t = np.arange(n_samples) * dt

# Generate Noise in Frequency Domain
# Random complex noise (White noise in time, flat-ish in freq)
np.random.seed(12345)
freqs = np.fft.rfftfreq(n_samples, d=dt) # GHz
fft_noise = np.random.normal(0, 1, len(freqs)) + 1j * np.random.normal(0, 1, len(freqs))

# Flat Value of 5 FFT (Comparison Signal)
fft_flat = np.full_like(freqs, 5.0, dtype=complex)

amplifiers = ['100', '200', '300']
colors = {'100': 'purple', '200': 'darkorange', '300': 'darkgreen'}

for amp in amplifiers:
    fig_amp, (ax_freq, ax_time) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get amplifier response
    gain_func = ac.amplifier_response[amp]['gain']
    response = gain_func(freqs)
    
    # Convolve Noise (multiply in freq domain)
    fft_noise_convolved = fft_noise * response
    trace_noise_convolved = np.fft.irfft(fft_noise_convolved, n=n_samples)
    
    # Convolve Flat-5 Signal
    fft_flat_convolved = fft_flat * response
    trace_flat_convolved = np.fft.irfft(fft_flat_convolved, n=n_samples)
    
    # Determine limits from Flat-5 trace
    # Use the maximum absolute value of the Flat-5 trace to set limits
    max_limit = np.max(np.abs(trace_flat_convolved))
    
    # Plot Frequency Spectrum (Magnitude) of Noise
    fft_mag = np.abs(fft_noise_convolved)
    
    ax_freq.plot(freqs, fft_mag, label=f'Noise', linestyle='dashed', color=colors[amp], linewidth=2)
    ax_freq.set_title(f'Frequency Spectrum - {amp}-series')
    ax_freq.set_xlabel('Frequency [GHz]')
    ax_freq.set_ylabel('Amplitude')
    ax_freq.legend()
    ax_freq.grid(True)

    # Plot Time Trace of Noise
    ax_time.plot(t, trace_noise_convolved, label=f'Noise', linestyle='dashed', color=colors[amp], linewidth=2)
    ax_time.set_title(f'Time Trace - {amp}-series')
    ax_time.set_xlabel('Time [ns]')
    ax_time.set_ylabel('Amplitude')
    
    # Set limits based on Flat-5 signal
    ax_time.set_ylim(-max_limit, max_limit)
    
    ax_time.legend()
    ax_time.grid(True)

    fig_amp.tight_layout()
    fig_amp.savefig(f'plots/noiseAnalysis_{amp}.png')
    plt.close(fig_amp)
