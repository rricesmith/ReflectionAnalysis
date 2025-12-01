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
fig2, (ax_freq, ax_time) = plt.subplots(2, 1, figsize=(12, 10))

# Parameters
n_samples = 256
sampling_rate = 2.0 # GHz
dt = 1.0 / sampling_rate # ns
t = np.arange(n_samples) * dt

# Generate Noise
# User specified: random rayleigh noise, sigma of 1.0
noise_trace = np.random.rayleigh(scale=1.0, size=n_samples)

# FFT
fft_noise = np.fft.rfft(noise_trace)
freqs = np.fft.rfftfreq(n_samples, d=dt) # GHz

amplifiers = ['100', '200', '300']
styles = {'100': 'solid', '200': 'dotted', '300': 'dashed'}
colors = {'100': 'darkblue', '200': 'darkorange', '300': 'darkgreen'}

for amp in amplifiers:
    # Get amplifier response
    gain_func = ac.amplifier_response[amp]['gain']
    response = gain_func(freqs)
    
    # Convolve (multiply in freq domain)
    fft_convolved = fft_noise * response
    
    # Inverse FFT to get time trace
    trace_convolved = np.fft.irfft(fft_convolved, n=n_samples)
    
    # Renormalize FFT (magnitude)
    fft_mag = np.abs(fft_convolved)
    if np.max(fft_mag) != 0:
        fft_mag /= np.max(fft_mag)
        
    # Renormalize Time Trace
    if np.max(np.abs(trace_convolved)) != 0:
        trace_convolved /= np.max(np.abs(trace_convolved))
        
    # Plot
    ax_freq.plot(freqs, fft_mag, label=f'{amp}-series', linestyle=styles[amp], color=colors[amp], linewidth=2)
    ax_time.plot(t, trace_convolved, label=f'{amp}-series', linestyle=styles[amp], color=colors[amp], linewidth=2)

ax_freq.set_title('Normalized Frequency Spectrum of Noise')
ax_freq.set_xlabel('Frequency [GHz]')
ax_freq.set_ylabel('Normalized Amplitude')
ax_freq.legend()
ax_freq.grid(True)

ax_time.set_title('Normalized Time Trace of Noise')
ax_time.set_xlabel('Time [ns]')
ax_time.set_ylabel('Normalized Amplitude')
ax_time.legend()
ax_time.grid(True)

fig2.tight_layout()

# plt.show()
fig2.savefig('plots/noiseAnalysis.png')
