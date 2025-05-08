import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import numpy as np
from icecream import ic

if __name__ == '__main__':

    # Using modules from channelGenericNoiseAdder, will generate pure noise traces
    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()

    sampling_rate = 2*units.GHz
    n_samples = 256

    min_freq = 50*units.MHz
    max_freq = 2000*units.MHz

    noise_type = 'rayleigh'
    amplitude = 10*units.mV
    bandwidth = None


    n_noise = 10
    noise_array = np.zeros((n_noise, n_samples))

    for i in range(n_noise):
        noise = channelGenericNoiseAdder.bandlimited_noise(min_freq=min_freq, max_freq=max_freq, n_samples=n_samples, sampling_rate=sampling_rate, amplitude=amplitude, type=noise_type, bandwidth=bandwidth)
        ic(noise.shape, noise.dtype, noise[0].shape, noise[0].dtype)
        ic(noise_array.shape, noise_array.dtype, noise_array[0].shape, noise_array[0].dtype)
        noise_array[i] = noise

        if True:
            # Plot the noise
            import matplotlib.pyplot as plt
            plt.plot(noise_array[i])
            plt.title('Noise Trace')
            plt.xlabel('Sample Number')
            plt.ylabel('Amplitude (V)')
            plt.grid()
            plt.savefig(f'SimpleFootprintSimulation/plots/noise_trace_{i}.png')
            plt.close()
        

    # Save the noise array
    np.save('SimpleFootprintSimulation/output/noise_array.npy', noise_array)