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

    min_freq = 30*units.MHz
    max_freq = 2000*units.MHz

    noise_type = 'rayleigh'
    amplitude = 20*units.mV
    bandwidth = None

    window = 80 # 40 ns
    thresh = 3.5 * amplitude

    n_noise = 5000
    noise_array = np.zeros((n_noise, 4, n_samples))

    for i in range(n_noise):
        
        pass_trigger = np.array([False, False, False, False])
        while np.sum(pass_trigger) < 2: 
            event = np.zeros((4, n_samples))
            pass_trigger = np.array([False, False, False, False])


            for j in range(4):
                noise = channelGenericNoiseAdder.bandlimited_noise(min_freq=min_freq, max_freq=max_freq, n_samples=n_samples, sampling_rate=sampling_rate, amplitude=amplitude, type=noise_type, bandwidth=bandwidth)
                event[i] = noise

                # Slide the window across the row
                # The loop stops when the window would go past the end of the array
                for j in range(256 - window + 1):
                    current_window = noise[j : j + window]

                    # Check if any value in the window exceeds the positive threshold
                    exceeds_positive = np.any(current_window > thresh)

                    # Check if any value in the window is below the negative threshold
                    exceeds_negative = np.any(current_window < -thresh)

                    # If both conditions are true for this window, mark the row as True
                    # and break to the next row, as the condition is met.
                    if exceeds_positive and exceeds_negative:
                        pass_trigger[i] = True



    # Save the noise array
    np.save(f'SimpleFootprintSimulation/output/noise_array_{n_noise}.npy', noise_array)