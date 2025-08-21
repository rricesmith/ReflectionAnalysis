import numpy as np
import functools
import logging
from scipy import constants as scipy_constants

from NuRadioReco.utilities import units
from NuRadioReco.utilities.geometryUtilities import get_fresnel_r_p, get_fresnel_r_s
from NuRadioReco.detector import antennapattern

# Set up a logger for warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EfieldProcessor')

c = scipy_constants.c * units.m / units.s

class EfieldProcessor:
    """
    A class to process electric fields, including surface reflection
    and conversion to voltage, with optional caching for antenna response.
    """
    def __init__(self, caching=True):
        self.__antenna_provider = antennapattern.AntennaPatternProvider()
        self.__caching = caching
        self.__freqs = None

    @functools.lru_cache(maxsize=1024)
    def _get_cached_antenna_response(self, ant_pattern, zen, azi, *ant_orient):
        return ant_pattern.get_antenna_response_vectorized(self.__freqs, zen, azi, *ant_orient)

    def modifyEfieldForSurfaceReflection(self, Efield, incoming_zenith, antenna_height=1*units.m, n_index=1.35):
        """
        Modifies an Efield object to account for surface reflection.

        This version explicitly handles trace boundaries and warns if the reflected
        signal is truncated.

        It returns only the reflected Efield signal, so the user has to manually add it to the original Efield if they wish
        """
        # Get original trace data
        original_traces = Efield.get_trace()
        num_samples = Efield.get_number_of_samples()
        sampling_rate = Efield.get_sampling_rate()
        
        # Calculate reflection coefficients and time delay
        fresnel_r_p = get_fresnel_r_p(incoming_zenith, n_index)
        fresnel_r_s = get_fresnel_r_s(incoming_zenith, n_index)
        distance_traveled = 2 * antenna_height / np.cos(incoming_zenith)
        time_delay = distance_traveled / c
        time_shift_samples = int(time_delay * sampling_rate)

        # Calculate the un-shifted reflected traces
        reflected_p_unshifted = original_traces[1] * fresnel_r_p
        reflected_s_unshifted = original_traces[2] * fresnel_r_s

        # Create new zero-filled arrays for the time-shifted reflected signal
        # This guarantees they are the same size as the original.
        ef_trace_p_reflected = np.zeros(num_samples)
        ef_trace_s_reflected = np.zeros(num_samples)

        # Determine how many samples of the reflected signal will fit in the trace window
        if time_shift_samples >= num_samples:
            logger.warning(
                f"Reflection time delay ({time_delay / units.ns:.1f} ns) is greater than or equal to the trace length. "
                f"The entire reflected signal is outside the window."
            )
        else:
            # Calculate the portion of the original signal that will not be cut off
            num_samples_to_copy = num_samples - time_shift_samples
            
            # This is the key check: warn the user that the signal is being cut.
            logger.info(
                f"Applying reflection with {time_delay / units.ns:.1f} ns time delay. "
                f"{time_shift_samples} samples will be truncated from the end of the reflected pulse."
            )
            
            # Copy the valid part of the reflected signal into the time-shifted array
            ef_trace_p_reflected[time_shift_samples:] = reflected_p_unshifted[:num_samples_to_copy]
            ef_trace_s_reflected[time_shift_samples:] = reflected_s_unshifted[:num_samples_to_copy]

        # Create the final combined trace reflected
        final_traces = np.array([
            original_traces[0],  # Longitude component remains unchanged
            ef_trace_p_reflected,
            ef_trace_s_reflected
        ])

        Efield.set_trace(final_traces, sampling_rate)
        return Efield

    def getVoltageFFTFromEfield(self, Efield, original_zenith_antenna, azimuth, det, sim_station, channel_id):
        # origninal zenith antenna needs to be in radians, and the angle from above
        zenith_antenna_after_reflection = np.pi - original_zenith_antenna


        ff = Efield.get_frequencies()
        if self.__caching:
            if self.__freqs is None:
                self.__freqs = ff
            elif len(self.__freqs) != len(ff) or not np.allclose(self.__freqs, ff, rtol=0, atol=0.01 * units.MHz):
                self.__freqs = ff
                self._get_cached_antenna_response.cache_clear()
                logger.warning("Frequencies have changed. Clearing antenna response cache.")

        antenna_model = det.get_antenna_model(sim_station.get_id(), channel_id, zenith_antenna_after_reflection)
        antenna_pattern = self.__antenna_provider.load_antenna_pattern(antenna_model)
        antenna_orientation = det.get_antenna_orientation(sim_station.get_id(), channel_id)

        if self.__caching:
            vel = self._get_cached_antenna_response(antenna_pattern, zenith_antenna_after_reflection, azimuth, *antenna_orientation)
        else:
            vel = antenna_pattern.get_antenna_response_vectorized(ff, zenith_antenna_after_reflection, azimuth, *antenna_orientation)

        Efield_fft = Efield.get_frequency_spectrum()
        t_theta = 1
        t_phi = 1
        vel_array = np.array([vel['theta'] * t_theta, vel['phi'] * t_phi])
        voltage_fft = np.sum(vel_array * np.array([Efield_fft[1], Efield_fft[2]]), axis=0)
        voltage_fft[np.where(ff < 5 * units.MHz)] = 0
        return voltage_fft