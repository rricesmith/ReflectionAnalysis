from NuRadioReco.utilities import units
from NuRadioReco.utilities.geometryUtilities import get_fresnel_r_p, get_fresnel_r_s
import numpy as np

from scipy import constants as scipy_constants
c = scipy_constants.c * units.m / units.s


def modifyEfieldForSurfaceReflection(Efield, incoming_zenith, antenna_height=1*units.m, n_index=1.35):
    # This function takes in NuRadioMC Efield objects
    # And modifies them to account for surface reflection
    # Efield: NuRadioMC Efield object
    # incoming_zenith: Zenith angle of the incoming wave in radians
    # antenna_height: Height of the antenna above the ground in meters
    # n_index: Refractive index of the medium (default is 1.35 for ice, assuming air above)

    # Returns a modified Efield object

    fresnel_r_p = get_fresnel_r_p(incoming_zenith, n_index)
    fresnel_r_s = get_fresnel_r_s(incoming_zenith, n_index)

    distance_traveled = 2 * antenna_height / np.cos(incoming_zenith) # Distance traveled by the wave to the surface and back
    time_delay = distance_traveled / c  # Time delay for the wave to travel

    ef_trace_0 = Efield.get_trace()[0]  # Get, but ignore the first trace
    ef_trace_p = Efield.get_trace()[1]
    ef_trace_s = Efield.get_trace()[2]

    # Modify the traces for the p and s polarizations
    ef_trace_p *= fresnel_r_p
    ef_trace_s *= fresnel_r_s

    # Now shift the traces by the time delay, adding zeros at the start and removing the end beyond the original length
    num_samples = Efield.get_trace().shape[1]
    time_shift_samples = int(time_delay * Efield.get_sampling_rate())

    ef_trace_p = np.roll(ef_trace_p, time_shift_samples)
    ef_trace_s = np.roll(ef_trace_s, time_shift_samples)

    ef_trace_p[:time_shift_samples] = 0  # Set the beginning to zero
    ef_trace_s[:time_shift_samples] = 0  # Set the beginning to zero


    # Put traces together into an array
    modified_traces = np.array([ef_trace_0, ef_trace_p, ef_trace_s])

    # Update the Efield object by adding modified traces to base traces
    Efield.set_trace(Efield.get_trace() + modified_traces, Efield.get_sampling_rate())

    return Efield

