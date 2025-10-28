from NuRadioReco.utilities import units
import os
import numpy as np
from scipy import constants


def pullFilesForSimulation(sim_type, min_file=0, max_file=-1):
    """
    Pull in files for simulation
    sim_type == 'IceTop:
        IceTop simulations range from 16.0-18.5 log10eV
        Sin^2(zenith) bins range from 0.0-1.0
        There are ~33 separate footprints per Energy/Sin^2 bin
    sim_type == 'SP':
        Scattered SP simulations, cover less area than IceTop. Suggest using IceTop instead
    sim_type == 'MB':
        Moores Bay simulations
        File range descriptions:
            7-239   : Proton primaries for ARIANNA site (not all with correctly rotated simulation star-grid), semi-random arrival direction and energies, Corsika 7.4005
            240-499 : Proton primaries for ARIANNA site, star corrected, semi-random arrival direction and energies, Corsika 7.4005,
            540-620 : Proton primaries for ARIANNA site, semi-random arrival direction and energies, Corsika 7.4005,
            ---- This is cuttoff for a fast simulation
            1000-1110 : Proton primaries for ARIANNA site, Golden event Dec 21st, different zenith angles, same azimuth, Corsika 7.4005,
            1500 - 2229 : Proton primaries for ARIANNA site, semi-random arrival direction and energies, corsika 7.500,
            2233-2329 : Proton primaries for ARIANNA site, higher zenith angles, corsika 7.500,
            2500-2799 : Proton primaries for ARIANNA site, very high zenith angles, two directions from North for HCR, corsika 7.500,
            2800-3299 : single selected arrival directions, many energies, corsika 7.6300,

            Files 400+ are very specific events trying to recreate found CRs, not good for a general simulation
    sim_type == 'GL':
        Greenland simulations
        Not a great range but useable for simple studies
    """

    i = min_file
    input_files = []
    if sim_type == 'SP':
        if max_file == -1:
            max_file = 2100
        while i < max_file:
            file = 'none'
            if i < 1000:
                file = f'../SPFootprints/000{i:03d}.hdf5'
            else:
                file = f'../SPFootprints/SIM00{i}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1
    elif sim_type == 'MB':
        if max_file == -1:
            max_file = 3999
        while i < max_file:
            file = f'../MBFootprints/00{i:04d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1
    elif sim_type == 'GL':
        if max_file == -1:
            max_file = 600
        while i < max_file:
            file = f'../../../../pub/arianna/SIM/greenland/output/hdf5/SIM{i:06d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1
    elif sim_type == 'IceTop':
        num_icetop=10 
        icetop_sin=-1
        if min_energy < 16.0:
            print(f'Setting IceTop min energy to 16.0 log10eV')
            min_energy = 16.0
        if max_energy == -1 or max_energy > 18.5:
            print(f'Setting IceTop max energy to 18.5 log10eV')
            max_energy = 18.5

        i = min_energy
        while i < max_energy:
            #Currently just iterating through all sin's equally. Can separate sin bins if needed
            if icetop_sin == -1:
                sin2Val = np.arange(0, 1.01, 0.1)
            else:
                sin2Val = [icetop_sin]
            for sin2 in sin2Val:
                num_in_bin = 0
                folder = f'../../../../../dfs8/sbarwick_lab/arianna/SIM/southpole/IceTop/lgE_{i:.1f}/sin2_{sin2:.1f}/'
                for (dirpath, dirnames, filenames) in os.walk(folder):
                    for file in filenames:
                        if num_in_bin == num_icetop:
                            continue
                        if not 'highlevel' in file:
                            file = os.path.join(folder, file)
                            input_files.append(file)
                            num_in_bin += 1
            i += 0.1

        return input_files

    return input_files


def calculateNoisePerChannel(det, station, amp=True, hardwareResponseIncorporator=None, channelBandPassFilter=None):
    """
    Calculate the noise per channel
        det: detector object
        station: station object
        passband: passband settings for trigger
        amp: boolean for whether to include amplifier
        hardwareResponseIncorporator: hardwareResponseIncorporator object, required if amp=True
        channelBandPassFilter: channelBandPassFilter object, required if amp=False
    """


    # Setup conditions for amplifier. Assuming a 200s amplifier
    noise_figure = 1.4
    noise_temp = 400 * units.kelvin
    passband = [50*units.MHz, 1000*units.MHz, 'butter', 2]

    preAmpVrms_per_channel = {}
    postAmpVrms_per_channel = {}
    for channel_id in station.get_channel_ids():
        channel = station.get_channel(channel_id)
        sim_sampling_rate = channel.get_sampling_rate()

        dt = 1 / sim_sampling_rate
        ff = np.linspace(0, 0.5/dt /units.GHz, 10000)
        filt = np.ones_like(ff, dtype=complex)

        if amp:
            filt = hardwareResponseIncorporator.get_filter(ff, station.get_id(), channel_id, det, sim_to_data=True)
        else:
            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[passband[0], passband[1]], filter_type=passband[2], order=passband[3], rp=0.1)

        bandwidth = np.trapz(np.abs(filt)**2, ff)
        Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5

        if amp:
            max_freq = 0.5 / dt
            preAmpVrms_per_channel[channel_id] = Vrms / (bandwidth / max_freq)**0.5
            postAmpVrms_per_channel[channel_id] = Vrms
        else:
            preAmpVrms_per_channel[channel_id] = Vrms
            postAmpVrms_per_channel[channel_id] = Vrms

    return preAmpVrms_per_channel, postAmpVrms_per_channel
