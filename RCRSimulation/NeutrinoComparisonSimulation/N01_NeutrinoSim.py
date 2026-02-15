"""Neutrino comparison simulation runner.

Runs a NuRadioMC neutrino simulation using the Gen2 hybrid detector
(shallow LPDAs + deep phased array) and extracts per-triggered-event
arrival angle and polarization for comparison with RCR backgrounds.

Based on NeutrinoAnalysis/T02_RunSimulation_Gen2CorePaper.py.

Usage:
    python N01_NeutrinoSim.py <input_hdf5> <detector_json> <config_yaml> <output_hdf5> \
        [--numpy-folder <path>] [--part NNNN]
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import copy
import logging

import numpy as np
import h5py

from NuRadioMC.simulation import simulation
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp

from NeutrinoEvent import NeutrinoEvent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("NeutrinoComparisonSim")

# ============================================================================
# Trigger thresholds — identical to T02_RunSimulation_Gen2CorePaper.py
# ============================================================================

# 4-channel PA thresholds (noise rate -> threshold)
# 100 Hz -> 30.68, 1 Hz -> 38.62, 1 mHz -> 50.53
thresholds_pa = {
    "8ch": {"100Hz": 61.90, "1Hz": 76.83, "1mHz": 99.22},
    "4ch": {"100Hz": 30.68, "1Hz": 38.62, "1mHz": 50.53},
}

# Shallow LPDA thresholds
thresholds = {
    "2/4_100Hz": 3.9498194908011524,
    "2/4_10mHz": 4.919151494949084,
    "fhigh": 0.15,
    "flow": 0.08,
}

# ============================================================================
# Channel groups
# ============================================================================

SHALLOW_CHANNELS = [0, 1, 2, 3]
PA_8CH_CHANNELS = [4, 5, 6, 7, 8, 9, 10, 11]
PA_4CH_CHANNELS = [8, 9, 10, 11]

# ============================================================================
# Initialize detector modules
# ============================================================================

highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=200 * units.ns)

# Phased array beam angles
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(
    np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11)
)
phasing_angles_8ant = np.arcsin(
    np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21)
)

# ============================================================================
# Per-channel filter settings
# ============================================================================

passband_low = {}
passband_low_trigger = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}

# PA channels: cheby1 filters
for channel_id in PA_8CH_CHANNELS:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = "cheby1"
    order_low[channel_id] = 7
    order_high[channel_id] = 4

# Shallow LPDA channels: butter filters
for channel_id in SHALLOW_CHANNELS:
    passband_low[channel_id] = [1 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [1 * units.MHz, thresholds["fhigh"]]
    passband_high[channel_id] = [thresholds["flow"], 800 * units.GHz]
    filter_type[channel_id] = "butter"
    order_low[channel_id] = 10
    order_high[channel_id] = 5


# ============================================================================
# Simulation class
# ============================================================================


class NeutrinoComparisonSim(simulation.simulation):
    """NuRadioMC simulation that extracts arrival angle and polarization."""

    def __init__(self, *args, numpy_output=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._nu_events = []  # Collected NeutrinoEvent objects
        self._numpy_output = numpy_output

    def _detector_simulation_filter_amp(self, evt, station, det):
        """Apply bandpass filters — identical to Gen2CorePaper."""
        channelBandPassFilter.run(
            evt, station, det,
            passband=passband_low, filter_type=filter_type,
            order=order_low, rp=0.1,
        )
        channelBandPassFilter.run(
            evt, station, det,
            passband=passband_high, filter_type=filter_type,
            order=order_high, rp=0.1,
        )

    def _detector_simulation_trigger(self, evt, station, det):
        """Run triggers and extract signal properties from triggered events."""
        # Apply trigger bandpass filter
        channelBandPassFilter.run(
            evt, station, det,
            passband=passband_low_trigger, filter_type=filter_type,
            order=order_low, rp=0.1,
        )

        # Recalculate per-channel Vrms for filtered bandwidth
        Vrms_per_channel_copy = copy.deepcopy(self._Vrms_per_channel)
        ff = np.linspace(0, 1 * units.GHz, 10000)

        for channel_id in range(station.get_number_of_channels()):
            filt = channelBandPassFilter.get_filter(
                ff, station.get_id(), channel_id, det,
                passband=passband_low_trigger, filter_type=filter_type,
                order=order_low, rp=0.1,
            )
            filt *= channelBandPassFilter.get_filter(
                ff, station.get_id(), channel_id, det,
                passband=passband_high, filter_type=filter_type,
                order=order_high, rp=0.1,
            )
            filt *= channelBandPassFilter.get_filter(
                ff, station.get_id(), channel_id, det,
                passband=passband_low, filter_type=filter_type,
                order=order_low, rp=0.1,
            )
            bandwidth = np.trapz(np.abs(filt) ** 2, ff)
            Vrms_per_channel_copy[station.get_id()][channel_id] *= (
                bandwidth / self._bandwidth_per_channel[station.get_id()][channel_id]
            ) ** 0.5

        # ---- SHALLOW TRIGGER: LPDA 2of4 ----
        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station.get_id()):
            threshold_high[channel_id] = (
                thresholds["2/4_100Hz"]
                * Vrms_per_channel_copy[station.get_id()][channel_id]
            )
            threshold_low[channel_id] = (
                -thresholds["2/4_100Hz"]
                * Vrms_per_channel_copy[station.get_id()][channel_id]
            )
        highLowThreshold.run(
            evt, station, det,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            coinc_window=40 * units.ns,
            triggered_channels=SHALLOW_CHANNELS,
            number_concidences=2,
            trigger_name="LPDA_2of4_100Hz",
        )

        # ---- DEEP TRIGGER: Phased array (if hybrid station) ----
        if station.get_number_of_channels() > 5:
            Vrms_PA = Vrms_per_channel_copy[station.get_id()][PA_4CH_CHANNELS[0]]
            det_channel = det.get_channel(station.get_id(), PA_4CH_CHANNELS[0])
            sampling_rate_pa = det_channel["trigger_adc_sampling_frequency"]

            # 8-channel PA
            window_8ant = int(16 * units.ns * sampling_rate_pa * 4.0)
            step_8ant = int(8 * units.ns * sampling_rate_pa * 4.0)

            phasedArrayTrigger.run(
                evt, station, det,
                Vrms=Vrms_PA,
                threshold=thresholds_pa["8ch"]["100Hz"] * np.power(Vrms_PA, 2.0),
                triggered_channels=PA_8CH_CHANNELS,
                phasing_angles=phasing_angles_8ant,
                ref_index=1.75,
                trigger_name="PA_8channel_100Hz",
                trigger_adc=True,
                adc_output="voltage",
                trigger_filter=None,
                upsampling_factor=4,
                window=window_8ant,
                step=step_8ant,
            )

            # 4-channel PA
            window_4ant = int(16 * units.ns * sampling_rate_pa * 2.0)
            step_4ant = int(8 * units.ns * sampling_rate_pa * 2.0)

            phasedArrayTrigger.run(
                evt, station, det,
                Vrms=Vrms_PA,
                threshold=thresholds_pa["4ch"]["100Hz"] * np.power(Vrms_PA, 2.0),
                triggered_channels=PA_4CH_CHANNELS,
                phasing_angles=phasing_angles_4ant,
                ref_index=1.75,
                trigger_name="PA_4channel_100Hz",
                trigger_adc=True,
                adc_output="voltage",
                trigger_filter=None,
                upsampling_factor=2,
                window=window_4ant,
                step=step_4ant,
            )

        # Set triggers on station
        for trigger in station.get_triggers().values():
            station.set_trigger(trigger)

        triggerTimeAdjuster.run(evt, station, det)

        # ---- EXTRACT signal properties from triggered events ----
        if station.has_triggered():
            self._extract_neutrino_data(evt, station)

    def _extract_neutrino_data(self, evt, station):
        """Extract arrival angle and polarization from triggered event."""
        # Neutrino properties from sim shower
        shower = evt.get_sim_shower(0)
        energy = float(shower[shp.energy] / units.eV)
        nu_zenith = float(shower[shp.zenith] / units.rad)
        nu_azimuth = float(shower[shp.azimuth] / units.rad)

        # Get electric fields from sim station
        sim_station = station.get_sim_station()
        if sim_station is None:
            return

        efields = sim_station.get_electric_fields()

        # Separate efield properties by channel group
        lpda_zeniths, lpda_pols = [], []
        pa_zeniths, pa_pols = [], []

        for ef in efields:
            try:
                zen = float(ef[efp.zenith])
            except (KeyError, TypeError):
                continue
            try:
                pol = float(ef[efp.polarization_angle])
            except (KeyError, TypeError):
                pol = None

            ch_ids = ef.get_channel_ids()
            for ch_id in ch_ids:
                if ch_id in SHALLOW_CHANNELS:
                    lpda_zeniths.append(zen)
                    if pol is not None:
                        lpda_pols.append(pol)
                elif ch_id in PA_8CH_CHANNELS:
                    pa_zeniths.append(zen)
                    if pol is not None:
                        pa_pols.append(pol)

        # Which triggers fired
        triggers = []
        for trigger in station.get_triggers().values():
            if trigger.has_triggered():
                triggers.append(trigger.get_name())

        nu_event = NeutrinoEvent(
            event_id=evt.get_id(),
            energy=energy,
            nu_zenith=nu_zenith,
            nu_azimuth=nu_azimuth,
            triggers=triggers,
            lpda_arrival_zenith=float(np.mean(lpda_zeniths)) if lpda_zeniths else None,
            lpda_polarization=float(np.mean(lpda_pols)) if lpda_pols else None,
            pa_arrival_zenith=float(np.mean(pa_zeniths)) if pa_zeniths else None,
            pa_polarization=float(np.mean(pa_pols)) if pa_pols else None,
        )
        self._nu_events.append(nu_event)

    def save_neutrino_events(self, output_hdf5_path):
        """Merge weights from HDF5 output and save NeutrinoEvent numpy array.

        The NuRadioMC HDF5 output contains event weights that encode Earth
        absorption and interaction probability. We read these and attach
        them to our collected NeutrinoEvent objects.
        """
        if not self._nu_events:
            logger.warning("No triggered neutrino events to save")
            return

        # Read weights from the output HDF5
        try:
            with h5py.File(output_hdf5_path, "r") as f:
                weights = np.array(f["weights"])
                for nu_evt in self._nu_events:
                    idx = nu_evt.event_id
                    if idx < len(weights):
                        nu_evt.weight = float(weights[idx])
                    else:
                        logger.warning(
                            "Event %d index out of range (weights has %d entries)",
                            idx, len(weights),
                        )
        except Exception as e:
            logger.warning("Could not read weights from %s: %s", output_hdf5_path, e)

        # Save as numpy object array
        if self._numpy_output:
            os.makedirs(os.path.dirname(self._numpy_output), exist_ok=True)
            npy_array = np.array(self._nu_events, dtype=object)
            np.save(self._numpy_output, npy_array, allow_pickle=True)
            logger.info(
                "Saved %d neutrino events to %s",
                len(self._nu_events), self._numpy_output,
            )
            print(f"Saved {len(self._nu_events)} neutrino events to {self._numpy_output}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run neutrino comparison simulation"
    )
    parser.add_argument("inputfilename", type=str, help="NuRadioMC input HDF5")
    parser.add_argument("detectordescription", type=str, help="Detector JSON")
    parser.add_argument("config", type=str, help="NuRadioMC YAML config")
    parser.add_argument("outputfilename", type=str, help="HDF5 output filename")
    parser.add_argument(
        "--numpy-folder", type=str, default=None,
        help="Folder for numpy output (NeutrinoEvent arrays)",
    )
    parser.add_argument("--part", type=str, default="None")
    args = parser.parse_args()

    # Handle file partitioning
    input_file = args.inputfilename
    output_file = args.outputfilename
    if args.part != "None":
        input_file += f".part{args.part}"
        output_file += f".part{args.part}"

    # Determine numpy output path
    numpy_output = None
    if args.numpy_folder:
        base = os.path.splitext(os.path.basename(output_file))[0]
        numpy_output = os.path.join(args.numpy_folder, f"{base}_neutrino_events.npy")

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Numpy:  {numpy_output}")

    sim = NeutrinoComparisonSim(
        inputfilename=input_file,
        outputfilename=output_file,
        detectorfile=args.detectordescription,
        outputfilenameNuRadioReco=None,  # No NUR output — keep it lightweight
        config_file=args.config,
        file_overwrite=True,
        numpy_output=numpy_output,
    )
    sim.run()

    # Post-simulation: merge weights and save numpy
    sim.save_neutrino_events(output_file)

    n_lpda = sum(1 for e in sim._nu_events if e.has_lpda_trigger())
    n_pa = sum(1 for e in sim._nu_events if e.has_pa_trigger())
    print(f"Summary: {len(sim._nu_events)} triggered events "
          f"({n_lpda} LPDA, {n_pa} PA)")


if __name__ == "__main__":
    main()
