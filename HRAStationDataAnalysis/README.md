# HRA Station Data Analysis

This folder contains scripts for analyzing High Radio Frequency Arianna (HRA) station data. The analysis pipeline follows a systematic workflow for processing, filtering, and analyzing neutrino detection events from multiple HRA stations.

## Configuration Files

- **`config.ini`** - Main configuration file containing processing parameters, simulation paths, and detector layout information
- **`cuts.ini`** - Configuration file for various data quality cuts and filtering parameters

## Workflow Scripts

### Phase 1: Data Conversion and Preprocessing

**`HRADataConvertToNpy.py`** - Converts raw HRA station data from .nur format to numpy arrays for analysis. Includes SNR calculations, chi parameter computation, and blackout time filtering.

**`batchHRADataConversion.py`** - Batch processing script for converting multiple station data files using SLURM job scheduling.

### Phase 2: Event Search and Quality Cuts (C00 Series)

**`C00_eventSearchCuts.py`** - Initial event search and quality cuts pipeline. Applies various filtering criteria to identify potential neutrino events and creates cut masks for further analysis.

**`batchEventSearchCuts.py`** - Batch processing version of the event search cuts for multiple stations and dates.

### Phase 3: Coincidence Analysis (C01 Series)

**`C01_coincidenceEventSearch.py`** - Searches for coincidence events between multiple stations within a one-second time window. Groups events by temporal proximity and tracks station participation.

**`C01B_specificCoincidenceSearch.py`** - Specialized coincidence search for specific event types or targeted analysis.

### Phase 4: Parameter Enhancement (C02 Series)

**`C02_coincidenceParameterAdding.py`** - Adds additional parameters and metadata to identified coincidence events for enhanced analysis capabilities.

**`C02A_coincidenceRecalcZenAzi.py`** - Recalculates zenith and azimuth angles for coincidence events. Computation of all events is computationally intensive and not useful for the many noise events in data.

**`C02B_coincidenceCalcPolarization.py`** - Calculates polarization parameters for coincidence events.

### Phase 5: Visualization and Analysis (C03 Series)

**`C03_coincidenceEventPlotting.py`** - Creates comprehensive plots and visualizations of coincidence events, including waveforms, parameter distributions, and event characteristics.

## Specialized Analysis Scripts

### Statistical Analysis (S Series)

**`S01_plotSNRChiComparisons.py`** - Generates comparative plots of Signal-to-Noise Ratio (SNR) and chi parameters between different datasets, comparing real data with simulations.

**`S02_plotEventsPassingSNRChiCuts.py`** - Visualizes events that pass SNR and chi quality cuts.

### Utility and Helper Scripts

**`C_utils.py`** - Common utility functions used across the C-series scripts, including time masking and data manipulation functions.

**`calculateChi.py`** - Functions for calculating chi parameters used in signal quality assessment.

**`calcVrmsPerStation.py`** - Calculates voltage RMS values per station for noise characterization.

**`checkData.py`** - Data integrity and quality checking utilities.

**`loadHRAConvertedData.py`** - Helper functions for loading previously converted HRA data files.

**`searchNurForEvents.py`** - Search utilities for finding specific events in .nur data files.

### Deep Learning and Advanced Analysis

**`simpleCutForDL.py`** - Simplified data cuts and preprocessing specifically designed for deep learning workflows.

## Data Files

**`vrms_per_station.txt`** - Text file containing voltage RMS values for each station, used for noise normalization and SNR calculations.

## Typical Workflow

1. **Data Conversion**: Run `HRADataConvertToNpy.py` to convert raw .nur files to numpy format
2. **Quality Cuts**: Apply `C00_eventSearchCuts.py` to filter events based on quality criteria
3. **Coincidence Search**: Use `C01_coincidenceEventSearch.py` to find multi-station events
4. **Parameter Addition**: Enhance events with `C02_coincidenceParameterAdding.py` and related scripts
5. **Visualization**: Generate plots and analysis with `C03_coincidenceEventPlotting.py`
6. **Statistical Analysis**: Compare results using S-series scripts

## Dependencies

The scripts rely on several key packages:
- NuRadioReco framework for radio detection analysis
- NumPy and matplotlib for data processing and visualization
- TensorFlow/Keras for machine learning components
- configparser for configuration management
- Various scientific Python libraries (scipy, sklearn, etc.)

## Notes

- Scripts are designed to work with the SLURM job scheduling system for batch processing
- Configuration files should be updated with appropriate paths and parameters before running
- The analysis pipeline generates intermediate data files that are reused in subsequent steps
- Many scripts include caching mechanisms to avoid recomputing expensive operations
