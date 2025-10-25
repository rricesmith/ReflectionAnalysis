RCR Simulation Pipeline
=======================

This directory hosts the single-station reflected cosmic-ray (RCR) simulation
workflow. It mirrors the legacy `HRASimulation` pipeline while simplifying the
interface to always run one station at a time with configurable propagation
(`direct` or `reflected`), station type (`HRA`, `SP`, or `Gen2`), and site
(`MB` or `SP`).

Files
-----

* `config.ini` – Central configuration for the RCR workflow. Defines the
	station/site defaults, CoREAS file ranges, trigger sigmas, attenuation model,
	output folders, detector configuration directories, and batch submission
	settings. The `config_dir` entry points to the directory tree that stores
	station-specific detector JSON files.
* `S01_RCRSim.py` – Simulation entry point. Reads the config (or command line
	overrides), loads the appropriate detector geometry, and runs NuRadioReco to
	produce a `.nur` file plus a compact NumPy event summary for the chosen
	station.
* `S01_RCRBatchJob.py` – Convenience driver that splits the requested CoREAS
	file range into jobs, constructs the `S01_RCRSim.py` command for each chunk,
	and submits the jobs via `A00_SlurmUtil`.
* `S02_RCRSimplePlots.py` – Quick-look plotting script that ingests the NumPy
	event summaries, computes effective areas and event-rate projections, and
	produces weighted histograms of reconstructed directions.
* `R_utils.py` – Reserved for future helper utilities specific to RCR runs.

Typical Usage
-------------

1. Adjust `config.ini` to select the station, propagation mode, site, and file
	 ranges you want to simulate.
2. For an interactive test or debugging run, execute
	 `python RCRSimulation/S01_RCRSim.py <output_name>` from the repository root.
	 Command-line flags allow overriding most configuration values.
3. For large production campaigns, run `python
	 RCRSimulation/S01_RCRBatchJob.py` to generate and submit the batch jobs.
4. After simulations finish, create diagnostic plots with `python
	RCRSimulation/S02_RCRSimplePlots.py`. By default it grabs the most recent
	`*_RCReventList.npy` file from the configured NumPy folder and saves plots under
	`save_folder/simple_plots/`.

Outputs land in the folders configured under `[FOLDERS]`, with `.nur` files and
NumPy summaries organized by station and file range. Generated plots or
follow-up analyses can use the same folder structure.



