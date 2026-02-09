# RCRSimulation

Reflected Cosmic Ray (RCR) simulation pipeline for Chapter 4. Simulates cosmic ray
radio signals through direct and reflected pathways using NuRadioReco, evaluates
detector triggers (phased array for deep, high/low threshold for shallow), and
produces event rate analysis plots.

All commands below assume you are in the `ReflectiveAnalysis/` directory.

---

## Quick Start: Run a Test Simulation

```bash
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --test
```

This submits a single SLURM array task that processes CoREAS files 100-300
(~100 actual files due to sparse numbering) with 50 random core positions each.
Takes roughly 30-60 minutes. You'll receive email at job start, end, and on failure.

Monitor status:
```bash
squeue -u $USER
```

Preview without submitting:
```bash
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --test --dry-run
```

---

## Checking Results

### SLURM Logs

Bash output (job timing, file ranges) goes to `.out` files.
Python logging (trigger diagnostics, warnings) goes to `.err` files.

```bash
# Job timing and file ranges
cat RCRSimulation/logs/*/Gen2_deep_MB_576m_*.out

# Python logging: trigger parameters, per-event diagnostics, summary
cat RCRSimulation/logs/*/Gen2_deep_MB_576m_*.err | head -200
```

Look for:
- `PA TRIGGER PARAMS` — one-time dump of trigger configuration
- `TRIGGER_DIAG` — per-event trigger results (energy, max voltage, Vrms ratio)
- `TRIGGER SUMMARY` — total events processed and trigger count

### Combine Part Files

Each SLURM array task produces its own `.npy` file. Combine them:

```bash
python RCRSimulation/S03_RCRCombineResults.py Gen2_deep_MB_576m
```

Output: prints number of events, trigger breakdown (direct/reflected/both/neither),
energy range, and saves a `*_combined_RCReventList.npy` file.

If `--numpy-dir` is not specified, it automatically finds the most recent output directory.

---

## Creating Plots

```bash
python RCRSimulation/RCRAnalysis.py
```

Reads `config.ini` for `numpy_folder` and `save_folder` paths. Processes all
`*combined_RCReventList.npy` files found in the numpy folder.

Produces per-simulation plot folders containing:
- **Event rate vs energy** — direct and reflected, broken out by zenith bin
- **2D energy-zenith histograms** — trigger rate and weighted event rate
- **Event rate vs distance** — radial distribution from station
- **Combined overlay** — direct (solid) + reflected (dashed) on same axes
- **Weighted distributions** — SNR, azimuth, zenith histograms weighted by event rate

---

## Running Full Simulations

Production mode processes all available CoREAS files with 100 random core positions per file:

```bash
bash RCRSimulation/submit_rcr_array.sh <sim_name>
```

- MB simulations: 1000 file indices, 20 array tasks (50 files each), ~3-8 hrs per task
- SP simulations: 2100 file indices, 42 array tasks (50 files each)
- Max 10 concurrent tasks to avoid filesystem overload
- 1-day time limit per task

### Full workflow for a simulation:

```bash
# 1. Submit
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m

# 2. Wait for completion (check email or squeue)
squeue -u $USER

# 3. Combine results
python RCRSimulation/S03_RCRCombineResults.py Gen2_deep_MB_576m

# 4. Generate plots (processes all combined files in numpy_folder)
python RCRSimulation/RCRAnalysis.py
```

---

## Available Simulations

### Direct (5) — surface layer, reduced throw area

| Name | Station | Depth | Site | Max Files |
|------|---------|-------|------|-----------|
| `HRA_MB_direct` | HRA | shallow | MB | 1000 |
| `Gen2_deep_MB_direct` | Gen2 | deep | MB | 1000 |
| `Gen2_shallow_MB_direct` | Gen2 | shallow | MB | 1000 |
| `Gen2_deep_SP_direct` | Gen2 | deep | SP | 2100 |
| `Gen2_shallow_SP_direct` | Gen2 | shallow | SP | 2100 |

### Reflected (9) — subsurface reflective layer

| Name | Station | Depth | Site | Layer | dB Loss | Max Files |
|------|---------|-------|------|-------|---------|-----------|
| `HRA_MB_576m` | HRA | shallow | MB | -576m | 1.7 | 1000 |
| `Gen2_deep_MB_576m` | Gen2 | deep | MB | -576m | 1.7 | 1000 |
| `Gen2_shallow_MB_576m` | Gen2 | shallow | MB | -576m | 1.7 | 1000 |
| `Gen2_deep_SP_300m` | Gen2 | deep | SP | -300m | 0 | 2100 |
| `Gen2_deep_SP_500m` | Gen2 | deep | SP | -500m | 0 | 2100 |
| `Gen2_deep_SP_830m` | Gen2 | deep | SP | -830m | 0 | 2100 |
| `Gen2_shallow_SP_300m` | Gen2 | shallow | SP | -300m | 0 | 2100 |
| `Gen2_shallow_SP_500m` | Gen2 | shallow | SP | -500m | 0 | 2100 |
| `Gen2_shallow_SP_830m` | Gen2 | shallow | SP | -830m | 0 | 2100 |

---

## Common Recipes

### Run all Moore's Bay reflected simulations
```bash
bash RCRSimulation/submit_rcr_array.sh HRA_MB_576m
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m
bash RCRSimulation/submit_rcr_array.sh Gen2_shallow_MB_576m
```

### Run all South Pole simulations
```bash
# Reflected
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_SP_300m
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_SP_500m
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_SP_830m
bash RCRSimulation/submit_rcr_array.sh Gen2_shallow_SP_300m
bash RCRSimulation/submit_rcr_array.sh Gen2_shallow_SP_500m
bash RCRSimulation/submit_rcr_array.sh Gen2_shallow_SP_830m

# Direct
bash RCRSimulation/submit_rcr_array.sh Gen2_deep_SP_direct
bash RCRSimulation/submit_rcr_array.sh Gen2_shallow_SP_direct
```

### Combine and plot all MB results
```bash
python RCRSimulation/S03_RCRCombineResults.py HRA_MB_576m
python RCRSimulation/S03_RCRCombineResults.py Gen2_deep_MB_576m
python RCRSimulation/S03_RCRCombineResults.py Gen2_shallow_MB_576m
python RCRSimulation/RCRAnalysis.py
```

### Debug file loading issues
```bash
# Check what files exist and their energies
python RCRSimulation/debug_sim.py --site MB --file-range 100 200

# Sample energy distribution across full range
python RCRSimulation/debug_sim.py --site MB --file-range 0 20 --sample-every 100

# Inspect detector configs and trigger parameters
python RCRSimulation/debug_sim.py --site MB --file-range 100 110
```

---

## File Reference

| File | Purpose |
|------|---------|
| `S01_RCRSim.py` | Core simulation: signal processing, trigger evaluation, event output |
| `S03_RCRCombineResults.py` | Combine per-task `.npy` outputs into single file per simulation |
| `RCRAnalysis.py` | Analysis pipeline: event rate calculation, all plotting functions |
| `RCREventObject.py` | `RCREvent` dataclass: per-event trigger, SNR, weight storage |
| `submit_rcr_array.sh` | SLURM array job submitter with `--test` and `--dry-run` flags |
| `debug_sim.py` | Diagnostic tool for file loading, energy distribution, trigger params |
| `config.ini` | Global config: folders, trigger thresholds, simulation defaults |
| `configurations/` | Detector JSON configs (MB/ and SP/ subdirectories) |

---

## Configuration

### config.ini

Key settings to update between runs:

```ini
[FOLDERS]
# Date-tagged paths — update these for each new batch
sim_output_folder = /dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/M.D.YY/
numpy_folder = RCRSimulation/output/MM.DD.YY/numpy/
save_folder = RCRSimulation/plots/MM.DD.YY/
```

The `submit_rcr_array.sh` script auto-generates date-tagged paths using `date +%m.%d.%y`,
so the config.ini dates should match (or be updated to point to the desired output).

### Trigger thresholds

Deep stations use phased array trigger with thresholds from the Gen2 TDR reference.
Shallow stations use high/low threshold trigger with sigma values from config.ini:

```ini
trigger_sigma_Gen2_shallow = 3.9498194908011524
trigger_sigma_Gen2_deep = 30.68
trigger_sigma_HRA_shallow = 4.5
```

---

## Notes

- **Sparse files**: MB footprints are not contiguous (indices 7-5699, ~2694 files).
  The simulation skips missing files automatically.
- **SP loading**: South Pole uses IceTop-style energy/sin2 binned loading, different
  from MB's sequential file indexing.
- **Direct sims**: Use 0.25x core count (reduced throw area, 2.5 km vs 5 km radius).
- **Logging**: Python `logging` output goes to SLURM `.err` files, not `.out` files.
  Always check both when debugging.
- **Combined configs**: Reflected simulations use `*_combined.json` detector configs
  that include both direct (station 1) and reflected (station 101) stations.
