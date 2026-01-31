#!/bin/bash
#SBATCH --job-name=RCR_Gen2deep_MB_test
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=RCRSimulation/logs/test_gen2_deep_mb_%j.out
#SBATCH --error=RCRSimulation/logs/test_gen2_deep_mb_%j.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=rricesmi@uci.edu

# Test simulation for Gen2 deep configuration at Moore's Bay
# Uses combined direct+reflected configuration with phased array trigger

export PYTHONPATH=$NuM:$PYTHONPATH
export PYTHONPATH=$Nu:$PYTHONPATH
export PYTHONPATH=$Radio:$PYTHONPATH
module load python/3.8.0

cd $ReflectiveAnalysis

# Create output directories if they don't exist
mkdir -p RCRSimulation/output/test/
mkdir -p RCRSimulation/output/test/numpy/
mkdir -p RCRSimulation/logs/

# Run small test simulation with:
# - 50 cores (instead of 1000)
# - Only first 10 CoREAS files
# - Gen2 deep at MB with combined direct+reflected config
python RCRSimulation/S01_RCRSim.py \
    test_gen2_deep_mb \
    --station-type Gen2 \
    --station-depth deep \
    --site MB \
    --propagation by_depth \
    --detector-config RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json \
    --n-cores 50 \
    --distance-km 5 \
    --min-file 0 \
    --max-file 10 \
    --seed 42 \
    --layer-depth -576 \
    --layer-db 1.7 \
    --attenuation-model MB_freq \
    --add-noise \
    --output-folder RCRSimulation/output/test/ \
    --numpy-folder RCRSimulation/output/test/numpy/

echo "Test simulation complete!"
echo "Output NUR file: RCRSimulation/output/test/test_gen2_deep_mb_*.nur"
echo "Numpy event list: RCRSimulation/output/test/numpy/test_gen2_deep_mb_*_RCReventList.npy"
