#!/bin/bash
#SBATCH --job-name=RCR_Gen2deep_MB
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=RCRSimulation/logs/gen2_deep_mb_%A_%a.out
#SBATCH --error=RCRSimulation/logs/gen2_deep_mb_%A_%a.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=rricesmi@uci.edu
#SBATCH --array=0-19

# Full production simulation for Gen2 deep at Moore's Bay
# Uses combined direct+reflected configuration with phased array trigger
# Splits 1000 files across 20 array jobs (50 files each)

export PYTHONPATH=$NuM:$PYTHONPATH
export PYTHONPATH=$Nu:$PYTHONPATH
export PYTHONPATH=$Radio:$PYTHONPATH
module load python/3.8.0

cd $ReflectiveAnalysis

# Calculate file range for this array task
FILES_PER_JOB=50
MIN_FILE=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
MAX_FILE=$((MIN_FILE + FILES_PER_JOB - 1))

# Output folder with date
OUTPUT_DATE=$(date +%m.%d.%y)
OUTPUT_DIR="RCRSimulation/output/${OUTPUT_DATE}/"
NUMPY_DIR="${OUTPUT_DIR}numpy/"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${NUMPY_DIR}
mkdir -p RCRSimulation/logs/

echo "Running Gen2 deep MB simulation"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "File range: ${MIN_FILE} - ${MAX_FILE}"

python RCRSimulation/S01_RCRSim.py \
    gen2_deep_mb_part${SLURM_ARRAY_TASK_ID} \
    --station-type Gen2 \
    --station-depth deep \
    --site MB \
    --propagation by_depth \
    --detector-config RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json \
    --n-cores 1000 \
    --distance-km 5 \
    --min-file ${MIN_FILE} \
    --max-file ${MAX_FILE} \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --layer-depth -576 \
    --layer-db 1.7 \
    --attenuation-model MB_freq \
    --add-noise \
    --output-folder ${OUTPUT_DIR} \
    --numpy-folder ${NUMPY_DIR}

echo "Job ${SLURM_ARRAY_TASK_ID} complete!"
