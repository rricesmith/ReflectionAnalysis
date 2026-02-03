#!/bin/bash
# ============================================================================
# RCR Simulation Batch Initialization Script
# ============================================================================
# This script submits all 9 simulation sets needed for Chapter 4:
#   1. HRA MB (shallow) - comparison baseline
#   2. Gen2 deep MB (576m layer)
#   3. Gen2 shallow MB (576m layer)
#   4. Gen2 deep SP (300m layer)
#   5. Gen2 deep SP (500m layer)
#   6. Gen2 deep SP (830m layer)
#   7. Gen2 shallow SP (300m layer)
#   8. Gen2 shallow SP (500m layer)
#   9. Gen2 shallow SP (830m layer)
#
# Usage:
#   bash RCRSimulation/init_all_simulations.sh [--test] [--dry-run]
#
# Options:
#   --test     Run small test simulations (50 cores, 10 files)
#   --dry-run  Print commands without executing
# ============================================================================

set -e

# Parse arguments
TEST_MODE=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# Configuration
DATE_TAG=$(date +%m.%d.%y)
BASE_OUTPUT="/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/${DATE_TAG}/"
NUMPY_OUTPUT="RCRSimulation/output/${DATE_TAG}/numpy/"
LOG_DIR="RCRSimulation/logs/${DATE_TAG}/"

# Test vs Production settings
if [ "$TEST_MODE" = true ]; then
    N_CORES=50
    MAX_FILE_MB=10
    MAX_FILE_SP=10
    ARRAY_JOBS=1
    TIME_LIMIT="0-02:00:00"
    echo "=== TEST MODE: Small simulations ==="
else
    N_CORES=1000
    MAX_FILE_MB=1000
    MAX_FILE_SP=2100
    ARRAY_JOBS=20
    TIME_LIMIT="3-00:00:00"
    echo "=== PRODUCTION MODE: Full simulations ==="
fi

# Create directories
mkdir -p ${NUMPY_OUTPUT}
mkdir -p ${LOG_DIR}

echo "Output directory: ${BASE_OUTPUT}"
echo "Numpy output: ${NUMPY_OUTPUT}"
echo "Log directory: ${LOG_DIR}"
echo ""

# Function to submit a simulation job
submit_simulation() {
    local NAME=$1
    local STATION_TYPE=$2
    local STATION_DEPTH=$3
    local SITE=$4
    local LAYER_DEPTH=$5
    local LAYER_DB=$6
    local ATTEN_MODEL=$7
    local DETECTOR_CONFIG=$8
    local MAX_FILE=$9

    local JOB_NAME="RCR_${NAME}"
    local OUTPUT_NAME="${NAME}"

    # Build the Python command
    local PY_CMD="python RCRSimulation/S01_RCRSim.py ${OUTPUT_NAME} \
        --station-type ${STATION_TYPE} \
        --station-depth ${STATION_DEPTH} \
        --site ${SITE} \
        --propagation by_depth \
        --detector-config ${DETECTOR_CONFIG} \
        --n-cores ${N_CORES} \
        --distance-km 5 \
        --min-file 0 \
        --max-file ${MAX_FILE} \
        --seed 0 \
        --layer-depth ${LAYER_DEPTH} \
        --layer-db ${LAYER_DB} \
        --attenuation-model ${ATTEN_MODEL} \
        --add-noise \
        --output-folder ${BASE_OUTPUT} \
        --numpy-folder ${NUMPY_OUTPUT}"

    echo "----------------------------------------"
    echo "Simulation: ${NAME}"
    echo "  Type: ${STATION_TYPE}, Depth: ${STATION_DEPTH}, Site: ${SITE}"
    echo "  Layer: ${LAYER_DEPTH}m, dB: ${LAYER_DB}, Config: ${DETECTOR_CONFIG}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would submit: ${PY_CMD}"
    else
        # Create SLURM batch script
        local BATCH_SCRIPT="${LOG_DIR}/${NAME}.sbatch"
        cat > ${BATCH_SCRIPT} << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=${LOG_DIR}/${NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_%j.err
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=rricesmi@uci.edu

export PYTHONPATH=\$NuM:\$PYTHONPATH
export PYTHONPATH=\$Nu:\$PYTHONPATH
export PYTHONPATH=\$Radio:\$PYTHONPATH
module load python/3.8.0

cd \$ReflectiveAnalysis

echo "Starting ${NAME} simulation at \$(date)"
${PY_CMD}
echo "Completed ${NAME} simulation at \$(date)"
EOF

        echo "  Created batch script: ${BATCH_SCRIPT}"
        sbatch ${BATCH_SCRIPT}
        echo "  Submitted job: ${JOB_NAME}"
    fi
}

# ============================================================================
# Submit all simulations
# ============================================================================

echo ""
echo "============================================================"
echo "Submitting RCR Simulations - ${DATE_TAG}"
echo "============================================================"

# 1. HRA MB (shallow) - baseline comparison
submit_simulation \
    "HRA_MB_shallow" \
    "HRA" \
    "shallow" \
    "MB" \
    "-576" \
    "1.7" \
    "MB_freq" \
    "RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json" \
    ${MAX_FILE_MB}

# 2. Gen2 deep MB (576m layer)
submit_simulation \
    "Gen2_deep_MB_576m" \
    "Gen2" \
    "deep" \
    "MB" \
    "-576" \
    "1.7" \
    "MB_freq" \
    "RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json" \
    ${MAX_FILE_MB}

# 3. Gen2 shallow MB (576m layer)
submit_simulation \
    "Gen2_shallow_MB_576m" \
    "Gen2" \
    "shallow" \
    "MB" \
    "-576" \
    "1.7" \
    "MB_freq" \
    "RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json" \
    ${MAX_FILE_MB}

# 4. Gen2 deep SP (300m layer)
submit_simulation \
    "Gen2_deep_SP_300m" \
    "Gen2" \
    "deep" \
    "SP" \
    "-300" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_deep_300m_combined.json" \
    ${MAX_FILE_SP}

# 5. Gen2 deep SP (500m layer)
submit_simulation \
    "Gen2_deep_SP_500m" \
    "Gen2" \
    "deep" \
    "SP" \
    "-500" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_deep_500m_combined.json" \
    ${MAX_FILE_SP}

# 6. Gen2 deep SP (830m layer)
submit_simulation \
    "Gen2_deep_SP_830m" \
    "Gen2" \
    "deep" \
    "SP" \
    "-830" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_deep_830m_combined.json" \
    ${MAX_FILE_SP}

# 7. Gen2 shallow SP (300m layer)
submit_simulation \
    "Gen2_shallow_SP_300m" \
    "Gen2" \
    "shallow" \
    "SP" \
    "-300" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_shallow_300m_combined.json" \
    ${MAX_FILE_SP}

# 8. Gen2 shallow SP (500m layer)
submit_simulation \
    "Gen2_shallow_SP_500m" \
    "Gen2" \
    "shallow" \
    "SP" \
    "-500" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_shallow_500m_combined.json" \
    ${MAX_FILE_SP}

# 9. Gen2 shallow SP (830m layer)
submit_simulation \
    "Gen2_shallow_SP_830m" \
    "Gen2" \
    "shallow" \
    "SP" \
    "-830" \
    "0" \
    "None" \
    "RCRSimulation/configurations/SP/Gen2_shallow_830m_combined.json" \
    ${MAX_FILE_SP}

echo ""
echo "============================================================"
echo "All simulations submitted! (9 total)"
echo "Monitor with: squeue -u \$USER"
echo "============================================================"
