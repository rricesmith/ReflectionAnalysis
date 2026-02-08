#!/bin/bash
# ============================================================================
# RCR Simulation Array Job Submitter
# ============================================================================
# Submits a single simulation as a SLURM array job. Each array task processes
# 50 CoREAS files. Run one simulation at a time to avoid filesystem overload.
#
# Usage:
#   bash RCRSimulation/submit_rcr_array.sh <sim_name> [--test] [--dry-run]
#
# Examples:
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m          # production
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --test   # small test
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --dry-run
#
# Available simulations (14 total):
#   Direct (5):
#     HRA_MB_direct, Gen2_deep_MB_direct, Gen2_shallow_MB_direct,
#     Gen2_deep_SP_direct, Gen2_shallow_SP_direct
#   Reflected (9):
#     HRA_MB_576m, Gen2_deep_MB_576m, Gen2_shallow_MB_576m,
#     Gen2_deep_SP_300m, Gen2_deep_SP_500m, Gen2_deep_SP_830m,
#     Gen2_shallow_SP_300m, Gen2_shallow_SP_500m, Gen2_shallow_SP_830m
# ============================================================================

set -e

SIM_NAME="${1:?Usage: $0 <sim_name> [--test] [--dry-run]}"
shift

# Parse flags
TEST_MODE=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# Lookup simulation parameters: STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MAX_FILE IS_DIRECT
case $SIM_NAME in
    # Direct simulations (layer_depth=surface, layer_dB=0)
    HRA_MB_direct)
        STATION_TYPE=HRA; DEPTH=shallow; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/HRA_shallow_direct.json; MAX_FILE=1000; IS_DIRECT=true ;;
    Gen2_deep_MB_direct)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_deep_direct.json; MAX_FILE=1000; IS_DIRECT=true ;;
    Gen2_shallow_MB_direct)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_shallow_direct.json; MAX_FILE=1000; IS_DIRECT=true ;;
    Gen2_deep_SP_direct)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_direct.json; MAX_FILE=2100; IS_DIRECT=true ;;
    Gen2_shallow_SP_direct)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_direct.json; MAX_FILE=2100; IS_DIRECT=true ;;

    # Reflected simulations (combined configs)
    HRA_MB_576m)
        STATION_TYPE=HRA; DEPTH=shallow; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false ;;
    Gen2_deep_MB_576m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false ;;
    Gen2_shallow_MB_576m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false ;;
    Gen2_deep_SP_300m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-300; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_300m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    Gen2_deep_SP_500m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-500; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_500m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    Gen2_deep_SP_830m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-830; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_830m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    Gen2_shallow_SP_300m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-300; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_300m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    Gen2_shallow_SP_500m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-500; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_500m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    Gen2_shallow_SP_830m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-830; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_830m_combined.json; MAX_FILE=2100; IS_DIRECT=false ;;
    *)
        echo "Error: Unknown simulation '$SIM_NAME'"
        echo "Run with no arguments to see available simulations."
        exit 1 ;;
esac

# Test vs production settings
if [ "$TEST_MODE" = true ]; then
    N_CORES=50
    MIN_FILE_START=600
    MAX_FILE=610
    FILES_PER_JOB=10
    N_TASKS=1
    TIME_LIMIT="0-04:00:00"
    echo "=== TEST MODE (files ${MIN_FILE_START}-${MAX_FILE}) ==="
else
    MIN_FILE_START=0
    FILES_PER_JOB=50
    N_TASKS=$(( (MAX_FILE + FILES_PER_JOB - 1) / FILES_PER_JOB ))
    N_CORES=100
    TIME_LIMIT="1-00:00:00"
    echo "=== PRODUCTION MODE ==="
fi

# Direct sims use reduced throw area (0.5x width = 0.25x cores)
DISTANCE_KM=5
if [ "$IS_DIRECT" = true ]; then
    DISTANCE_KM=2.5
    N_CORES=$(( N_CORES / 4 ))
    if [ $N_CORES -lt 1 ]; then N_CORES=1; fi
fi

ARRAY_MAX=$(( N_TASKS - 1 ))
# Limit concurrent tasks to 10 to avoid filesystem overload
ARRAY_SPEC="0-${ARRAY_MAX}%10"

DATE_TAG=$(date +%m.%d.%y)
OUTPUT_DIR="/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/${DATE_TAG}/"
NUMPY_DIR="RCRSimulation/output/${DATE_TAG}/numpy/"
LOG_DIR="RCRSimulation/logs/${DATE_TAG}/"

echo "Simulation: ${SIM_NAME}"
echo "  Type: ${STATION_TYPE}, Depth: ${DEPTH}, Site: ${SITE}"
echo "  Layer: ${LAYER_DEPTH}, dB: ${LAYER_DB}"
echo "  Config: ${CONFIG}"
echo "  Files: ${MIN_FILE_START}-${MAX_FILE}, ${FILES_PER_JOB} per task, ${N_TASKS} tasks"
echo "  Cores: ${N_CORES}, Distance: ${DISTANCE_KM} km"
echo "  Array spec: ${ARRAY_SPEC}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Numpy:  ${NUMPY_DIR}"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY-RUN] Would submit array job with ${N_TASKS} tasks"
    echo "[DRY-RUN] Sample command for task 0:"
    echo "  python RCRSimulation/S01_RCRSim.py ${SIM_NAME}_part0 \\"
    echo "    --station-type ${STATION_TYPE} --station-depth ${DEPTH} --site ${SITE} \\"
    echo "    --propagation by_depth --detector-config ${CONFIG} \\"
    echo "    --n-cores ${N_CORES} --distance-km ${DISTANCE_KM} \\"
    echo "    --min-file ${MIN_FILE_START} --max-file $((MIN_FILE_START + FILES_PER_JOB)) --seed 0 \\"
    echo "    --layer-depth ${LAYER_DEPTH} --layer-db ${LAYER_DB} \\"
    echo "    --attenuation-model ${ATTEN} --add-noise \\"
    echo "    --output-folder ${OUTPUT_DIR} --numpy-folder ${NUMPY_DIR}"
    exit 0
fi

# Create directories
mkdir -p ${NUMPY_DIR}
mkdir -p ${LOG_DIR}

# Generate and submit SLURM batch script
BATCH_SCRIPT="${LOG_DIR}/${SIM_NAME}.sbatch"
cat > ${BATCH_SCRIPT} << EOF
#!/bin/bash
#SBATCH --job-name=RCR_${SIM_NAME}
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=${LOG_DIR}/${SIM_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${SIM_NAME}_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rricesmi@uci.edu
#SBATCH --array=${ARRAY_SPEC}

export PYTHONPATH=\$NuM:\$PYTHONPATH
export PYTHONPATH=\$Nu:\$PYTHONPATH
export PYTHONPATH=\$Radio:\$PYTHONPATH
module load python/3.8.0

cd \$ReflectiveAnalysis

# Calculate file range for this array task
MIN_FILE_START=${MIN_FILE_START}
FILES_PER_JOB=${FILES_PER_JOB}
MIN_FILE=\$((MIN_FILE_START + SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
MAX_FILE=\$((MIN_FILE + FILES_PER_JOB))
if [ \$MAX_FILE -gt ${MAX_FILE} ]; then MAX_FILE=${MAX_FILE}; fi

mkdir -p ${NUMPY_DIR}

echo "Starting ${SIM_NAME} task \${SLURM_ARRAY_TASK_ID} at \$(date)"
echo "File range: \${MIN_FILE} - \${MAX_FILE}"

python RCRSimulation/S01_RCRSim.py \\
    ${SIM_NAME}_part\${SLURM_ARRAY_TASK_ID} \\
    --station-type ${STATION_TYPE} \\
    --station-depth ${DEPTH} \\
    --site ${SITE} \\
    --propagation by_depth \\
    --detector-config ${CONFIG} \\
    --n-cores ${N_CORES} \\
    --distance-km ${DISTANCE_KM} \\
    --min-file \${MIN_FILE} \\
    --max-file \${MAX_FILE} \\
    --seed \${SLURM_ARRAY_TASK_ID} \\
    --layer-depth ${LAYER_DEPTH} \\
    --layer-db ${LAYER_DB} \\
    --attenuation-model ${ATTEN} \\
    --add-noise \\
    --output-folder ${OUTPUT_DIR} \\
    --numpy-folder ${NUMPY_DIR}

echo "Task \${SLURM_ARRAY_TASK_ID} complete at \$(date)"
EOF

echo ""
echo "Created batch script: ${BATCH_SCRIPT}"
sbatch ${BATCH_SCRIPT}
echo "Submitted: RCR_${SIM_NAME} (${N_TASKS} array tasks)"
echo "Monitor with: squeue -u \$USER"
