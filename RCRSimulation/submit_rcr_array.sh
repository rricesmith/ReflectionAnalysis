#!/bin/bash
# ============================================================================
# RCR Simulation Array Job Submitter
# ============================================================================
# Submits a single simulation as a SLURM array job. Each array task processes
# 50 CoREAS files. Run one simulation at a time to avoid filesystem overload.
#
# Usage:
#   bash RCRSimulation/submit_rcr_array.sh <sim_name> [flags]
#
# Flags:
#   --test              Small test mode (files 100-300)
#   --dry-run           Print what would be submitted without submitting
#   --n-cores N         Override n_cores_production from config.ini
#   --min-energy E      Skip events with log10(E/eV) < E (e.g., 18.0)
#   --run-suffix TAG    Append TAG to output names (avoids filename clashes)
#   --numpy-dir DIR     Override numpy output directory
#   --output-dir DIR    Override .nur output directory
#   --save-nur          Save .nur event files (off by default to save disk)
#
# Examples:
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m          # production
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --test   # small test
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_MB_576m --dry-run
#   bash RCRSimulation/submit_rcr_array.sh Gen2_deep_SP_300m --n-cores 2000 \
#       --min-energy 18.0 --run-suffix highstats --numpy-dir RCRSimulation/output/02.14.26/numpy/
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
CLI_N_CORES=""
MIN_ENERGY=""
RUN_SUFFIX=""
CLI_NUMPY_DIR=""
CLI_OUTPUT_DIR=""
SAVE_NUR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --n-cores) CLI_N_CORES="$2"; shift ;;
        --min-energy) MIN_ENERGY="$2"; shift ;;
        --run-suffix) RUN_SUFFIX="$2"; shift ;;
        --numpy-dir) CLI_NUMPY_DIR="$2"; shift ;;
        --output-dir) CLI_OUTPUT_DIR="$2"; shift ;;
        --save-nur) SAVE_NUR=true ;;
    esac
    shift
done

# Read simulation size settings from config.ini
CONFIG_FILE="RCRSimulation/config.ini"
cfg_val() { grep "^$1" "$CONFIG_FILE" 2>/dev/null | tail -1 | sed 's/.*= *//' | tr -d ' '; }
CFG_N_CORES_PROD=$(cfg_val n_cores_production)
CFG_N_CORES_TEST=$(cfg_val n_cores_test)
CFG_FILES_PER_JOB=$(cfg_val files_per_job)
# Apply defaults if config values are empty
N_CORES_PROD=${CFG_N_CORES_PROD:-100}
N_CORES_TEST=${CFG_N_CORES_TEST:-50}
FILES_PER_JOB_CFG=${CFG_FILES_PER_JOB:-50}

# CLI --n-cores overrides config value
if [ -n "${CLI_N_CORES}" ]; then
    N_CORES_PROD=${CLI_N_CORES}
    echo "n_cores_production overridden to ${N_CORES_PROD} via --n-cores"
fi

# Lookup simulation parameters: STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MAX_FILE IS_DIRECT
case $SIM_NAME in
    # Direct simulations (layer_depth=surface, layer_dB=0, no multi-dB)
    HRA_MB_direct)
        STATION_TYPE=HRA; DEPTH=shallow; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/HRA_shallow_direct.json; MAX_FILE=1000; IS_DIRECT=true
        LAYER_DB_LIST="" ;;
    Gen2_deep_MB_direct)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_deep_direct.json; MAX_FILE=1000; IS_DIRECT=true
        LAYER_DB_LIST="" ;;
    Gen2_shallow_MB_direct)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=MB; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_shallow_direct.json; MAX_FILE=1000; IS_DIRECT=true
        LAYER_DB_LIST="" ;;
    Gen2_deep_SP_direct)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_direct.json; MAX_FILE=2100; IS_DIRECT=true
        LAYER_DB_LIST="" ;;
    Gen2_shallow_SP_direct)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=surface; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_direct.json; MAX_FILE=2100; IS_DIRECT=true
        LAYER_DB_LIST="" ;;

    # Reflected simulations (combined configs)
    # LAYER_DB_LIST: comma-separated dB values for multi-reflectivity sweep
    #   SP: 40,45,50,55 dB (R_amp = 0.01, 0.0056, 0.0032, 0.0018)
    #   MB: 0,1.5,3.0 dB (R_power ≈ 1.0, 0.7, 0.5)
    HRA_MB_576m)
        STATION_TYPE=HRA; DEPTH=shallow; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false
        LAYER_DB_LIST="0,1.5,3.0" ;;
    Gen2_deep_MB_576m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false
        LAYER_DB_LIST="0,1.5,3.0" ;;
    Gen2_shallow_MB_576m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=MB; LAYER_DEPTH=-576; LAYER_DB=1.7; ATTEN=MB_freq
        CONFIG=RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json; MAX_FILE=1000; IS_DIRECT=false
        LAYER_DB_LIST="0,1.5,3.0" ;;
    Gen2_deep_SP_300m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-300; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_300m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    Gen2_deep_SP_500m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-500; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_500m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    Gen2_deep_SP_830m)
        STATION_TYPE=Gen2; DEPTH=deep; SITE=SP; LAYER_DEPTH=-830; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_deep_830m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    Gen2_shallow_SP_300m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-300; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_300m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    Gen2_shallow_SP_500m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-500; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_500m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    Gen2_shallow_SP_830m)
        STATION_TYPE=Gen2; DEPTH=shallow; SITE=SP; LAYER_DEPTH=-830; LAYER_DB=0; ATTEN=None
        CONFIG=RCRSimulation/configurations/SP/Gen2_shallow_830m_combined.json; MAX_FILE=2100; IS_DIRECT=false
        LAYER_DB_LIST="40,45,50,55" ;;
    *)
        echo "Error: Unknown simulation '$SIM_NAME'"
        echo "Run with no arguments to see available simulations."
        exit 1 ;;
esac

# Test vs production settings
if [ "$TEST_MODE" = true ]; then
    N_CORES=$N_CORES_TEST
    MIN_FILE_START=100
    MAX_FILE=300
    FILES_PER_JOB=200
    N_TASKS=1
    TIME_LIMIT="0-04:00:00"
    echo "=== TEST MODE (files ${MIN_FILE_START}-${MAX_FILE}, ${N_CORES} cores) ==="
else
    MIN_FILE_START=0
    FILES_PER_JOB=$FILES_PER_JOB_CFG
    N_TASKS=$(( (MAX_FILE + FILES_PER_JOB - 1) / FILES_PER_JOB ))
    N_CORES=$N_CORES_PROD
    TIME_LIMIT="1-00:00:00"
    echo "=== PRODUCTION MODE (${N_CORES} cores, ${FILES_PER_JOB} files/job) ==="
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

# CLI overrides for output directories
if [ -n "${CLI_OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${CLI_OUTPUT_DIR}"
fi
if [ -n "${CLI_NUMPY_DIR}" ]; then
    NUMPY_DIR="${CLI_NUMPY_DIR}"
fi

echo "Simulation: ${SIM_NAME}"
echo "  Type: ${STATION_TYPE}, Depth: ${DEPTH}, Site: ${SITE}"
echo "  Layer: ${LAYER_DEPTH}, dB: ${LAYER_DB}"
if [ -n "${LAYER_DB_LIST}" ]; then
    echo "  Multi-dB sweep: ${LAYER_DB_LIST}"
fi
echo "  Config: ${CONFIG}"
echo "  Files: ${MIN_FILE_START}-${MAX_FILE}, ${FILES_PER_JOB} per task, ${N_TASKS} tasks"
echo "  Cores: ${N_CORES}, Distance: ${DISTANCE_KM} km"
echo "  Array spec: ${ARRAY_SPEC}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Numpy:  ${NUMPY_DIR}"
if [ -n "${MIN_ENERGY}" ]; then
    echo "  Min energy: 10^${MIN_ENERGY} eV"
fi
if [ -n "${RUN_SUFFIX}" ]; then
    echo "  Run suffix: ${RUN_SUFFIX}"
fi

if [ "$DRY_RUN" = true ]; then
    # Build sample output name
    DRY_OUTPUT_NAME="${SIM_NAME}_part0"
    if [ -n "${RUN_SUFFIX}" ]; then
        DRY_OUTPUT_NAME="${DRY_OUTPUT_NAME}_${RUN_SUFFIX}"
    fi
    echo ""
    echo "[DRY-RUN] Would submit array job with ${N_TASKS} tasks"
    echo "[DRY-RUN] Sample command for task 0:"
    echo "  python RCRSimulation/S01_RCRSim.py ${DRY_OUTPUT_NAME} \\"
    echo "    --station-type ${STATION_TYPE} --station-depth ${DEPTH} --site ${SITE} \\"
    echo "    --propagation by_depth --detector-config ${CONFIG} \\"
    echo "    --n-cores ${N_CORES} --distance-km ${DISTANCE_KM} \\"
    echo "    --min-file ${MIN_FILE_START} --max-file $((MIN_FILE_START + FILES_PER_JOB)) --seed 0 \\"
    echo "    --layer-depth ${LAYER_DEPTH} --layer-db ${LAYER_DB} \\"
    echo "    --attenuation-model ${ATTEN} --add-noise \\"
    echo "    --output-folder ${OUTPUT_DIR} --numpy-folder ${NUMPY_DIR}"
    if [ -n "${LAYER_DB_LIST}" ]; then
        echo "    --layer-db-list ${LAYER_DB_LIST}"
    fi
    if [ -n "${MIN_ENERGY}" ]; then
        echo "    --min-energy-log10 ${MIN_ENERGY}"
    fi
    exit 0
fi

# Build optional CLI flags for S01_RCRSim.py
EXTRA_ARGS=""
if [ -n "${LAYER_DB_LIST}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --layer-db-list ${LAYER_DB_LIST}"
fi
if [ -n "${MIN_ENERGY}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --min-energy-log10 ${MIN_ENERGY}"
fi
if [ "${SAVE_NUR}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save-nur"
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

# Build output name with optional run suffix
RUN_SUFFIX_VAL="${RUN_SUFFIX}"
if [ -n "\${RUN_SUFFIX_VAL}" ]; then
    OUTPUT_NAME="${SIM_NAME}_part\${SLURM_ARRAY_TASK_ID}_\${RUN_SUFFIX_VAL}"
else
    OUTPUT_NAME="${SIM_NAME}_part\${SLURM_ARRAY_TASK_ID}"
fi

python RCRSimulation/S01_RCRSim.py \\
    \${OUTPUT_NAME} \\
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
    --numpy-folder ${NUMPY_DIR} ${EXTRA_ARGS}

echo "Task \${SLURM_ARRAY_TASK_ID} complete at \$(date)"
EOF

echo ""
echo "Created batch script: ${BATCH_SCRIPT}"
sbatch ${BATCH_SCRIPT}
echo "Submitted: RCR_${SIM_NAME} (${N_TASKS} array tasks)"
echo "Monitor with: squeue -u \$USER"
