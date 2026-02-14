#!/bin/bash
# ============================================================================
# RCR Simulation Batch Submitter
# ============================================================================
# Submits MULTIPLE simulations as a SINGLE SLURM array job. This ensures
# global concurrency: when tasks from one simulation finish, freed slots are
# immediately available for remaining tasks from ANY simulation.
#
# Usage:
#   bash RCRSimulation/submit_rcr_batch.sh <sim1> [sim2 ...] [flags]
#   bash RCRSimulation/submit_rcr_batch.sh --all [flags]
#
# Flags:
#   --all               Submit all 14 simulations
#   --test              Small test mode (files 100-300 per sim)
#   --dry-run           Print what would be submitted without submitting
#   --max-concurrent N  Max tasks running at once (default: 40)
#
# Examples:
#   bash RCRSimulation/submit_rcr_batch.sh --all                    # all 14 sims
#   bash RCRSimulation/submit_rcr_batch.sh --all --max-concurrent 20
#   bash RCRSimulation/submit_rcr_batch.sh HRA_MB_576m Gen2_deep_MB_576m
#   bash RCRSimulation/submit_rcr_batch.sh --all --test --dry-run
#
# How it works:
#   1. Builds a task list mapping each array index to (sim, file range)
#   2. Submits one SLURM array job with a single %N concurrency limit
#   3. Each array task reads its assignment from the task list
# ============================================================================

set -e

# ---- All available simulation names ----
ALL_SIMS=(
    # HRA_MB_direct Gen2_deep_MB_direct Gen2_shallow_MB_direct
    # Gen2_deep_SP_direct Gen2_shallow_SP_direct
    HRA_MB_576m Gen2_deep_MB_576m Gen2_shallow_MB_576m
    Gen2_deep_SP_300m Gen2_shallow_SP_300m
    Gen2_deep_SP_500m Gen2_shallow_SP_500m 
    Gen2_deep_SP_830m Gen2_shallow_SP_830m
)

# ---- Parse arguments ----
SIMS=()
TEST_MODE=false
DRY_RUN=false
MAX_CONCURRENT=40

while [[ $# -gt 0 ]]; do
    case $1 in
        --all) SIMS=("${ALL_SIMS[@]}") ;;
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift ;;
        --*) echo "Unknown flag: $1"; exit 1 ;;
        *) SIMS+=("$1") ;;
    esac
    shift
done

if [ ${#SIMS[@]} -eq 0 ]; then
    echo "Error: No simulations specified."
    echo "Usage: $0 <sim1> [sim2 ...] [--all] [--test] [--dry-run] [--max-concurrent N]"
    exit 1
fi

# ---- Read config ----
CONFIG_FILE="RCRSimulation/config.ini"
cfg_val() { grep "^$1" "$CONFIG_FILE" 2>/dev/null | tail -1 | sed 's/.*= *//' | tr -d ' '; }
CFG_N_CORES_PROD=$(cfg_val n_cores_production)
CFG_N_CORES_TEST=$(cfg_val n_cores_test)
CFG_FILES_PER_JOB=$(cfg_val files_per_job)
N_CORES_PROD=${CFG_N_CORES_PROD:-100}
N_CORES_TEST=${CFG_N_CORES_TEST:-50}
FILES_PER_JOB_CFG=${CFG_FILES_PER_JOB:-50}

# ---- Lookup simulation parameters ----
# Returns: STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MAX_FILE IS_DIRECT LAYER_DB_LIST
# LAYER_DB_LIST: comma-separated dB values for multi-reflectivity sweep (or "none" if not used)
lookup_sim() {
    local sim=$1
    case $sim in
        HRA_MB_576m)
            echo "HRA shallow MB -576 1.7 MB_freq RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json 1000 false 0,1.5,3.0" ;;
        Gen2_deep_MB_576m)
            echo "Gen2 deep MB -576 1.7 MB_freq RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json 1000 false 0,1.5,3.0" ;;
        Gen2_shallow_MB_576m)
            echo "Gen2 shallow MB -576 1.7 MB_freq RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json 1000 false 0,1.5,3.0" ;;
        Gen2_deep_SP_300m)
            echo "Gen2 deep SP -300 0 None RCRSimulation/configurations/SP/Gen2_deep_300m_combined.json 2100 false 40,45,50,55" ;;
        Gen2_shallow_SP_300m)
            echo "Gen2 shallow SP -300 0 None RCRSimulation/configurations/SP/Gen2_shallow_300m_combined.json 2100 false 40,45,50,55" ;;
        HRA_MB_direct)
            echo "HRA shallow MB surface 0 MB_freq RCRSimulation/configurations/MB/HRA_shallow_direct.json 1000 true" ;;
        Gen2_deep_MB_direct)
            echo "Gen2 deep MB surface 0 MB_freq RCRSimulation/configurations/MB/Gen2_deep_direct.json 1000 true" ;;
        Gen2_shallow_MB_direct)
            echo "Gen2 shallow MB surface 0 MB_freq RCRSimulation/configurations/MB/Gen2_shallow_direct.json 1000 true" ;;
        Gen2_deep_SP_direct)
            echo "Gen2 deep SP surface 0 None RCRSimulation/configurations/SP/Gen2_deep_direct.json 2100 true" ;;
        Gen2_shallow_SP_direct)
            echo "Gen2 shallow SP surface 0 None RCRSimulation/configurations/SP/Gen2_shallow_direct.json 2100 true" ;;
        Gen2_deep_SP_500m)
            echo "Gen2 deep SP -500 0 None RCRSimulation/configurations/SP/Gen2_deep_500m_combined.json 2100 false 40,45,50,55" ;;
        Gen2_deep_SP_830m)
            echo "Gen2 deep SP -830 0 None RCRSimulation/configurations/SP/Gen2_deep_830m_combined.json 2100 false 40,45,50,55" ;;
        Gen2_shallow_SP_500m)
            echo "Gen2 shallow SP -500 0 None RCRSimulation/configurations/SP/Gen2_shallow_500m_combined.json 2100 false 40,45,50,55" ;;
        Gen2_shallow_SP_830m)
            echo "Gen2 shallow SP -830 0 None RCRSimulation/configurations/SP/Gen2_shallow_830m_combined.json 2100 false 40,45,50,55" ;;
        *)
            echo "UNKNOWN"; return 1 ;;
    esac
}
# Can add in to above the below


# ---- Build the task list ----
# Each line: SIM_NAME STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MIN_FILE MAX_FILE N_CORES DISTANCE_KM SEED LAYER_DB_LIST
DATE_TAG=$(date +%m.%d.%y)
OUTPUT_DIR="/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/${DATE_TAG}/"
NUMPY_DIR="RCRSimulation/output/${DATE_TAG}/numpy/"
LOG_DIR="RCRSimulation/logs/${DATE_TAG}/"

TASK_LINES=()
TOTAL_TASKS=0

echo "============================================"
echo "RCR Batch Submission"
echo "============================================"

for SIM_NAME in "${SIMS[@]}"; do
    PARAMS=$(lookup_sim "$SIM_NAME")
    if [ "$PARAMS" = "UNKNOWN" ]; then
        echo "Error: Unknown simulation '$SIM_NAME'"
        exit 1
    fi

    read -r STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG SIM_MAX_FILE IS_DIRECT LAYER_DB_LIST <<< "$PARAMS"

    # Test vs production settings
    if [ "$TEST_MODE" = true ]; then
        N_CORES=$N_CORES_TEST
        MIN_FILE_START=100
        SIM_MAX_FILE=300
        FILES_PER_JOB=200
        N_TASKS=1
    else
        MIN_FILE_START=0
        FILES_PER_JOB=$FILES_PER_JOB_CFG
        N_TASKS=$(( (SIM_MAX_FILE + FILES_PER_JOB - 1) / FILES_PER_JOB ))
        N_CORES=$N_CORES_PROD
    fi

    # Direct sims use reduced throw area
    DISTANCE_KM=5
    if [ "$IS_DIRECT" = true ]; then
        DISTANCE_KM=2.5
        N_CORES=$(( N_CORES / 4 ))
        if [ $N_CORES -lt 1 ]; then N_CORES=1; fi
    fi

    echo "  ${SIM_NAME}: ${N_TASKS} tasks (files ${MIN_FILE_START}-${SIM_MAX_FILE}, ${FILES_PER_JOB}/task)"

    # Generate one task list entry per chunk
    for (( t=0; t<N_TASKS; t++ )); do
        CHUNK_MIN=$(( MIN_FILE_START + t * FILES_PER_JOB ))
        CHUNK_MAX=$(( CHUNK_MIN + FILES_PER_JOB ))
        if [ $CHUNK_MAX -gt $SIM_MAX_FILE ]; then CHUNK_MAX=$SIM_MAX_FILE; fi
        SEED=$t
        # Tab-separated: SIM_NAME STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MIN_FILE MAX_FILE N_CORES DISTANCE_KM SEED LAYER_DB_LIST
        TASK_LINES+=("${SIM_NAME}	${STATION_TYPE}	${DEPTH}	${SITE}	${LAYER_DEPTH}	${LAYER_DB}	${ATTEN}	${CONFIG}	${CHUNK_MIN}	${CHUNK_MAX}	${N_CORES}	${DISTANCE_KM}	${SEED}	${LAYER_DB_LIST}")
    done

    TOTAL_TASKS=$(( TOTAL_TASKS + N_TASKS ))
done

ARRAY_MAX=$(( TOTAL_TASKS - 1 ))
ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"

if [ "$TEST_MODE" = true ]; then
    TIME_LIMIT="0-04:00:00"
else
    TIME_LIMIT="3-00:00:00"
fi

echo "--------------------------------------------"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Array spec: ${ARRAY_SPEC}"
echo "Output: ${OUTPUT_DIR}"
echo "Numpy:  ${NUMPY_DIR}"
echo "============================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY-RUN] Task list (${TOTAL_TASKS} tasks):"
    for (( i=0; i<${#TASK_LINES[@]}; i++ )); do
        echo "  [$i] ${TASK_LINES[$i]}"
    done
    echo ""
    echo "[DRY-RUN] Would submit array job with spec: ${ARRAY_SPEC}"
    exit 0
fi

# ---- Create directories and task list file ----
mkdir -p "${NUMPY_DIR}"
mkdir -p "${LOG_DIR}"

TASK_LIST_FILE="${LOG_DIR}/batch_task_list.tsv"
printf "%s\n" "${TASK_LINES[@]}" > "${TASK_LIST_FILE}"
echo "Task list written to: ${TASK_LIST_FILE}"

# ---- Generate and submit SLURM batch script ----
BATCH_SCRIPT="${LOG_DIR}/batch_all.sbatch"
cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=RCR_batch
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=${LOG_DIR}/batch_%A_%a.out
#SBATCH --error=${LOG_DIR}/batch_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rricesmi@uci.edu
#SBATCH --array=${ARRAY_SPEC}

export PYTHONPATH=\$NuM:\$PYTHONPATH
export PYTHONPATH=\$Nu:\$PYTHONPATH
export PYTHONPATH=\$Radio:\$PYTHONPATH
module load python/3.8.0

cd \$ReflectiveAnalysis

# Read this task's parameters from the task list
TASK_LIST="${TASK_LIST_FILE}"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\$TASK_LIST")

IFS=\$'\\t' read -r SIM_NAME STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MIN_FILE MAX_FILE N_CORES DISTANCE_KM SEED LAYER_DB_LIST <<< "\$LINE"

# Build optional --layer-db-list flag
EXTRA_ARGS=""
if [ -n "\${LAYER_DB_LIST}" ]; then
    EXTRA_ARGS="--layer-db-list \${LAYER_DB_LIST}"
fi

echo "Starting batch task \${SLURM_ARRAY_TASK_ID}: \${SIM_NAME} (files \${MIN_FILE}-\${MAX_FILE}) at \$(date)"

mkdir -p ${NUMPY_DIR}

python RCRSimulation/S01_RCRSim.py \\
    \${SIM_NAME}_part\${SEED} \\
    --station-type \${STATION_TYPE} \\
    --station-depth \${DEPTH} \\
    --site \${SITE} \\
    --propagation by_depth \\
    --detector-config \${CONFIG} \\
    --n-cores \${N_CORES} \\
    --distance-km \${DISTANCE_KM} \\
    --min-file \${MIN_FILE} \\
    --max-file \${MAX_FILE} \\
    --seed \${SEED} \\
    --layer-depth \${LAYER_DEPTH} \\
    --layer-db \${LAYER_DB} \\
    --attenuation-model \${ATTEN} \\
    --add-noise \\
    --output-folder ${OUTPUT_DIR} \\
    --numpy-folder ${NUMPY_DIR} \${EXTRA_ARGS}

echo "Task \${SLURM_ARRAY_TASK_ID} (\${SIM_NAME} part\${SEED}) complete at \$(date)"
EOF

echo "Created batch script: ${BATCH_SCRIPT}"
sbatch "${BATCH_SCRIPT}"
echo "Submitted: RCR_batch (${TOTAL_TASKS} total tasks, %${MAX_CONCURRENT} concurrent)"
echo "Monitor with: squeue -u \$USER"
