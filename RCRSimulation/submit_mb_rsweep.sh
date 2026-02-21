#!/bin/bash
# ============================================================================
# MB R-Sweep Batch Submitter
# ============================================================================
# Submits all MB 576m simulations with the R-sweep and A/B error flags.
# Uses the same SLURM array approach as submit_rcr_batch.sh.
#
# Usage:
#   bash RCRSimulation/submit_mb_rsweep.sh [flags]
#   bash RCRSimulation/submit_mb_rsweep.sh HRA_MB_576m [flags]
#
# Flags:
#   --all               Submit all 3 MB sims (default if no sims specified)
#   --test              Small test mode (files 100-300 per sim)
#   --dry-run           Print what would be submitted without submitting
#   --max-concurrent N  Max tasks running at once (default: 40)
#   --n-cores N         Override n_cores_production from config.ini
#   --min-energy E      Skip events with log10(E/eV) < E
#   --run-suffix TAG    Append TAG to output names
#   --numpy-dir DIR     Override numpy output directory
#   --output-dir DIR    Override .nur output directory
#   --save-nur          Save .nur event files
#   --r-sweep VALUES    Override R sweep values (default: 0.5,0.75,0.82,0.89,1.0)
#   --no-ab-error       Disable A/B error variants (enabled by default)
#
# Examples:
#   bash RCRSimulation/submit_mb_rsweep.sh                          # all 3 MB sims
#   bash RCRSimulation/submit_mb_rsweep.sh --test --dry-run
#   bash RCRSimulation/submit_mb_rsweep.sh HRA_MB_576m --dry-run
#   bash RCRSimulation/submit_mb_rsweep.sh --r-sweep "0.82,1.0"    # fewer R values
# ============================================================================

set -e

# ---- All MB simulation names ----
ALL_SIMS=(
    HRA_MB_576m
    Gen2_deep_MB_576m
    Gen2_shallow_MB_576m
)

# ---- Defaults ----
SIMS=()
TEST_MODE=false
DRY_RUN=false
MAX_CONCURRENT=40
CLI_N_CORES=""
MIN_ENERGY=""
RUN_SUFFIX=""
CLI_NUMPY_DIR=""
CLI_OUTPUT_DIR=""
SAVE_NUR=false
R_SWEEP="0.5,0.75,0.82,0.89,1.0"
AB_ERROR=true

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --all) SIMS=("${ALL_SIMS[@]}") ;;
        --test) TEST_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift ;;
        --n-cores) CLI_N_CORES="$2"; shift ;;
        --min-energy) MIN_ENERGY="$2"; shift ;;
        --run-suffix) RUN_SUFFIX="$2"; shift ;;
        --numpy-dir) CLI_NUMPY_DIR="$2"; shift ;;
        --output-dir) CLI_OUTPUT_DIR="$2"; shift ;;
        --save-nur) SAVE_NUR=true ;;
        --r-sweep) R_SWEEP="$2"; shift ;;
        --no-ab-error) AB_ERROR=false ;;
        --*) echo "Unknown flag: $1"; exit 1 ;;
        *) SIMS+=("$1") ;;
    esac
    shift
done

# Default to all MB sims if none specified
if [ ${#SIMS[@]} -eq 0 ]; then
    SIMS=("${ALL_SIMS[@]}")
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

if [ -n "${CLI_N_CORES}" ]; then
    N_CORES_PROD=${CLI_N_CORES}
    echo "n_cores_production overridden to ${N_CORES_PROD} via --n-cores"
fi

# ---- Lookup simulation parameters ----
# Returns: STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MAX_FILE
lookup_sim() {
    local sim=$1
    case $sim in
        HRA_MB_576m)
            echo "HRA shallow MB -576 1.7 MB_freq RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json 1000" ;;
        Gen2_deep_MB_576m)
            echo "Gen2 deep MB -576 1.7 MB_freq RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json 1000" ;;
        Gen2_shallow_MB_576m)
            echo "Gen2 shallow MB -576 1.7 MB_freq RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json 1000" ;;
        *)
            echo "UNKNOWN"; return 1 ;;
    esac
}

# ---- Build task list ----
DATE_TAG=$(date +%m.%d.%y)
OUTPUT_DIR="/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/${DATE_TAG}/"
NUMPY_DIR="RCRSimulation/output/${DATE_TAG}/numpy/"
LOG_DIR="RCRSimulation/logs/${DATE_TAG}/"

if [ -n "${CLI_OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${CLI_OUTPUT_DIR}"
fi
if [ -n "${CLI_NUMPY_DIR}" ]; then
    NUMPY_DIR="${CLI_NUMPY_DIR}"
fi

TASK_LINES=()
TOTAL_TASKS=0

echo "============================================"
echo "MB R-Sweep Batch Submission"
echo "============================================"
echo "R sweep:   ${R_SWEEP}"
echo "AB error:  ${AB_ERROR}"
echo "--------------------------------------------"

for SIM_NAME in "${SIMS[@]}"; do
    PARAMS=$(lookup_sim "$SIM_NAME")
    if [ "$PARAMS" = "UNKNOWN" ]; then
        echo "Error: Unknown simulation '$SIM_NAME'"
        exit 1
    fi

    read -r STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG SIM_MAX_FILE <<< "$PARAMS"

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

    DISTANCE_KM=5

    echo "  ${SIM_NAME}: ${N_TASKS} tasks (files ${MIN_FILE_START}-${SIM_MAX_FILE}, ${FILES_PER_JOB}/task)"

    for (( t=0; t<N_TASKS; t++ )); do
        CHUNK_MIN=$(( MIN_FILE_START + t * FILES_PER_JOB ))
        CHUNK_MAX=$(( CHUNK_MIN + FILES_PER_JOB ))
        if [ $CHUNK_MAX -gt $SIM_MAX_FILE ]; then CHUNK_MAX=$SIM_MAX_FILE; fi
        SEED=$t
        # Tab-separated: SIM_NAME STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MIN_FILE MAX_FILE N_CORES DISTANCE_KM SEED
        TASK_LINES+=("${SIM_NAME}	${STATION_TYPE}	${DEPTH}	${SITE}	${LAYER_DEPTH}	${LAYER_DB}	${ATTEN}	${CONFIG}	${CHUNK_MIN}	${CHUNK_MAX}	${N_CORES}	${DISTANCE_KM}	${SEED}")
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
if [ -n "${MIN_ENERGY}" ]; then
    echo "Min energy: 10^${MIN_ENERGY} eV"
fi
if [ -n "${RUN_SUFFIX}" ]; then
    echo "Run suffix: ${RUN_SUFFIX}"
fi
echo "============================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY-RUN] Task list (${TOTAL_TASKS} tasks):"
    for (( i=0; i<${#TASK_LINES[@]}; i++ )); do
        echo "  [$i] ${TASK_LINES[$i]}"
    done
    echo ""
    echo "[DRY-RUN] R sweep: ${R_SWEEP}"
    echo "[DRY-RUN] AB error: ${AB_ERROR}"
    echo "[DRY-RUN] Would submit array job with spec: ${ARRAY_SPEC}"
    exit 0
fi

# ---- Create directories and task list file ----
mkdir -p "${NUMPY_DIR}"
mkdir -p "${LOG_DIR}"

TASK_LIST_FILE="${LOG_DIR}/mb_rsweep_task_list.tsv"
printf "%s\n" "${TASK_LINES[@]}" > "${TASK_LIST_FILE}"
echo "Task list written to: ${TASK_LIST_FILE}"

# ---- Build extra args string for the sbatch script ----
# R-sweep and AB error are always passed (core purpose of this script)
RSWEEP_ARGS="--r-sweep ${R_SWEEP}"
if [ "$AB_ERROR" = true ]; then
    RSWEEP_ARGS="${RSWEEP_ARGS} --ab-error"
fi

# ---- Generate and submit SLURM batch script ----
BATCH_SCRIPT="${LOG_DIR}/mb_rsweep_batch.sbatch"
cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=MB_rsweep
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --output=${LOG_DIR}/mb_rsweep_%A_%a.out
#SBATCH --error=${LOG_DIR}/mb_rsweep_%A_%a.err
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

IFS=\$'\\t' read -r SIM_NAME STATION_TYPE DEPTH SITE LAYER_DEPTH LAYER_DB ATTEN CONFIG MIN_FILE MAX_FILE N_CORES DISTANCE_KM SEED <<< "\$LINE"

# Build optional extra args
EXTRA_ARGS=""
MIN_ENERGY_FLAG="${MIN_ENERGY}"
if [ -n "\${MIN_ENERGY_FLAG}" ]; then
    EXTRA_ARGS="\${EXTRA_ARGS} --min-energy-log10 \${MIN_ENERGY_FLAG}"
fi
SAVE_NUR_FLAG="${SAVE_NUR}"
if [ "\${SAVE_NUR_FLAG}" = "true" ]; then
    EXTRA_ARGS="\${EXTRA_ARGS} --save-nur"
fi

# Build output name with optional run suffix
RUN_SUFFIX_VAL="${RUN_SUFFIX}"
if [ -n "\${RUN_SUFFIX_VAL}" ]; then
    OUTPUT_NAME="\${SIM_NAME}_part\${SEED}_\${RUN_SUFFIX_VAL}"
else
    OUTPUT_NAME="\${SIM_NAME}_part\${SEED}"
fi

echo "Starting MB R-sweep task \${SLURM_ARRAY_TASK_ID}: \${SIM_NAME} (files \${MIN_FILE}-\${MAX_FILE}) at \$(date)"

mkdir -p ${NUMPY_DIR}

python RCRSimulation/S01_RCRSim.py \\
    \${OUTPUT_NAME} \\
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
    --numpy-folder ${NUMPY_DIR} \\
    ${RSWEEP_ARGS} \${EXTRA_ARGS}

echo "Task \${SLURM_ARRAY_TASK_ID} (\${SIM_NAME} part\${SEED}) complete at \$(date)"
EOF

echo "Created batch script: ${BATCH_SCRIPT}"
sbatch "${BATCH_SCRIPT}"
echo "Submitted: MB_rsweep (${TOTAL_TASKS} total tasks, %${MAX_CONCURRENT} concurrent)"
echo "Monitor with: squeue -u \$USER"
