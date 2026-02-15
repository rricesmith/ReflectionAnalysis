#!/bin/bash
# ============================================================================
# Neutrino Comparison Simulation Batch Submitter
# ============================================================================
# Submits neutrino simulations as a SLURM array job for MB and SP sites.
# Each task processes one energy bin for one site using the Gen2 hybrid
# detector (shallow LPDAs + deep phased array).
#
# Usage:
#   bash RCRSimulation/NeutrinoComparisonSimulation/submit_neutrino_batch.sh [flags]
#
# Flags:
#   --site MB|SP|both   Which site(s) to simulate (default: both)
#   --dry-run           Print task list without submitting
#   --test              Use test event files (100 events per bin)
#   --max-concurrent N  Max concurrent SLURM tasks (default: 12)
#
# Prerequisites:
#   Run N00_GenerateEvents.py first to create input HDF5 event lists.
# ============================================================================

set -e

# ---- Configuration ----
SCRIPT_DIR="RCRSimulation/NeutrinoComparisonSimulation"
CONFIG_DIR="${SCRIPT_DIR}/configs"
EVENTS_DIR="${SCRIPT_DIR}/GeneratedEvents"
DATE_TAG=$(date +%m.%d.%y)
HDF5_OUTPUT_DIR="/dfs8/sbarwick_lab/ariannaproject/rricesmi/neutrinoComparison/${DATE_TAG}/"
NUMPY_DIR="${SCRIPT_DIR}/output/${DATE_TAG}/numpy/"
LOG_DIR="${SCRIPT_DIR}/logs/${DATE_TAG}/"

# Energy bins matching N00_GenerateEvents.py
ENERGIES=(
    "1.0000e+17"
    "3.1623e+17"
    "1.0000e+18"
    "3.1623e+18"
    "1.0000e+19"
    "3.1623e+19"
)

# ---- Parse arguments ----
SITES=("MB" "SP")
DRY_RUN=false
TEST_MODE=false
MAX_CONCURRENT=12

while [[ $# -gt 0 ]]; do
    case $1 in
        --site)
            if [ "$2" = "both" ]; then
                SITES=("MB" "SP")
            else
                SITES=("$2")
            fi
            shift ;;
        --dry-run) DRY_RUN=true ;;
        --test) TEST_MODE=true ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift ;;
        --*) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

# ---- Lookup config files per site ----
get_detector_json() {
    local site=$1
    case $site in
        MB) echo "${CONFIG_DIR}/gen2_hybrid_MB.json" ;;
        SP) echo "${CONFIG_DIR}/gen2_hybrid_SP.json" ;;
    esac
}

get_yaml_config() {
    local site=$1
    case $site in
        MB) echo "${CONFIG_DIR}/MB_neutrino.yaml" ;;
        SP) echo "${CONFIG_DIR}/SP_neutrino.yaml" ;;
    esac
}

# ---- Build task list ----
# Each line: SITE ENERGY DETECTOR_JSON YAML_CONFIG INPUT_HDF5
TASK_LINES=()
TOTAL_TASKS=0

echo "============================================"
echo "Neutrino Comparison Batch Submission"
echo "============================================"

for SITE in "${SITES[@]}"; do
    DET_JSON=$(get_detector_json "$SITE")
    YAML_CFG=$(get_yaml_config "$SITE")

    for ENERGY in "${ENERGIES[@]}"; do
        INPUT_HDF5="${EVENTS_DIR}/${SITE}/nu_${SITE}_${ENERGY}.hdf5"

        # Check for partitioned files
        if [ -f "${INPUT_HDF5}.part0000" ]; then
            # Find max part number
            PART_MAX=$(ls "${INPUT_HDF5}".part* 2>/dev/null | wc -l)
            PART_MAX=$((PART_MAX - 1))
            for (( p=0; p<=PART_MAX; p++ )); do
                PART_STR=$(printf "%04d" $p)
                TASK_LINES+=("${SITE}\t${ENERGY}\t${DET_JSON}\t${YAML_CFG}\t${INPUT_HDF5}\t${PART_STR}")
            done
            TOTAL_TASKS=$((TOTAL_TASKS + PART_MAX + 1))
            echo "  ${SITE} E=${ENERGY}: $((PART_MAX + 1)) parts"
        elif [ -f "${INPUT_HDF5}" ]; then
            TASK_LINES+=("${SITE}\t${ENERGY}\t${DET_JSON}\t${YAML_CFG}\t${INPUT_HDF5}\tnone")
            TOTAL_TASKS=$((TOTAL_TASKS + 1))
            echo "  ${SITE} E=${ENERGY}: 1 task"
        else
            echo "  WARNING: ${INPUT_HDF5} not found, skipping"
        fi
    done
done

if [ $TOTAL_TASKS -eq 0 ]; then
    echo "Error: No input files found. Run N00_GenerateEvents.py first."
    exit 1
fi

ARRAY_MAX=$((TOTAL_TASKS - 1))
ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"
TIME_LIMIT="3-00:00:00"

echo "--------------------------------------------"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Array spec: ${ARRAY_SPEC}"
echo "HDF5 output: ${HDF5_OUTPUT_DIR}"
echo "Numpy output: ${NUMPY_DIR}"
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

# ---- Create directories and task list ----
mkdir -p "${NUMPY_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${HDF5_OUTPUT_DIR}"

TASK_LIST_FILE="${LOG_DIR}/neutrino_task_list.tsv"
printf "%s\n" "${TASK_LINES[@]}" > "${TASK_LIST_FILE}"
echo "Task list written to: ${TASK_LIST_FILE}"

# ---- Generate and submit SLURM batch script ----
BATCH_SCRIPT="${LOG_DIR}/neutrino_batch.sbatch"
cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=NuComp
#SBATCH -A sbarwick_lab
#SBATCH -p standard
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=${LOG_DIR}/nu_%A_%a.out
#SBATCH --error=${LOG_DIR}/nu_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rricesmi@uci.edu
#SBATCH --array=${ARRAY_SPEC}

export PYTHONPATH=\$NuM:\$PYTHONPATH
export PYTHONPATH=\$Nu:\$PYTHONPATH
export PYTHONPATH=\$Radio:\$PYTHONPATH
module load python/3.8.0

cd \$ReflectiveAnalysis

# Read task parameters
TASK_LIST="${TASK_LIST_FILE}"
LINE=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\$TASK_LIST")

IFS=\$'\\t' read -r SITE ENERGY DET_JSON YAML_CFG INPUT_HDF5 PART_STR <<< "\$LINE"

echo "Starting task \${SLURM_ARRAY_TASK_ID}: \${SITE} E=\${ENERGY} at \$(date)"

# Build output filenames
OUT_BASE="nu_\${SITE}_\${ENERGY}"
if [ "\${PART_STR}" != "none" ]; then
    OUT_BASE="\${OUT_BASE}_part\${PART_STR}"
fi

PART_ARG=""
if [ "\${PART_STR}" != "none" ]; then
    PART_ARG="--part \${PART_STR}"
fi

mkdir -p ${NUMPY_DIR}

python ${SCRIPT_DIR}/N01_NeutrinoSim.py \\
    \${INPUT_HDF5} \\
    \${DET_JSON} \\
    \${YAML_CFG} \\
    ${HDF5_OUTPUT_DIR}/\${OUT_BASE}.hdf5 \\
    --numpy-folder ${NUMPY_DIR} \\
    \${PART_ARG}

echo "Task \${SLURM_ARRAY_TASK_ID} (\${SITE} E=\${ENERGY}) complete at \$(date)"
EOF

echo "Created batch script: ${BATCH_SCRIPT}"
sbatch "${BATCH_SCRIPT}"
echo "Submitted: NuComp (${TOTAL_TASKS} total tasks, %${MAX_CONCURRENT} concurrent)"
echo "Monitor with: squeue -u \$USER"
