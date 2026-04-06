#!/bin/bash
#SBATCH --job-name=astra_dr1
#SBATCH --account=desi
#SBATCH --partition=regular
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --array=0-1%1
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/cosmic-web/dr1/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/ASTRA-DESI

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export ASTRA_CLASS_SPLIT_ITER=0
export ASTRA_CLASS_SKIP_COMBINED=0
export ASTRA_PROB_SPLIT_ITER=0
export ASTRA_PROB_SKIP_COMBINED=0
export ASTRA_SKIP_GROUPS=1

R_LOWER=${R_LOWER:--0.25}
R_MED=${R_MED:-0.25}
R_UPPER=${R_UPPER:-0.65}
export PAIR_NJOBS_CAP=${PAIR_NJOBS_CAP:-128}

PROJECT_ROOT=$PWD
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"

CLUSTER_DIR="/global/cfs/cdirs/desi/public/dr1/vac/dr1/lss/guadalupe/v1.0/LSScats/clustering"
SCRATCH_ROOT="/pscratch/sd/v/vtorresg/cosmic-web/dr1"

RAW_OUT="${SCRATCH_ROOT}/raw"
CLASS_ROOT="${SCRATCH_ROOT}"
GROUPS_OUT="${SCRATCH_ROOT}/groups"
PLOTS_OUT="${SCRATCH_ROOT}/figs"
LOGDIR="${SCRATCH_ROOT}/logs"
PAIR_SPILL="${SCRATCH_ROOT}/pair_spill"

mkdir -p "${RAW_OUT}" "${CLASS_ROOT}" "${GROUPS_OUT}" "${PLOTS_OUT}" "${LOGDIR}" "${PAIR_SPILL}"

ZLABELS=("NGC1" "NGC2")
CONFIG_JSON=${CONFIG_JSON:-}

if (( SLURM_ARRAY_TASK_ID >= ${#ZLABELS[@]} )); then
  echo "error: SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds ZLABELS" >&2
  exit 1
fi
ZONE_LABEL=${ZLABELS[SLURM_ARRAY_TASK_ID]}

TRACE_ARGS=()
if [[ -n ${CONFIG_JSON} ]]; then
  TRACE_ARGS+=(--config "${CONFIG_JSON}")
fi

LOG_PREFIX="${LOGDIR}/zone_${ZONE_LABEL}"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOG_PREFIX}.time" \
     python "${MAIN_SCRIPT}" \
       --release DR1 \
       --zones "${ZONE_LABEL}" \
       ${TRACE_ARGS[@]} \
       --base-dir  "${CLUSTER_DIR}" \
       --raw-out   "${RAW_OUT}" \
       --class-out "${CLASS_ROOT}" \
       --groups-out "${GROUPS_OUT}" \
       --plot-output "${PLOTS_OUT}" \
       --spill-dir "${PAIR_SPILL}" \
       --r-lower "${R_LOWER}" \
       --r-med "${R_MED}" \
       --r-upper "${R_UPPER}" \
       --n-random 100