#!/usr/bin/env bash
set -euo pipefail

USER="won0316"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="${BASE_DIR}/submit_ask1_ic50_selenium.sbatch"
PARTS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
CHECK_ITER=10
CHECK_SLEEP=1

submit_to () {
  local part="$1"
  sbatch --parsable -p "${part}" "${SBATCH_FILE}"
}

is_allocated () {
  local jobid="$1"
  local status
  status=$(squeue -u "${USER}" -j "${jobid}" -h -o "%T %R" || true)
  [[ -z "${status}" ]] && return 1
  local state node
  state=$(awk '{print $1}' <<< "${status}")
  node=$(awk '{print $2}' <<< "${status}")
  [[ "${state}" != "PD" && -n "${node}" && "${node}" != "(null)" ]]
}

for p in "${PARTS[@]}"; do
  echo "[INFO] submit to ${p}"
  jid="$(submit_to "${p}")"
  echo "[INFO] job ${jid}"
  for _ in $(seq 1 "${CHECK_ITER}"); do
    sleep "${CHECK_SLEEP}"
    if is_allocated "${jid}"; then
      echo "[OK] running on ${p} (job ${jid})"; exit 0
    fi
  done
  echo "[INFO] cancel ${jid} (no immediate allocation)"; scancel "${jid}" || true
done

echo "[INFO] leave pending on gpu1"
jid="$(submit_to gpu1)"
echo "[INFO] fallback job ${jid}"
