#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${1:-$PROJECT_ROOT/images/noisy/1024X1024_noise.jpg}"
TRIALS="${TRIALS:-5}"
OUTPUT_CSV="${OUTPUT_CSV:-$PROJECT_ROOT/benchmark_results.csv}"
MPI_OUTPUT_CSV="${MPI_OUTPUT_CSV:-$PROJECT_ROOT/benchmark_results_mpi_hybrid.csv}"

SERIAL_EXE="$PROJECT_ROOT/serial/denoise_serial_edge"
OPENMP_EXE="$PROJECT_ROOT/openmp/denoise_openmp_edge"
PTHREADS_EXE="$PROJECT_ROOT/pthreads/denoise_pthreads_edge"
MPI_EXE="$PROJECT_ROOT/mpi/denoise_mpi_edge"
HYBRID_EXE="$PROJECT_ROOT/hybrid/denoise_hybrid"

extract_time() {
  awk 'BEGIN{IGNORECASE=1} /execution time:/ {for (i=1; i<=NF; i++) if ($i ~ /^[0-9.]+$/) {print $i; exit}}'
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    exit 1
  fi
}

run_and_record() {
  local csv="$1"
  local impl="$2"
  local config="$3"
  local trial="$4"
  shift 4

  echo "[RUN] $impl config=$config trial=$trial"
  local output
  if ! output="$("$@" 2>&1)"; then
    echo "$output" >&2
    echo "[ERROR] $impl failed for config=$config trial=$trial" >&2
    return 1
  fi

  local elapsed
  elapsed="$(printf '%s\n' "$output" | extract_time)"
  if [[ -z "$elapsed" ]]; then
    echo "$output" >&2
    echo "[ERROR] Could not parse execution time for $impl config=$config trial=$trial" >&2
    return 1
  fi

  printf '%s,%s,%s,%s\n' "$impl" "$config" "$trial" "$elapsed" >> "$csv"
}

for exe in "$SERIAL_EXE" "$OPENMP_EXE" "$PTHREADS_EXE" "$MPI_EXE" "$HYBRID_EXE"; do
  require_file "$exe"
done
require_file "$IMAGE"

cd "$PROJECT_ROOT" || exit 1

printf 'Implementation,Threads,Trial,ExecutionTime_ms\n' > "$OUTPUT_CSV"
printf 'Implementation,Config,Trial,ExecutionTime_ms\n' > "$MPI_OUTPUT_CSV"

echo "Benchmark image: $IMAGE"
echo "Trials per configuration: $TRIALS"
echo "Trial 0 is kept as warm-up data; exclude it for report averages."

for trial in $(seq 0 "$TRIALS"); do
  run_and_record "$OUTPUT_CSV" "Serial" "1" "$trial" "$SERIAL_EXE" "$IMAGE" || exit 1
done

for threads in 1 2 4 8 16; do
  for trial in $(seq 0 "$TRIALS"); do
    run_and_record "$OUTPUT_CSV" "OpenMP" "$threads" "$trial" "$OPENMP_EXE" "$IMAGE" "$threads" || exit 1
  done
done

for threads in 1 2 4 8 16; do
  for trial in $(seq 0 "$TRIALS"); do
    run_and_record "$OUTPUT_CSV" "Pthreads" "$threads" "$trial" "$PTHREADS_EXE" "$IMAGE" "$threads" || exit 1
  done
done

for processes in 1 2 4 8; do
  for trial in $(seq 0 "$TRIALS"); do
    run_and_record "$MPI_OUTPUT_CSV" "MPI" "$processes" "$trial" mpirun -np "$processes" "$MPI_EXE" "$IMAGE" || exit 1
  done
done

for config in 1x1 1x4 2x2 2x4 4x2 4x4; do
  processes="${config%x*}"
  threads="${config#*x}"
  for trial in $(seq 0 "$TRIALS"); do
    run_and_record "$MPI_OUTPUT_CSV" "Hybrid" "$config" "$trial" mpirun -np "$processes" "$HYBRID_EXE" "$IMAGE" "$threads" || exit 1
  done
done

echo "Wrote CPU/shared-memory results to: $OUTPUT_CSV"
echo "Wrote MPI/Hybrid results to: $MPI_OUTPUT_CSV"
