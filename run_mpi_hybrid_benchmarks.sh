#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-images/noisy/1024X1024_noise.jpg}"
trials="${TRIALS:-3}"
output_file="${OUTPUT_CSV:-benchmark_results_mpi_hybrid.csv}"
staging_file="${output_file}.tmp"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required benchmark input does not exist: %s\n' "$1" >&2
        exit 1
    fi
}

if ! command -v mpirun >/dev/null 2>&1; then
    printf 'No MPI launcher was found. Install OpenMPI before running benchmarks.\n' >&2
    exit 1
fi

require_file "$image_path"
require_file "./mpi/denoise_mpi_edge"
require_file "./hybrid/denoise_hybrid"

cleanup() {
    rm -f "$staging_file"
}
trap cleanup ERR INT TERM

printf 'Implementation,Config,Trial,ExecutionTime_ms\n' > "$staging_file"

run_one() {
    local implementation="$1"
    local config="$2"
    local processes="$3"
    local executable="$4"
    shift 4

    local trial start end elapsed_ns elapsed output
    for ((trial = 0; trial < trials; trial++)); do
        start="$(date +%s%N)"
        if ! output="$(mpirun --oversubscribe -np "$processes" "$executable" "$@" 2>&1)"; then
            printf '%s\n' "$output" >&2
            printf '%s failed for config %s, trial %s.\n' "$implementation" "$config" "$trial" >&2
            return 1
        fi
        end="$(date +%s%N)"
        elapsed_ns=$((end - start))
        elapsed="$(printf '%d.%02d' "$((elapsed_ns / 1000000))" "$(((elapsed_ns / 10000) % 100))")"
        printf '%s,%s,%s,%s\n' "$implementation" "$config" "$trial" "$elapsed" >> "$staging_file"
        printf '%s config=%s trial=%s wall_ms=%s | %s\n' \
            "$implementation" "$config" "$trial" "$elapsed" "$(printf '%s\n' "$output" | head -n 1)"
    done
}

for processes in 1 2 4 8; do
    run_one "MPI_${processes}procs" "$processes" "$processes" "./mpi/denoise_mpi_edge" "$image_path"
done

for config in 1x1 2x2 4x4 2x4 4x2; do
    processes="${config%x*}"
    threads="${config#*x}"
    run_one "Hybrid_${config}" "$config" "$processes" "./hybrid/denoise_hybrid" "$image_path" "$threads"
done

mv "$staging_file" "$output_file"
trap - ERR INT TERM
printf 'Results saved to %s\n' "$output_file"
