#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset.csv> <process_counts...>" >&2
  exit 1
fi

data_file="$1"
shift

if [[ ! -f "$data_file" ]]; then
  echo "Dataset not found: $data_file" >&2
  exit 1
fi

for np in "$@"; do
  echo "== Running with $np processes =="
  mpirun -np "$np" ./bin/parallel_spotify "$data_file"
  echo
  sleep 1
done
