#!/usr/bin/env bash
#
# Executa a aplicação MPI com diferentes quantidades de processos para coletar
# métricas de desempenho. A documentação foi escrita em português conforme o
# pedido do projeto.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Uso: $0 <dataset.csv> <quantidades_de_processos...>" >&2
  exit 1
fi

data_file="$1"
shift

if [[ ! -f "$data_file" ]]; then
  echo "Dataset não encontrado: $data_file" >&2
  exit 1
fi

for np in "$@"; do
  echo "== Executando com $np processos =="
  mpirun -np "$np" ./bin/parallel_spotify "$data_file"
  echo
  sleep 1
done
