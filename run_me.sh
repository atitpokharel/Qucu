#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

rm -f qucu_simulation.qsub_out

qsub submission.pbs

until compgen -G '*.qsub_out' >/dev/null; do
  sleep 2
done

less '+F' ./*.qsub_out


