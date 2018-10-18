#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 dataBundle.json (johnN)" 1>&2
  exit 1
fi
dataset="$1"
if [ -z "${2-}" ]
then
  queue="john"
else
  queue="host=john${2}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run data.json:${dataset} :eval_squad.py :nltk_data :save :bi-att-flow-dev :glove.6B.100d.txt 'export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH" ; bi-att-flow-dev/basic/run_single.sh data.json single-server.json; python eval_squad.py data.json single-server.json -o eval.json' --request-docker-image minjoon/tf4:latest -n run -d "BiDAF Single, ${dataset}" --request-cpus 1 --request-queue ${queue}
