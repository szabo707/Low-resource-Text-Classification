#!/bin/bash

for entry in "contracts_by_relevance"/*.jsonl
do
  qsub -q cpu.q -cwd -b y -l mem_free=32G,h_data=32G python3 create_prefinal_dataset.py $entry
done
