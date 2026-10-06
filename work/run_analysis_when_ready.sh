#!/usr/bin/env bash
# Wait for the marker OCR process to finish, then run the first-pass analysis.
set -uo pipefail
cd /mnt/sharedOs/blood-electric-workspace
mkdir -p analysis

PID="${1:-$(pgrep -f 'bin/marker work/input' | head -1)}"
echo "waiting for marker pid=$PID"
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20

source /home/goya/anaconda3/etc/profile.d/conda.sh
conda activate marker

MD=docs/markdown/blood-electric/blood-electric.md
if [ -s "$MD" ]; then
  python work/analyze.py "$MD" analysis/first_pass > analysis/first_pass_run.log 2>&1
  echo "ANALYSIS DONE $(date -Is)" >> analysis/first_pass_run.log
else
  echo "OCR OUTPUT MISSING at $(date -Is)" > analysis/ERROR.txt
fi
