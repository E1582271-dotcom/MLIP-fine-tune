#!/bin/bash
# Download the Li6PS5Cl AIMD records of Deng et al. (2017) at a pinned commit of
# materialsvirtuallab/matgenb (BSD-3-Clause) and check their sha256 (amendment A3).
#   ingest/fetch_aimd.sh <folder>   then   python -m mlipft ingest aimd --source <folder>
set -euo pipefail
OUT=${1:?usage: fetch_aimd.sh <folder>}
COMMIT=2e77d515f23c507887856c265cdab84fa1a30082
mkdir -p "$OUT"
cd "$OUT"
for T in 600 800 1000 1200; do
  curl -sfL -o "$T.json" "https://raw.githubusercontent.com/materialsvirtuallab/matgenb/$COMMIT/notebooks/aimd_data/$T.json"
done
shasum -a 256 -c - <<SUMS
9157f2128ee88a508ca0560989d9b0c401f612d5c59d61a277ae89ca4a5deabe  600.json
fb31dce75edb856ae03a3fb0138c61dd7b8bb111501c2af2d661de0cf87c093a  800.json
bf4ea1d1104279d422fa460e5d4c50dedeb163e846ea8cbbead285f2378a0701  1000.json
a5d283ca17110e17d1350bc1593ad43f45974ab54047959598d3964861f8ca49  1200.json
SUMS
