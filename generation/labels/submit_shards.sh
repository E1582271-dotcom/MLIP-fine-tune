#!/bin/bash
# Step 4 -- label the 27 snapshots with one CPU job per snapshot (generation/labels/label_qe.pbs).
# This is how the Gamma-point labels (data/dft/raw_inputs/labelled_gamma.xyz) were produced: Gamma
# point, conv_thr 1e-7, 16 MPI ranks per job, npool 1. Measured: one SCF takes about 24 min on 16
# cores (about 120 s per iteration, about 12 iterations, ecutrho 480 Ry -> 135^3 FFT grid); with
# npool 1 the parallel FFT stops scaling at about 16 ranks, so more cores per job do not help.
# Throughput comes from running the 27 single-snapshot jobs concurrently; all 27 finished in about
# 1.8 h with no failure. The 2x2x2 k-point labels were submitted with other settings
# (generation/README.md, step 4).
#
# Each shard has its own --workdir and --out (generation/output/labels/, ignored by git), so shards
# never race; each is resumable. Merge them afterwards in the order that the published label file
# has (see generation/README.md, step 4), because the seeded split of step 5 depends on that order.
#
# Run from the repository root; site-specific qsub options (project, queue) go in QSUB_EXTRA:
#   bash generation/labels/submit_shards.sh
#   QSUB_EXTRA='-P <project>' bash generation/labels/submit_shards.sh
set -euo pipefail
NTOT=${NTOT:-27}
START=${START:-0}
NCPUS=${NCPUS:-16}
CONVTHR=${CONVTHR:-1e-7}   # Ry, on the total energy; forces good to about 1 meV/A or better
WALL=${WALL:-01:00:00}
echo "[shards] NTOT=$NTOT START=$START NCPUS=$NCPUS CONVTHR=$CONVTHR WALL=$WALL"
n=0
for s in $(seq "$START" $((NTOT-1))); do
  # paths are relative to the repository root, where label_qe.pbs runs python
  PW="--start $s --limit 1 --kpts gamma --conv-thr $CONVTHR --workdir generation/output/labels/qe_work_s$s --out generation/output/labels/labelled_s$s.xyz"
  # shellcheck disable=SC2086  # QSUB_EXTRA may hold several options
  jid=$(qsub ${QSUB_EXTRA:-} -l walltime="$WALL" \
             -l select=1:ncpus="$NCPUS":mpiprocs="$NCPUS":mem=24gb \
             -N "li6_qe_s$s" \
             -v "NCPUS=$NCPUS,NPOOL=1,PWARGS=$PW" \
             generation/labels/label_qe.pbs)
  echo "[shards] cfg $s -> $jid"
  n=$((n+1))
done
echo "[shards] submitted $n single-snapshot jobs ($NCPUS cores each, Gamma point)"
