#!/bin/bash
# Step 4 (preparation) -- install the four PBE pseudopotentials of the July labels into
# generation/output/pseudo/ (ignored by git), where label_qe.py finds them by element prefix.
# Run from the repository root on a machine with internet access (compute nodes are often offline):
#   bash generation/labels/fetch_pseudos.sh
#
# The July labels used GBRV ultrasoft PBE potentials: li_pbe_v1.4, p_pbe_v1.5, s_pbe_v1.4 and
# cl_pbe_v1.4 (.uspp.F.UPF). The four files in GBRV's public archive all_pbe_UPF_v1.5.tar.gz are
# byte for byte the files the July runs read; the script checks them against the sha256 sums in
# data/dft/pseudopotentials.csv and stops on any difference.
set -euo pipefail
cd "$(dirname "$0")/../.."
PDIR="generation/output/pseudo"; mkdir -p "$PDIR"
URL="${GBRV_TAR_URL:-https://www.physics.rutgers.edu/gbrv/all_pbe_UPF_v1.5.tar.gz}"
FILES="li_pbe_v1.4.uspp.F.UPF p_pbe_v1.5.uspp.F.UPF s_pbe_v1.4.uspp.F.UPF cl_pbe_v1.4.uspp.F.UPF"

missing=0
for f in $FILES; do [ -f "$PDIR/$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
    tarball=$(mktemp)
    trap 'rm -f "$tarball"' EXIT
    echo "[pseudo] downloading $URL"
    curl -fL "$URL" -o "$tarball"
    # shellcheck disable=SC2086
    tar -xzf "$tarball" -C "$PDIR" $FILES
fi

status=0
for f in $FILES; do
    expected=$(awk -F, -v f="$f" '$2 == f {print $4}' data/dft/pseudopotentials.csv)
    actual=$( (sha256sum "$PDIR/$f" 2>/dev/null || shasum -a 256 "$PDIR/$f") | cut -d' ' -f1)
    if [ "$expected" = "$actual" ]; then echo "[pseudo] $f OK"; else echo "[pseudo] $f sha256 MISMATCH" >&2; status=1; fi
done
exit $status
