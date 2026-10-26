"""Li6PS5Cl argyrodite cells: S/Cl site disorder, Ewald-ranked orderings, supercells, Cl-excess series.

Ported from the working repository (``src/structure.py`` and the builder functions of
``05_build_doped.py``). The algorithms and defaults are those that produced data/structures/:

* Base cell: Materials Project mp-985592 (Li6PS5Cl, F-43m), an ordered DFT approximant, as its
  conventional standard cell (52 atoms, a = 10.28 A). It is cached as ``Li6PS5Cl_mp-985592.cif``
  (committed in data/structures/), so later runs are offline.
* Free anions (the S2-/Cl- sublattice outside the PS4 tetrahedra) are Cl plus every S that is at
  least ``FREE_ANION_CUTOFF`` (2.6 A) from all P atoms; the geometry is never hard-coded.
* Li6PS5Cl: the free-anion sites are set to 50/50 S2-/Cl-, and ordered approximants are enumerated
  with pymatgen's ``OrderDisorderedStructureTransformation(algo=2)``, ranked by Ewald energy;
  orderings that StructureMatcher finds equivalent to a kept one are skipped. config0 is the
  lowest-Ewald ordering.
* Cl-excess Li(6-x)PS(5-x)Cl(1+x): on the free-anion sites S2- is replaced by Cl- and the charge is
  compensated by Li vacancies (one uniform partial occupancy on every Li site); the lowest-Ewald
  ordering of the same transformation is kept.
* Supercells are diagonal; 2x2x2 gives the 416-atom (408/400/392 for the Cl-excess series) cells.

Changes from the working repository: the Materials Project API key comes only from the explicit
argument or $MP_API_KEY (the fallback to a key file in a sibling project was removed), and helpers
that no step of this study used (``from_cif``, ``to_ase``) were removed. Requires pymatgen
(optional dependency group ``ingest``); mp-api is needed only when the cached CIF is absent.
"""
from __future__ import annotations

import os

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

MP_ID = "mp-985592"                 # Li6PS5Cl, F-43m, ordered DFT approximant
CACHE_CIF = "Li6PS5Cl_mp-985592.cif"
EXPT_A = 9.859                      # experimental cubic lattice parameter (A), for the lattice-bias note
FREE_ANION_CUTOFF = 2.6             # P-S distance (A) separating PS4 sulfurs from free anions
OXI = {"Li": 1, "P": 5, "S": -2, "Cl": -1}
CL_EXCESS_X = (0.0, 0.25, 0.5, 0.75)  # x of Li(6-x)PS(5-x)Cl(1+x) in the published series


# --- base cell -----------------------------------------------------------------
def fetch_from_mp(api_key: str | None = None) -> Structure:
    """Download mp-985592 and return its conventional standard cell (network and API key needed)."""
    key = (api_key or os.environ.get("MP_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("No Materials Project API key: pass api_key=... or set $MP_API_KEY "
                           f"(or place the cached {CACHE_CIF} in the base directory).")
    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        prim = mpr.get_structure_by_material_id(MP_ID)
    return SpacegroupAnalyzer(prim).get_conventional_standard_structure()


def load_base(data_dir: str, api_key: str | None = None) -> Structure:
    """Conventional Li6PS5Cl cell from ``data_dir/CACHE_CIF``; fetched and cached there if absent."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, CACHE_CIF)
    if os.path.exists(path):
        return Structure.from_file(path)
    struct = fetch_from_mp(api_key)
    struct.to(filename=path)
    return struct


# --- Li6PS5Cl: S/Cl disorder and its ordered approximants -------------------------
def free_anion_indices(struct: Structure, cutoff: float = FREE_ANION_CUTOFF):
    """Return (free_S_indices, Cl_indices): the free-anion sublattice.

    Free S are the sulfurs that are not bonded to any P (PS4 sulfurs are within ``cutoff``)."""
    p_idx = [i for i, s in enumerate(struct) if s.specie.symbol == "P"]
    if not p_idx:
        raise ValueError("No P found -- is this an argyrodite cell?")
    dm = struct.distance_matrix
    free_s, cl = [], []
    for i, site in enumerate(struct):
        sym = site.specie.symbol
        if sym == "Cl":
            cl.append(i)
        elif sym == "S" and min(dm[i][p] for p in p_idx) >= cutoff:
            free_s.append(i)
    return free_s, cl


def make_disordered(struct: Structure) -> Structure:
    """Add oxidation states and set every free-anion site to 50/50 S2-/Cl-; checks neutrality."""
    free_s, cl = free_anion_indices(struct)
    dis = struct.copy()
    dis.add_oxidation_state_by_element(OXI)
    for idx in free_s + cl:
        dis.replace(idx, {"S2-": 0.5, "Cl-": 0.5})
    if abs(dis.charge) > 1e-6:
        raise ValueError(f"Disordered cell not neutral (charge={dis.charge:.3f})")
    return dis


def enumerate_orderings(struct: Structure, n_configs: int = 2, pool: int = 16):
    """Up to ``n_configs`` distinct ordered approximants of the S/Cl disorder, lowest Ewald energy
    first (config 0 = the lowest-energy arrangement), and their Ewald energies (eV).

    Oxidation states are removed from the returned structures."""
    dis = make_disordered(struct)
    odt = OrderDisorderedStructureTransformation(algo=2)
    ranked = odt.apply_transformation(dis, return_ranked_list=pool)
    matcher = StructureMatcher()
    distinct, energies = [], []
    for item in ranked:
        s = item["structure"]
        if any(matcher.fit(s, kept) for kept in distinct):
            continue
        distinct.append(s)
        energies.append(item["energy"])
        if len(distinct) >= n_configs:
            break
    out = []
    for s in distinct:
        s = s.copy()
        s.remove_oxidation_states()
        out.append(s)
    return out, energies[: len(out)]


# --- Cl-excess series --------------------------------------------------------------
def make_cl_excess_disordered(base: Structure, x: float):
    """Li(6-x)PS(5-x)Cl(1+x) as a disordered Structure with oxidation states.

    The free-anion sublattice gets the target S2-/Cl- ratio and the Li sublattice the
    vacancy-reduced occupancy. Returns (disordered structure, counts per conventional cell)."""
    free_s, cl = free_anion_indices(base)
    anion_sites = sorted(free_s + cl)
    li_sites = [i for i, s in enumerate(base) if s.specie.symbol == "Li"]
    n_anion, n_li = len(anion_sites), len(li_sites)
    Z = n_anion // 2                       # formula units per cell (2 free-anion sites per formula)
    n_cl = round((1.0 + x) * Z)            # Cl on the free-anion sublattice
    n_free_s = n_anion - n_cl              # remaining free S2-
    n_li_t = round((6.0 - x) * Z)          # Li after vacancy compensation
    if not (0 <= n_free_s <= n_anion and 0 < n_li_t <= n_li):
        raise ValueError(f"x={x} gives non-physical counts (Cl={n_cl}, freeS={n_free_s}, Li={n_li_t})")

    dis = base.copy()
    dis.add_oxidation_state_by_element(OXI)
    for idx in anion_sites:
        dis.replace(idx, {"S2-": n_free_s / n_anion, "Cl-": n_cl / n_anion})
    for idx in li_sites:
        dis.replace(idx, {"Li+": n_li_t / n_li})
    if abs(dis.charge) > 1e-6:
        raise ValueError(f"x={x} disordered cell not neutral (charge={dis.charge:.4f})")
    counts = {"Z": Z, "n_Cl": n_cl, "n_free_S": n_free_s, "n_Li": n_li_t,
              "cl_content": round(n_cl / Z, 4)}
    return dis, counts


def enumerate_doped(base: Structure, x: float, pool: int = 24):
    """Lowest-Ewald ordered approximant of the Cl-excess cell, without oxidation states.

    Returns (structure, Ewald energy in eV, counts)."""
    dis, counts = make_cl_excess_disordered(base, x)
    odt = OrderDisorderedStructureTransformation(algo=2)
    ranked = odt.apply_transformation(dis, return_ranked_list=pool)
    best = ranked[0] if isinstance(ranked, list) else {"structure": ranked, "energy": float("nan")}
    s = best["structure"].copy()
    s.remove_oxidation_states()
    return s, float(best.get("energy", float("nan"))), counts


def cl_tag(cl_content: float) -> str:
    """'cl100', 'cl125', ... from the Cl content per formula unit."""
    return f"cl{int(round(cl_content * 100)):03d}"


# --- supercell and output ------------------------------------------------------------
def make_supercell(struct: Structure, reps=(1, 1, 1)) -> Structure:
    """Diagonal supercell; reps=(2, 2, 2) on the 52-atom conventional cell gives 416 atoms."""
    sc = struct.copy()
    sc.make_supercell(list(reps))
    return sc


def write_structure(struct: Structure, path: str) -> str:
    """Write a CIF or POSCAR (chosen by pymatgen from the extension); returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    struct.to(filename=path)
    return path


def summary(struct: Structure) -> dict:
    """Compact, JSON-friendly description of a cell."""
    n_li = sum(1 for s in struct if s.specie.symbol == "Li")
    a = struct.lattice.abc[0]
    return {
        "formula": struct.composition.reduced_formula,
        "n_atoms": len(struct),
        "n_Li": n_li,
        "abc": [round(x, 4) for x in struct.lattice.abc],
        "angles": [round(x, 2) for x in struct.lattice.angles],
        "volume": round(struct.volume, 2),
        "density_g_cm3": round(float(struct.density), 4),
        "a_vs_expt_pct": round(100 * (a - EXPT_A) / EXPT_A, 2),  # PBE lattice bias
    }
