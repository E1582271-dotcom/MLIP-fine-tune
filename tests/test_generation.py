"""The code in generation/ and src/mlipft/simulation/ is the code that produced the July 2026 data.

None of these tests needs torch, mace or mattersim. The Lennard-Jones smoke test and the split
test need ase, the structure rebuild needs pymatgen, and the tests on the label files need
data/dft/; each is skipped where its requirement is missing.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from mlipft import constants as C

REPO = Path(__file__).resolve().parents[1]
SIMULATION = REPO / "src" / "mlipft" / "simulation"
GENERATION = REPO / "generation"
README = GENERATION / "README.md"
STRUCTURES = REPO / "data" / "structures"
DFT_RAW = REPO / "data" / "dft" / "raw_inputs"
DFT_SETS = REPO / "data" / "dft" / "training_sets"
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))}

# src/mlipft/simulation/md.py: sha256 of the whole file that ran (working repository, src/md.py at
# commit e2b7416; the same bytes since commit 250c37a) and of its bytes after the module docstring.
LEGACY_MD_SHA256 = "a773248ae302b8fec18c15cbf5caf78c3e7d21efc687c74ce94576eecc8aa0be"
MD_CODE_SHA256 = "87d15adc817c7c4709371234c5d05c2661a9010c15681795611a2be7b3a76d9a"
# The same for the code without the Lennard-Jones branch (commit cd51da0, 2026-06-30), used by
# the runs before commit 7d28d41 (2026-07-02).
MD_CODE_BEFORE_LJ_SHA256 = "d9dcb75bcd6f63fab70d67c10dc85330d8f21f900a2d0fc36324961a68b69598"
LJ_BRANCH = (
    ('SUPPORTED = ("mace", "mattersim", "lj")', 'SUPPORTED = ("mace", "mattersim")'),
    ("{'mace','mattersim','lj'}.", "{'mace','mattersim'}."),
    ("    - lj        : ASE Lennard-Jones -- a CPU plumbing fixture ONLY (same convention as\n"
     "                  project 3's stability.py): energies are meaningless, never a physics result.\n", ""),
    ('    if mlip == "lj":                      # before pick_device: no torch on the laptop\n'
     "        from ase.calculators.lj import LennardJones\n\n        return LennardJones()\n", ""),
)

# Ewald energies (eV) of the 52-atom orderings behind the committed cells (July structure records).
EWALD_EV = {"config0.cif": -1010.681, "config1.cif": -1000.657, "doped_cl100_sc222.cif": -1010.681,
            "doped_cl125_sc222.cif": -986.64, "doped_cl150_sc222.cif": -963.141, "doped_cl175_sc222.cif": -939.374}


def _md_code() -> str:
    text = (SIMULATION / "md.py").read_text(encoding="utf-8")
    return text[text.index('"""', 3) + 3:]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _run(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], check=True, env=ENV, cwd=cwd, capture_output=True, text=True)


def _generation_files(*patterns: str) -> list[Path]:
    """Files under generation/, without the (git-ignored) outputs."""
    files = {p for pattern in patterns for p in GENERATION.rglob(pattern)}
    return sorted(p for p in files if p.relative_to(GENERATION).parts[0] != "output")


# --- (a) the MD code ------------------------------------------------------------------------
def test_md_code_is_the_code_that_ran():
    assert _sha(_md_code()) == MD_CODE_SHA256
    docstring = (SIMULATION / "md.py").read_text(encoding="utf-8").split('"""')[1]
    assert LEGACY_MD_SHA256 in docstring and "7d28d41" in docstring


def test_md_code_without_the_lj_branch_is_the_earlier_version():
    code = _md_code()
    for new, old in LJ_BRANCH:
        assert code.count(new) == 1, new
        code = code.replace(new, old)
    assert _sha(code) == MD_CODE_BEFORE_LJ_SHA256


# --- (b) everything compiles; the MD code imports without the GPU stack ----------------------
def _python_files() -> list[Path]:
    return sorted([*_generation_files("*.py"), *SIMULATION.glob("*.py")])


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_python_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)


SHELL_FILES = _generation_files("*.sh", "*.pbs")


LAYOUT = {
    "src/mlipft/simulation": ["__init__.py", "md.py", "structure.py"],
    "generation/structures": ["build_structure.py", "build_doped.py"],
    "generation/md": ["run_md.py"],
    "generation/labels": ["fetch_pseudos.sh", "sample_snapshots.py", "label_qe.py", "split_train_valid.py",
                          "label_qe.pbs", "submit_shards.sh", "qe_scf_template.in"],
    "generation/finetune": ["finetune_mace.sh", "finetune_mace.pbs"],
    "generation/hpc": ["run_md.pbs", "run_in_container.sh", "site.env.example"],
}


def test_every_script_is_checked():
    expected = {REPO / folder / name for folder, names in LAYOUT.items() for name in names}
    assert all(path.is_file() for path in expected)
    checked = {*_python_files(), *SHELL_FILES}
    assert {p for p in expected if p.suffix in {".py", ".sh", ".pbs"}} <= checked


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_shell_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_md_imports_without_torch_or_mace():
    pytest.importorskip("ase")
    code = """
import sys
class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "mace", "mattersim", "pymatgen"}:
            raise ImportError("blocked: " + name)
        return None
sys.meta_path.insert(0, Blocked())
import mlipft.simulation.md as md
assert md.SUPPORTED == ("mace", "mattersim", "lj")
assert type(md.load_calculator("lj")).__name__ == "LennardJones"
assert not {"torch", "mace", "mattersim", "pymatgen"} & {name.split(".")[0] for name in sys.modules}
"""
    _run("-c", code)


def test_run_md_smoke_with_lennard_jones(tmp_path):
    """The driver finds data/structures/config0.cif and writes a trajectory and a run record."""
    pytest.importorskip("ase")
    from ase.io import read

    _run(str(GENERATION / "md" / "run_md.py"), "--mlip", "lj", "--temps", "300", "--steps", "20", "--equilib",
         "10", "--log-every", "5", "--data-dir", str(tmp_path), "--traj-tag", "_smoke")
    record = json.loads((tmp_path / "traj_smoke" / "lj_300K.json").read_text(encoding="utf-8"))
    frames = read(tmp_path / "traj_smoke" / "lj_300K.traj", index=":")
    assert record["cif"] == "config0.cif" and record["equilib_steps"] == 10 and record["friction_per_fs"] == 0.01
    assert len(frames[0]) == 52
    assert record["n_frames"] == len(frames) + 1 == 5   # the record has one extra sample, at the start


# --- (c) the documented settings are the scripts' settings -----------------------------------
def _settings_table() -> list[tuple[str, str, str, str]]:
    rows = re.findall(r"^\| (\d) \| `([\w./]+)` \| `(--\w[\w-]*)` \| `([^`]*)` \|$",
                      README.read_text(encoding="utf-8"), flags=re.M)
    assert len(rows) >= 25
    return rows


def _argparse_defaults(path: Path) -> dict[str, object]:
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    defaults[node.args[0].value] = keyword.value.value
    return defaults


def _shell_option(text: str, option: str) -> str:
    if option == "--swa":
        return "on" if re.search(r"\s--swa\s", text) else "off"
    value = re.search(rf"{option}=\"?([^\"\s\\]+)\"?", text)[1]
    default = re.fullmatch(r"\$\{(\w+):-([^}]*)\}", value)
    if default:                                  # "${LR:-0.0001}"
        return default[2]
    if value.startswith("$"):                    # "$FOUNDATION" with FOUNDATION="${FOUNDATION:-small}"
        name = value[1:]
        return re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', text, flags=re.M)[1]
    return value


def _same(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def test_readme_settings_match_the_scripts():
    for _step, file, option, value in _settings_table():
        path = GENERATION / file
        if path.suffix == ".py":
            actual = _argparse_defaults(path)[option]
        else:
            actual = _shell_option(path.read_text(encoding="utf-8"), option)
        assert _same(actual, value), (file, option, actual, value)


def test_md_defaults_match_the_analysis_constants():
    defaults = _argparse_defaults(GENERATION / "md" / "run_md.py")
    assert defaults["--timestep"] == C.TIME_STEP_FS
    assert defaults["--log-every"] == C.STEP_SKIP
    assert tuple(int(t) for t in defaults["--temps"].split(",")) == C.TEMPERATURES


# --- (d) the DFT label files: merge order and training/validation split ------------------------
MERGE = re.compile(r"for s in ([^;]+); do cat generation/output/labels/(labelled(?:_k222)?)_s\$s\.xyz; "
                   r"done > generation/output/labels/\2\.xyz")


def _merge_orders() -> dict[str, list[int]]:
    orders = {}
    for loop, name in MERGE.findall(README.read_text(encoding="utf-8")):
        order = []
        for first, last, single in re.findall(r"\$\(seq (\d+) (\d+)\)|(\d+)", loop):
            order += list(range(int(first), int(last) + 1)) if first else [int(single)]
        orders[name] = order
    return orders


def _snapshot_ids(path: Path) -> list[tuple[str, int]]:
    """(source trajectory, frame) of every configuration of an extended-XYZ file, without ase."""
    lines = path.read_text(encoding="utf-8").splitlines()
    ids, i = [], 0
    while i < len(lines):
        comment = lines[i + 1]
        source = re.search(r"\bsource=\"?([^\"\s]+)", comment)[1]
        ids.append((source, int(re.search(r"\bframe_in_traj=(\d+)", comment)[1])))
        i += int(lines[i]) + 2
    return ids


def test_both_label_sets_are_merged_in_the_same_order():
    orders = _merge_orders()
    assert set(orders) == {"labelled", "labelled_k222"}
    assert orders["labelled"] == orders["labelled_k222"]
    assert sorted(orders["labelled"]) == list(range(27))


@pytest.mark.parametrize("name", ["labelled_gamma.xyz", "labelled_k222.xyz"])
def test_merge_order_reproduces_the_published_label_order(name):
    """Shard i labels snapshot i of sample_snapshots.py (files in name order, frames 100..900)."""
    path = DFT_RAW / name
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} is not in the repository yet")
    sources = sorted(["mace_600K", "mace_800K", "mace_1000K"])
    snapshots = [(source, frame) for source in sources for frame in range(100, 1000, 100)]
    assert [snapshots[i] for i in _merge_orders()["labelled"]] == _snapshot_ids(path)


@pytest.mark.parametrize("labelled, frac_valid, model", [("labelled_gamma.xyz", "0.2", "ftG"),
                                                         ("labelled_k222.xyz", "0.1", "ftK")], ids=["gamma", "k222"])
def test_split_reproduces_the_training_sets(labelled, frac_valid, model, tmp_path):
    files = [DFT_RAW / labelled, DFT_SETS / f"{model}_train.xyz", DFT_SETS / f"{model}_valid.xyz"]
    missing = [str(p.relative_to(REPO)) for p in files if not p.is_file()]
    if missing:
        pytest.skip(f"not in the repository yet: {', '.join(missing)}")
    pytest.importorskip("ase")
    # the Gamma split as documented relies on the default input; the 2x2x2 split names its file
    extra = [] if model == "ftG" else ["--labelled", str(files[0])]
    _run(str(GENERATION / "labels" / "split_train_valid.py"), *extra, "--seed", "0", "--frac-valid", frac_valid,
         "--out-dir", str(tmp_path))
    assert (tmp_path / "train.xyz").read_bytes() == files[1].read_bytes()
    assert (tmp_path / "valid.xyz").read_bytes() == files[2].read_bytes()


# --- (e) optional: the structure builders reproduce data/structures/ ---------------------------
def test_structure_builders_reproduce_the_committed_cells(tmp_path):
    pytest.importorskip("pymatgen.transformations.standard_transformations")
    import numpy as np
    from pymatgen.core import Structure

    def sites(path):
        return sorted((site.specie.symbol, *np.round(site.frac_coords % 1.0, 5)) for site in Structure.from_file(path))

    builders = GENERATION / "structures"
    _run(str(builders / "build_structure.py"), "--out-dir", str(tmp_path))
    _run(str(builders / "build_structure.py"), "--supercell", "2,2,2", "--out-dir", str(tmp_path))
    _run(str(builders / "build_doped.py"), "--out-dir", str(tmp_path))

    energies = {c["cif"]: c["ewald_energy"] for c in json.loads((tmp_path / "structures.json").read_text())["configs"]}
    doped = json.loads((tmp_path / "doped.json").read_text())["compositions"]
    energies.update({c["cif"]: c["ewald_energy_eV"] for c in doped.values()})
    byte_identical = metadata.version("pymatgen").startswith("2026.5")   # the version that wrote them
    for name in ["config0.cif", "config1.cif", "config0_sc222.cif", *(f for f in EWALD_EV if f.startswith("doped"))]:
        assert sites(tmp_path / name) == sites(STRUCTURES / name), name
        if name in EWALD_EV:
            assert energies[name] == EWALD_EV[name], name
        if byte_identical:
            assert (tmp_path / name).read_bytes() == (STRUCTURES / name).read_bytes(), name
