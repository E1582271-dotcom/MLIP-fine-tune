"""Physical constants, the catalogue of MD series and DFT label sets, and reference values."""
from __future__ import annotations

Q_C = 1.602176634e-19
KB_J_K = 1.380649e-23
KB_EV_K = 8.617333262e-5
EV_A3_TO_GPA = 160.21766208
T_REF_K = 300.0
TEMPERATURES = (600, 800, 1000)

# MD settings shared by every run (src/mlipft/simulation/md.py): 1 fs time step, one frame every 50 steps.
TIME_STEP_FS = 1.0
STEP_SKIP = 50
FRAME_PS = TIME_STEP_FS * STEP_SKIP / 1000.0
KINISI_START_FRACTION = 0.3
KINISI_SEEDS = tuple(range(11))  # seed 0 is the primary value, 1-10 measure the Monte-Carlo scatter

# Measured room-temperature conductivity of sintered Li6PS5Cl (Wang et al. 2018); compared with the
# universal potentials only.
EXPERIMENT_mS_cm = 3.15

# One entry per MD series (three runs each, at 600, 800 and 1000 K). ``folder`` and ``prefix`` locate
# the July trajectories in the raw backup (``<folder>/<prefix>_<T>K.traj``); ``published`` names the
# July analysis file and key, or None for series that were never analysed in July.
SERIES = (
    {"id": "mace_52_050", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 50, "part": "A", "folder": "traj", "prefix": "mace",
     "published": ("metrics.json", "mace"), "label": "MACE-MP-0, 52 atoms, 50 ps"},
    {"id": "mattersim_52_050", "potential": "MatterSim", "model": "mattersim", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 50, "part": "A", "folder": "traj", "prefix": "mattersim",
     "published": ("metrics.json", "mattersim"), "label": "MatterSim, 52 atoms, 50 ps"},
    {"id": "mace_52_100", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 100, "part": "A", "folder": "traj_100ps", "prefix": "mace",
     "published": None, "label": "MACE-MP-0, 52 atoms, 100 ps"},
    {"id": "mattersim_52_100", "potential": "MatterSim", "model": "mattersim", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 100, "part": "A", "folder": "traj_100ps",
     "prefix": "mattersim", "published": None, "label": "MatterSim, 52 atoms, 100 ps"},
    {"id": "mace_52_150", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 150, "part": "A", "folder": "traj_long", "prefix": "mace",
     "published": ("metrics_long.json", "mace"), "label": "MACE-MP-0, 52 atoms, 150 ps"},
    {"id": "mattersim_52_150", "potential": "MatterSim", "model": "mattersim", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 52, "production_ps": 150, "part": "A", "folder": "traj_long",
     "prefix": "mattersim", "published": ("metrics_long.json", "mattersim"), "label": "MatterSim, 52 atoms, 150 ps"},
    {"id": "mace_416_200", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 416, "production_ps": 200, "part": "A", "folder": "traj_prod", "prefix": "mace",
     "published": ("metrics_prod.json", "mace"), "label": "MACE-MP-0, 416 atoms, 200 ps"},
    {"id": "mattersim_416_200", "potential": "MatterSim", "model": "mattersim", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 416, "production_ps": 200, "part": "A", "folder": "traj_prod_ms",
     "prefix": "mattersim", "published": ("metrics_prod.json", "mattersim"),
     "label": "MatterSim, 416 atoms, 200 ps"},
    {"id": "ftG_416_200", "potential": "MACE fine-tuned (Gamma labels)", "model": "ftG", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 416, "production_ps": 200, "part": "C", "folder": "traj_ft", "prefix": "mace",
     "published": ("metrics_ft.json", "mace"), "label": "Fine-tuned on Gamma labels, 416 atoms, 200 ps"},
    {"id": "ftK_416_200", "potential": "MACE fine-tuned (2x2x2 labels)", "model": "ftK", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 416, "production_ps": 200, "part": "C", "folder": "traj_ft27k", "prefix": "mace",
     "published": None, "label": "Fine-tuned on 2x2x2 labels, 416 atoms, 200 ps"},
    {"id": "clx000_416_150", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li6PS5Cl",
     "cl_content": 1.0, "n_atoms": 416, "production_ps": 150, "part": "D", "folder": "traj_dope_cl100",
     "prefix": "mace", "published": ("metrics_dope_cl100.json", "mace"), "label": "Li6PS5Cl (x = 0), 150 ps"},
    {"id": "clx025_408_150", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li5.75PS4.75Cl1.25",
     "cl_content": 1.25, "n_atoms": 408, "production_ps": 150, "part": "D", "folder": "traj_dope_cl125",
     "prefix": "mace", "published": ("metrics_dope_cl125.json", "mace"),
     "label": "Li5.75PS4.75Cl1.25 (x = 0.25), 150 ps"},
    {"id": "clx050_400_150", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li5.5PS4.5Cl1.5",
     "cl_content": 1.5, "n_atoms": 400, "production_ps": 150, "part": "D", "folder": "traj_dope_cl150",
     "prefix": "mace", "published": ("metrics_dope_cl150.json", "mace"),
     "label": "Li5.5PS4.5Cl1.5 (x = 0.50), 150 ps"},
    {"id": "clx075_392_150", "potential": "MACE-MP-0", "model": "foundation", "composition": "Li5.25PS4.25Cl1.75",
     "cl_content": 1.75, "n_atoms": 392, "production_ps": 150, "part": "D", "folder": "traj_dope_cl175",
     "prefix": "mace", "published": ("metrics_dope_cl175.json", "mace"),
     "label": "Li5.25PS4.25Cl1.75 (x = 0.75), 150 ps"},
)
SERIES_BY_ID = {series["id"]: series for series in SERIES}
# Production runs (416 atoms, 200 ps) get truncation windows, block estimates and kinisi start sensitivity.
PRODUCTION_SERIES = ("mace_416_200", "mattersim_416_200", "ftG_416_200", "ftK_416_200")
WINDOW_PS = (50, 100, 150, 200)
BLOCK_PS = 50
START_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)
UNIVERSAL_POTENTIALS = ("MACE-MP-0", "MatterSim")


def run_id(series_id: str, temperature: int) -> str:
    return f"{series_id}_{temperature}K"


def runs():
    """(run_id, series, temperature) for all 42 runs, in catalogue order."""
    return [(run_id(series["id"], t), series, t) for series in SERIES for t in TEMPERATURES]
