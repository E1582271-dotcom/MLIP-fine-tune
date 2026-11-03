"""Training records of the two fine-tunes (Extended Data, registry key "training").

a, validation force RMSE and b, validation energy RMSE against epoch, as logged by MACE, for the
fine-tune on Gamma labels and the fine-tune on 2x2x2 labels, each on its own validation set and
label set; c, the logged errors of the selected models against the re-evaluated ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import paths
from . import common, registry, style

KEY = "training"
NAME = registry.number(KEY)
PANELS = "abc"
WIDTH, HEIGHT = 183.0, 115.0
RECORDS = {"ftG": paths.MODELS / "training" / "ftG_record.jsonl", "ftK": paths.MODELS / "training" / "ftK_record.jsonl"}
EVALUATION = paths.MODELS / "evaluation.json"
REPRODUCTION = paths.MODELS / "log_reproduction.csv"
CONFIGS = paths.DFT / "configs.csv"
CROSS = paths.RESULTS / "cross_evaluation.csv"  # drove_md: which selected model drove which MD run
LABELS = {"ftG": "gamma", "ftK": "k222"}
MD_RUNS = {"ftG": "the Gamma fine-tune run", "ftK": "the July 2x2x2 fine-tune run",
           "ftK2": "the 2x2x2 fine-tune run of protocol amendment A5"}  # fine-tuned run families (common.RUN_FAMILIES)
START_X = -9.0  # where the error before training (epoch null in the record) is drawn
START_SIZE = 3.2  # marker of MACE-MP-0 before fine-tuning
SELECTED_SIZE = 3.6  # markers of the selected models


def _eval_lines(run: str) -> pd.DataFrame:
    """The per-epoch validation lines of a MACE training record (epoch NaN: before training)."""
    frame = pd.read_json(RECORDS[run], lines=True, dtype=False)
    frame = frame[frame["mode"] == "eval"].copy()
    frame["force_rmse_meV_A"] = frame["rmse_f"] * 1000.0
    frame["energy_rmse_meV_atom"] = frame["rmse_e_per_atom"] * 1000.0
    return frame[["epoch", "force_rmse_meV_A", "energy_rmse_meV_atom", "loss"]].reset_index(drop=True)


def _load():
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))["training"]
    records = {run: _eval_lines(run) for run in RECORDS}
    reproduction = common.read(REPRODUCTION)
    configs = common.read(CONFIGS)
    return evaluation, records, reproduction, configs


def _selected(evaluation: dict, run: str) -> dict:
    tables = evaluation[run]["error_tables"]
    return {stage: int(tables[stage]["epoch"]) for stage in ("s1", "s2")}


def _valid_text(run: str, evaluation: dict) -> str:
    return (f"{evaluation[run]['n_valid']} configurations, {common.LABEL_SET_NAMES[LABELS[run]]} labels")


def _curves(ax, evaluation: dict, records: dict, column: str, start: bool) -> list[dict]:
    rows = []
    stage_two = min(evaluation[run]["stage_two_start_epoch"] for run in RECORDS)
    last = max(records[run]["epoch"].max() for run in RECORDS)
    ax.axvspan(stage_two - 0.5, last + 0.5, **style.BAND_FILL)
    for run, record in records.items():
        color = style.MODEL_COLORS[run]
        trained = record[record["epoch"].notna()]
        ax.plot(trained["epoch"], trained[column], color=color, lw=0.8, zorder=2)
        rows.extend({"fine_tune": run, "epoch": int(e), column: v} for e, v in zip(trained["epoch"], trained[column]))
        before = record[record["epoch"].isna()]
        if start and len(before) == 1:
            value = float(before[column].iloc[0])
            ax.plot([START_X, trained["epoch"].iloc[0]], [value, trained[column].iloc[0]], color=color, lw=0.6,
                    ls=(0, (1.5, 1.5)), zorder=2)
            ax.plot(START_X, value, **common.model_style("foundation", size=START_SIZE, zorder=4))
            rows.append({"fine_tune": run, "epoch": "before training (MACE-MP-0)", column: value})
        for stage, epoch in _selected(evaluation, run).items():
            value = float(trained.loc[trained["epoch"] == epoch, column].iloc[0])
            ax.plot(epoch, value, **common.model_style(f"{run}_{stage}", size=SELECTED_SIZE, edge=0.8, zorder=5))
            rows.append({"fine_tune": run, "model": f"{run}_{stage}", "epoch": f"selected {stage} ({epoch})",
                         column: value})
    ax.text(stage_two + 1.0, 0.98, "stage two", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=style.SMALL, color=style.CONTEXT_TEXT)
    ax.set_xlim(START_X - 4.0, last + 4.0)  # room for the marker of the last selected epoch
    ticks = [0, 20, 40, 60, 80, 100, 120]
    ax.set_xticks([START_X, *ticks], ["start", *[str(t) for t in ticks]] if start else ["", *[str(t) for t in ticks]])
    if not start:
        ax.tick_params(axis="x", which="major")
        ax.get_xticklines()[0].set_visible(False)
    ax.set_xlabel("Epoch")
    return rows


def _panel_a(ax, evaluation: dict, records: dict) -> None:
    rows = _curves(ax, evaluation, records, "force_rmse_meV_A", start=True)
    ax.set_ylim(40, 250)  # the two stage-one selections at epoch 89 (78.9 and 70.3) stay apart
    ax.set_ylabel("Validation force RMSE (meV/Å)")
    style.save_source(pd.DataFrame(rows), NAME, "a")


def _panel_b(ax, evaluation: dict, records: dict) -> None:
    rows = _curves(ax, evaluation, records, "energy_rmse_meV_atom", start=False)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 300)
    style.log_ticks(ax, "y")
    ax.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax.set_ylabel("Validation energy RMSE (meV/atom)")
    style.save_source(pd.DataFrame(rows), NAME, "b")


def _legend(fig, layout: common.Layout, evaluation: dict, top: float) -> None:
    """Three columns: the two training runs and the stage-two band; MACE-MP-0 and the Gamma selections; the
    2x2x2 selections."""
    handles = [Line2D([0], [0], color=style.MODEL_COLORS[run], lw=0.9,
                      label=f"{'Gamma' if run == 'ftG' else '2x2x2'} fine-tune (validation: {_valid_text(run, evaluation)})")
               for run in RECORDS]
    handles.append(Patch(facecolor=style.BAND, edgecolor="none", label="stage two of training"))
    handles.append(common.handle(common.model_style("foundation", size=START_SIZE),
                                 "MACE-MP-0 before fine-tuning (forces only)"))
    handles += [common.handle(common.model_style(model, size=SELECTED_SIZE, edge=0.8),
                              f"selected: {common.SHORT_LABELS[model]}")
                for model in ("ftG_s1", "ftG_s2", "ftK_s1", "ftK_s2")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(layout.x(14.0), layout.y(top)), ncol=3,
               fontsize=style.SMALL, handletextpad=0.4, columnspacing=2.4, labelspacing=0.5, borderaxespad=0.0,
               frameon=False)


def _panel_c(ax, evaluation: dict, reproduction: pd.DataFrame, configs: pd.DataFrame) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    columns = [(0.00, "left", "Model"), (0.17, "center", "Epoch"), (0.29, "center", "Validation set"),
               (0.45, "center", "Logged"), (0.56, "center", "Re-evaluated"), (0.70, "center", "Logged"),
               (0.81, "center", "Re-evaluated"), (0.94, "center", "Configurations\nin the logged value")]
    n_rows = len(reproduction)
    ax.set_ylim(n_rows + 2.6, -0.6)
    ax.text((0.45 + 0.56) / 2, -0.35, "Validation force RMSE (meV/Å)", ha="center", va="bottom",
            fontsize=style.SMALL)
    ax.text((0.70 + 0.81) / 2, -0.35, "Training force RMSE (meV/Å)", ha="center", va="bottom", fontsize=style.SMALL)
    for x0, x1 in ((0.40, 0.61), (0.65, 0.86)):
        ax.plot([x0, x1], [-0.25, -0.25], color=style.BLACK, lw=0.4)
    for x, ha, text in columns:
        ax.text(x, 0.55, text, ha=ha, va="center", fontsize=style.SMALL, fontweight="bold", linespacing=1.1)
    ax.plot([0, 1], [1.25, 1.25], color=style.BLACK, lw=0.5)
    valid_names = {"ftG": f"V5, {common.LABEL_SET_NAMES['gamma']}", "ftK": f"H3, {common.LABEL_SET_NAMES['k222']}"}
    n_train = {run: int((configs[f"{run}_split"] == "train").sum()) for run in ("ftG", "ftK")}
    rows = []
    for index, row in enumerate(reproduction.itertuples()):
        y = index + 1.9
        model = row.model
        run = model.split("_")[0] if model != "foundation" else None
        label = (common.SHORT_LABELS[model] if run else f"MACE-MP-0 ({common.LABEL_SET_NAMES[row.labels]})")
        if run:
            valid = valid_names[run]
        else:
            valid = "V5, " + common.LABEL_SET_NAMES[row.labels]
        epoch = "before training" if pd.isna(row.epoch) else str(int(row.epoch))
        cells = [label, epoch, valid, f"{row.logged_valid_rmse_f_meV_A:g}",
                 f"{row.reevaluated_valid_rmse_f_meV_A:.2f}",
                 "-" if pd.isna(row.logged_train_rmse_f_meV_A) else f"{row.logged_train_rmse_f_meV_A:g}",
                 "-" if pd.isna(row.reevaluated_train_rmse_f_meV_A) else f"{row.reevaluated_train_rmse_f_meV_A:.2f}",
                 "-" if pd.isna(row.train_evaluated_by_log) else
                 f"{int(row.train_evaluated_by_log)} of {n_train[run]}"
                 + ("*" if int(row.train_evaluated_by_log) < n_train[run] else "")]
        ax.plot(-0.012, y, **common.model_style(model, size=3.0, edge=0.6, clip_on=False))
        for (x, ha, _), text in zip(columns, cells):
            ax.text(x + (0.008 if ha == "left" else 0.0), y, text, ha=ha, va="center", fontsize=style.SMALL)
        rows.append(dict(zip(["model", "epoch", "validation_set", "logged_valid_meV_A", "reevaluated_valid_meV_A",
                              "logged_train_meV_A", "reevaluated_train_meV_A", "train_configs_in_log"], cells)))
    ax.plot([0, 1], [n_rows + 1.5, n_rows + 1.5], color=style.BLACK, lw=0.5)
    ax.text(1.0, n_rows + 1.75, "* MACE drops the last incomplete training batch (batch size "
            f"{evaluation['ftG']['batch_size']}); protocol amendment A1", ha="right", va="top", fontsize=style.SMALL)
    style.save_source(pd.DataFrame(rows), NAME, "c")


def make() -> list[Path]:
    evaluation, records, reproduction, configs = _load()
    fig, L = common.new_figure(WIDTH, HEIGHT)
    _legend(fig, L, evaluation, 3.0)
    ax_a = L.axes(14.0, 19.0, 84.0, 46.0, row="curves", col="left")  # the hero panel
    _panel_a(ax_a, evaluation, records)
    ax_b = L.axes(116.0, 19.0, 65.0, 46.0, row="curves")
    _panel_b(ax_b, evaluation, records)
    ax_c = L.axes(12.0, 81.0, 169.0, 32.0, frame="none")
    _panel_c(ax_c, evaluation, reproduction, configs)
    L.letter(ax_a, "a", 3.0, 18.0)
    L.letter(ax_b, "b", 105.0, 18.0)
    L.letter(ax_c, "c", 3.0, 80.0)
    return L.save(*registry.target(KEY))


def _caption() -> str:
    evaluation, records, reproduction, configs = _load()
    g, k = evaluation["ftG"], evaluation["ftK"]
    sel = {run: _selected(evaluation, run) for run in RECORDS}
    stage_two = {run: evaluation[run]["stage_two_start_epoch"] for run in RECORDS}
    tables = {run: evaluation[run]["error_tables"] for run in RECORDS}
    rep = reproduction.set_index("model")
    partial = rep.loc["ftG_s1"]
    first, total = (int(v) for v in str(partial["train_subsets_matching"]).split("/"))
    second, total2 = (int(v) for v in str(rep.loc["ftG_s2", "train_subsets_matching"]).split("/"))
    if total != total2:
        raise ValueError("different numbers of training subsets")
    matching = (first, second, total)
    spike = {run: records[run].loc[records[run]["epoch"].between(stage_two[run], stage_two[run] + 5),
                                   "force_rmse_meV_A"].max() for run in RECORDS}
    for key in ("stage_two_start_epoch", "batch_size", "learning_rate", "stage_one_loss_weights",
                "stage_two_loss_weights", "stage_two_learning_rate"):
        if g[key] != k[key]:
            raise ValueError(f"the two fine-tunes differ in {key}")
    max_valid_diff = float(np.max(np.abs(reproduction["logged_valid_rmse_f_meV_A"]
                                         - reproduction["reevaluated_valid_rmse_f_meV_A"])))
    md = common.md_models(common.read(CROSS))  # run family -> the model that drove its MD run
    no_md = [f"{run}_{stage}" for run in RECORDS for stage in sel[run] if f"{run}_{stage}" not in md.values()]
    return (
        registry.heading(KEY) + "Both start from MACE-MP-0 small with the "
        f"same settings (batch size {g['batch_size']}; stage one with energy and force loss weights "
        f"{g['stage_one_loss_weights'][0]:g} and {g['stage_one_loss_weights'][1]:g} and learning rate "
        f"{g['learning_rate']:g}; stage two, from epoch {stage_two['ftG']} (shaded), with weights "
        f"{g['stage_two_loss_weights'][0]:g} and {g['stage_two_loss_weights'][1]:g}, learning rate "
        f"{g['stage_two_learning_rate']:g} and stochastic weight averaging). "
        f"**a**, Validation force RMSE and **b**, validation energy RMSE per atom against epoch, as written by "
        f"MACE: one training run per fine-tune, each point a single evaluation of the whole validation set "
        f"(single values, no error bars). The two curves use different validation sets and different labels and are not "
        f"directly comparable: the Gamma fine-tune is validated on n = {g['n_valid']} configurations (V5 = H3 + H2) "
        f"against the Gamma labels, the 2x2x2 fine-tune on n = {k['n_valid']} (H3) against the 2x2x2 labels. Red "
        f"circles at 'start', MACE-MP-0 small before fine-tuning on the same validation sets "
        f"({tables['ftG']['foundation_valid_rmse_f_meV_A']:.1f} and "
        f"{tables['ftK']['foundation_valid_rmse_f_meV_A']:.1f} meV/A; dotted, joined to the first epoch). The "
        f"energy error before training is not drawn because it is an absolute energy error of the foundation model "
        f"(it depends on the reference energies), which this study does not report. Large circles, the selected "
        f"models, coloured by label set (navy, Gamma; blue, 2x2x2) as in the table (c); open, stage one (epoch {sel['ftG']['s1']} for both; "
        f"{tables['ftG']['s1']['valid_rmse_f_meV_A']:g} and {tables['ftK']['s1']['valid_rmse_f_meV_A']:g} meV/A); "
        f"filled, stage two (epochs {sel['ftG']['s2']} and {sel['ftK']['s2']}; "
        f"{tables['ftG']['s2']['valid_rmse_f_meV_A']:g} and {tables['ftK']['s2']['valid_rmse_f_meV_A']:g} meV/A). "
        f"The switch to the stage-two weights and learning rate raises the force error transiently (to "
        f"{spike['ftG']:.0f} and {spike['ftK']:.0f} meV/A). Logarithmic energy axis in b. "
        f"**c**, Errors in the training logs against the same errors re-evaluated from the model files in float64 "
        f"on a CPU, as root-mean-square errors over the n configurations of each set (single values, no error "
        f"bars; validation n = "
        f"{g['n_valid']} for V5 and {k['n_valid']} for H3; training n = {g['n_train']} and {k['n_train']}). "
        f"Validation errors agree to within {max_valid_diff:.2f} meV/A (the logs print one decimal). "
        f"The logged training errors of the Gamma fine-tune cover {int(partial['train_evaluated_by_log'])} of its "
        f"{g['n_train']} training configurations, because MACE drops the last incomplete batch of the shuffled "
        f"training loader (asterisks; protocol amendment A1): {matching[0]} and {matching[1]} of the {matching[2]} "
        f"subsets of {int(partial['train_evaluated_by_log'])} configurations reproduce the logged stage-one and "
        f"stage-two values. The 2x2x2 fine-tune has {k['n_train']} training "
        f"configurations, a multiple of the batch size, and its logged training errors cover all of them. "
        + common.naming_sentence(configs) + " Selected models that drove MD runs (identified by force "
        f"fingerprints, {registry.label('models', 'a')}): "
        + "; ".join(f"{common.SHORT_LABELS[md[family]]} ({MD_RUNS[family]})" for family in MD_RUNS)
        + f". {' and '.join(common.SHORT_LABELS[m] for m in no_md)} drove no MD run. The force errors of both "
        f"stages are in {registry.label('models', 'b')}, and the effect of the stage on Li diffusion at equal "
        f"labels in {registry.label('finetune', 'c')}. "
        + common.sources([RECORDS["ftG"], RECORDS["ftK"], EVALUATION, REPRODUCTION, CONFIGS, CROSS])
    )


CAPTION = _caption()
