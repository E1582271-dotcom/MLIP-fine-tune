# Amendments

Dated changes to `reproduction_rule.md` and `analysis_plan.md` made after they were committed,
with the reason and whether results had been seen.

## A1 (2026-10-26): training-set errors in the MACE logs

**Rule affected:** `reproduction_rule.md`, item 8.

**What was found.** Item 8 assumed that the training-set errors printed in the MACE logs cover all
training configurations. mace-torch 0.3.16 evaluates the training set with its training data
loader, which shuffles and drops the last incomplete batch (`drop_last=True`). With a batch size
of 4:

- the Gamma fine-tune has 22 training configurations, so each logged training error covers a
  random 20 of them;
- the 2x2x2 fine-tune has 24, so all of them are used.

The validation loader drops nothing.

**Change.**

- Validation errors must equal the logged values, as before.
- Training errors must equal the logged values when the training set is a multiple of the batch
  size.
- Otherwise the logged value must be produced by at least one subset of the evaluated size
  (here 20 of 22). Model files are assigned to a training stage by their validation error.

**Results seen before the change.** Yes:

- Gamma fine-tune, stage one: the re-evaluated training error over all 22 configurations is
  72.18 meV/A, against 72.4 logged. 11 of the 231 subsets of 20 configurations give 72.4.
- Gamma fine-tune, stage two: 47.93 over all 22, against 47.9 logged. 21 of the 231 subsets give
  47.9.
- All validation errors and both 2x2x2 training errors matched as originally stated.
