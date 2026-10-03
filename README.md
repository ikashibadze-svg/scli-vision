# SCLI-Vision: Observable Constraint Recognition

Research prototype for testing the principle:

> preserve/create identity-relevant distinctions; return `UNDERDETERMINED` when those distinctions are not observable.

This repository is intentionally small and inspectable. It does **not** claim production autonomous-driving readiness.

## Codespaces quick start

If this is an existing repo, update it first:

```bash
git pull
```

For the ZIP provided with this project, upload/unzip it into a GitHub repository and open **Code → Codespaces → Create codespace**. The devcontainer installs the Python requirements automatically.

Manual setup:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Fast smoke test

This uses real face/non-face images bundled with scikit-image, so no dataset download is required:

```bash
export PYTHONPATH="$PWD/src"
python scripts/smoke_test.py
python scripts/run_experiment.py \
  --manifest data/smoke/manifest.csv \
  --output-dir outputs/smoke
```

You should get:

```text
outputs/smoke/
├── accuracy.png
├── safety_metrics.png
├── results.csv
├── details.csv
├── false_clear_cases.csv
├── resolved_manifest.csv
└── run.json
```

## First real low-light experiment: ExDark Cat vs Rest

Download ExDark outside the repo. See `PHOTO_SOURCES.md` and `data/README.md`.

Create the manifest:

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_exdark.py \
  --image-root /workspaces/data/ExDark/Dataset \
  --image-class-list /workspaces/data/ExDark/Groundtruth/imageclasslist.txt \
  --target-class Cat \
  --output data/exdark_cat.csv
```

Run:

```bash
python scripts/run_experiment.py \
  --manifest data/exdark_cat.csv \
  --config configs/default.yaml \
  --output-dir outputs/exdark_cat_v1
```

## What is being compared?

### Baseline

A conventional HOG + logistic-regression recognizer.

### SCLI observable-constraint identity

The current prototype:

```text
image
  -> canonical structural frame
  -> persistent latent roles (learned from positive examples)
  -> pairwise relational constraints
  -> raw-unit observability gate
  -> identity score
  -> KNOWN positive / KNOWN negative / UNDERDETERMINED
```

A constraint is only allowed to vote if its measured magnitude is larger than an estimated raw measurement-noise threshold. This prevents normalization from turning nearly pure noise into a fake invariant.

## Metrics that matter

`results.csv` reports:

- `hog_accuracy`
- `scli_raw_accuracy`
- `known_coverage`
- `known_accuracy`
- `underdetermined_rate`
- `false_clear_rate`
- `false_alarm_rate`
- `hog_false_clear_rate`

For the Musk-style safety framing, the most important number is:

```text
false_clear_rate
```

A positive image that becomes `UNDERDETERMINED` is **not** counted as false-clear. The point of the epistemic gate is to prefer refusal/extra probing over declaring a weakly observed scene clear.

## Recommended experiment order

1. `smoke`: verify repo and plots.
2. ExDark `Cat` vs rest: real low-light recognition.
3. Your own matched grey-on-grey positive/negative set.
4. BDD100K night images: road-driving domain.
5. Video/temporal extension: flow/parallax as an added probe.
6. Optional LLVIP visible-vs-infrared comparison: test measurement-mechanism substitution.

## Important limitations

- This is a research prototype, not a certified safety system.
- A single photo cannot recover information that the sensor did not record.
- The SCLI gate should return `UNDERDETERMINED` when identity-relevant distinctions are not observable.
- For a real driving system, the next stage needs temporal frames, ego-motion and safety-action logic.
