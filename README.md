# ExDark object-centered patch

This patch changes the ExDark test from whole-image classification to object-centered
classification using the official ExDark bounding boxes.

Why:
- Whole ExDark images contain large, highly variable backgrounds.
- SCLI role discovery averages positive images.
- On full Cat images, role positions can be dominated by background/scene structure.
- ExDark provides local bounding boxes, so object crops are the correct first test
  of cross-instance relational identity.

## 1. Generate real object crops

```bash
python scripts/prepare_exdark_crops.py \
  --root data/external/ExDark/ExDark \
  --target-class Cat \
  --output-root data/exdark_object_crops \
  --manifest data/exdark_cat_crops.csv
```

## 2. Inspect counts

```bash
python - <<'PY'
import pandas as pd
df=pd.read_csv("data/exdark_cat_crops.csv")
print(df.groupby(["split","label"]).size())
print(df.class_name.value_counts())
PY
```

## 3. Run SCLI

Use the memory-safe ExDark runner already in the repository:

```bash
export PYTHONPATH="$PWD/src"

python scripts/run_experiment.py \
  --manifest data/exdark_cat_crops.csv \
  --config configs/exdark_crop.yaml \
  --output-dir outputs/exdark_cat_crops_v1 \
  --working-size 96 \
  --negative-ratio 3 \
  --max-val-per-class 500 \
  --batch-size 64
```

If the model still reports no stable constraints, do NOT blindly keep lowering the
threshold. First print the stability percentiles using the diagnostic snippet in
`MODEL_DIAGNOSTIC.txt`. If the maximum stability is still close to 0.5, the current
role representation is inadequate for cross-instance cat identity; that is a
representation failure, not a parameter problem.

Scientific note:
This crop benchmark tests "Cat identity vs other annotated object identity".
It is closer to object recognition than the whole-frame run, but it is not yet the
full Robotaxi "object present vs empty road" safety experiment. That comes next by
constructing positive object boxes and negative background/free-space patches.
