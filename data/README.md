# Data layout

Do **not** commit downloaded datasets to this repo.

The experiment reads a CSV manifest with at least:

```csv
path,label
/absolute/or/relative/image.jpg,1
/absolute/or/relative/image2.jpg,0
```

Optional columns:

- `split`: `train`, `val`, or `test`
- `source`: dataset/source name
- any metadata you want to preserve

For a target-presence experiment:

- `label=1`: target object/condition is present
- `label=0`: target is absent

## Folder dataset

```text
data/raw/my_test/
├── positive/
│   ├── 0001.jpg
│   └── ...
└── negative/
    ├── 0001.jpg
    └── ...
```

Build a manifest:

```bash
python scripts/prepare_folder_dataset.py \
  --positive-dir data/raw/my_test/positive \
  --negative-dir data/raw/my_test/negative \
  --output data/my_test.csv
```

## Recommended first public dataset: ExDark

ExDark is especially useful for the first SCLI-Vision test because it contains real low-light images across multiple illumination conditions and includes a `Cat` class.

Download the images and ground truth separately from the official project, then create a cat-vs-rest manifest:

```bash
python scripts/prepare_exdark.py \
  --image-root /workspaces/data/ExDark/Dataset \
  --image-class-list /workspaces/data/ExDark/Groundtruth/imageclasslist.txt \
  --target-class Cat \
  --output data/exdark_cat.csv
```

Then run:

```bash
python scripts/run_experiment.py \
  --manifest data/exdark_cat.csv \
  --output-dir outputs/exdark_cat_v1
```
