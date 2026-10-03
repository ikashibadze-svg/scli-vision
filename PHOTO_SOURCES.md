# Where to get real photos

## 1. ExDark — recommended first

Official project:
https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

Why first:
- 7,363 real low-light images
- 12 object classes
- Cat class included
- image-level labels and object bounding boxes
- 10 lighting conditions

Download workflow:
1. Open the official GitHub project.
2. Open `Dataset/README.md` and use the dataset download link.
3. Open `Groundtruth/README.md` and download the annotations.
4. Put them outside the repo, for example:

```text
/workspaces/data/ExDark/
├── Dataset/
│   ├── Cat/
│   ├── Dog/
│   ├── Car/
│   └── ...
└── Groundtruth/
    └── imageclasslist.txt
```

License note: the ExDark authors state that dataset use is for non-commercial research; contact them for commercial use.

## 2. BDD100K — later, for the actual road/night extension

Official toolkit/project:
https://github.com/bdd100k/bdd100k

Use the 100K image set + detection labels. BDD100K annotations include a `timeofday` field with `night`, so you can filter night scenes.

This should be the second phase after the still-photo proof works.

## 3. NightOwls — later, night pedestrian stress test

Official download page:
https://www.nightowls-dataset.org/download/

Very large nighttime pedestrian dataset. It is useful for testing whether the epistemic gate avoids false-clear decisions under hard night visibility.

License note: NightOwls is distributed for non-commercial purposes under its dataset terms.

## 4. LLVIP — later, visible vs infrared comparison

Project:
https://github.com/CyberPegasus/LLVIPDataset

LLVIP contains paired visible/infrared low-light data. It is useful later when testing the CIT hypothesis that changing the measurement mechanism/probe breaks observational equivalence.

## 5. Your own focused “grey-on-grey” set — strongly recommended

Public datasets are useful, but Musk's exact edge case is narrow. Create a small controlled set yourself:

```text
data/raw/grey_test/
├── positive/
│   ├── grey_cat_on_grey_surface_001.jpg
│   ├── small_dark_object_002.jpg
│   └── ...
└── negative/
    ├── same_surface_empty_001.jpg
    ├── same_surface_empty_002.jpg
    └── ...
```

For every positive photo, try to shoot a matched negative photo of the same background, camera position, exposure and lighting with the object removed. This is much more informative than collecting unrelated negatives.

Recommended progression:
- 50 matched positive/negative pairs for a first test
- 200+ pairs for a meaningful benchmark
- vary distance, object size, lighting, ISO/noise, exposure, background tone and partial occlusion
- keep an untouched held-out test set from different sessions/locations
