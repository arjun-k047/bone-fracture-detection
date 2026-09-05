# REPORT

## 1. What existed

Recovered from the 3 zips (extracted into `runs/`, now under
`_archive_not_for_repo/runs/`):

| Run folder | Model | imgsz | Data path (from args.yaml) | Epochs configured | results.csv? | weights present |
|---|---|---|---|---|---|---|
| `fracture_project/yolo11m_clahe_run` | yolo11m.pt | 640 | `.../grazpedwri-dx-1_enhanced/data.yaml` | 100 | yes (47 rows logged) | `last.pt` only |
| `fracture_project/yolo11m_fracture_final` | yolo11m.pt | 640 | `.../dataset_fracture_final/data.yaml` | 60 | yes (60 rows) | `best.pt` + `last.pt` |
| `fracture_project/yolo11m_fracture_only` | yolo11m.pt | 640 | `.../grazpedwri-dx-1_enhanced/data.yaml` | 80 | **no** | **none** (empty `weights/`) |
| `run_small_10242` | yolo11s.pt | 1024 | `.../Bone-fracture-detection-1/data.yaml` | 100 | yes (100 rows) | `best.pt` + `last.pt` |
| `run_small_1024` | yolo11s.pt | 1024 | `./bone-fracture-detection-1/data.yaml` | 100 | **no** | **none** (empty `weights/`) |

No `dataset.yaml`, notebook, README, or preprocessing script exists anywhere
in any of the 3 zips — only `args.yaml` per run (Ultralytics' own recorded
CLI args) plus the standard plot/image artifacts.

## 2. Best-model determination (metrics-only)

All five `args.yaml` files show `resume: false` and a generic pretrained base
(`yolo11m.pt` / `yolo11s.pt`) as `model:` — **none of them record resuming
from another local run's checkpoint.** So the "booster pass = a resume of an
earlier run" theory is not confirmed by any metadata that survived; if it
happened, Ultralytics didn't log it, or that record didn't make it into these
zips.

Peak logged metrics per run (from `results.csv`, only 3 of 5 runs have one):

| Run | Peak mAP50 | Epoch of peak | mAP50-95 at that epoch | Checkpoint size |
|---|---|---|---|---|
| **yolo11m_fracture_final** | **0.797** | 51 / 60 | 0.459 | best.pt 121 MB |
| run_small_10242 | 0.763 | 100 / 100 (still rising, not converged) | 0.341 | best.pt 19 MB |
| yolo11m_clahe_run | 0.370 | 43 / 100 (run stopped early, diverging) | 0.198 | last.pt only, 121 MB |
| yolo11m_fracture_only | n/a — no results.csv | — | — | n/a |
| run_small_1024 | n/a — no results.csv | — | — | n/a |

`yolo11m_fracture_final`'s own `best.pt` is saved on Ultralytics' fitness
score (`0.1·mAP50 + 0.9·mAP50-95`), not raw mAP50, so the checkpoint on disk
most likely corresponds to epoch ~59 (mAP50 0.739, mAP50-95 0.487 — the
highest-fitness epoch in that log), not the epoch-51 mAP50 peak of 0.797.
Either way it's the strongest logged run.

**None of the surviving runs reach 0.86 mAP50.** The closest is
`yolo11m_fracture_final` at 0.797 peak (0.739 at its actual best.pt epoch) —
still 6-12 points short of the 0.86 you remember. The two runs with no
metrics or weights at all (`yolo11m_fracture_only`, `run_small_1024`) are the
most likely candidates for where a higher-scoring "booster" result would
have lived — their `weights/` folders are present but **empty**, and they
have no `results.csv`, meaning the actual checkpoint/log from those runs
did not survive into these 3 zips. I can't recover metrics that aren't on
disk.

**Picked: `yolo11m_fracture_final/weights/best.pt`** — copied to
`model/best.pt`. It's the best-performing checkpoint that actually exists in
the recovered data, by a clear margin over the other complete run. It is
very likely *not* the 0.86 booster model you remember; treat 0.86 as
unverified until/unless a surviving copy of `yolo11m_fracture_only` or
`run_small_1024`'s weights turns up elsewhere.

## 3. Preprocessing script: **confirmed lost**

Searched every file in all 3 zips for CLAHE/preprocessing scripts, notebooks,
`dataset.yaml`, or README-like notes. Found nothing except the string
"clahe" in the `yolo11m_clahe_run` folder name and its `args.yaml` save path
— no actual code. `app.py` implements a standard OpenCV CLAHE
(`clipLimit=2.0`, `tileGridSize=(8,8)`) + grayscale pipeline as a documented
guess, not a recovered original.

## 4. What moved to `_archive_not_for_repo/`

Everything under `runs/` (all 5 run folders — checkpoints, `results.csv`,
`args.yaml`, training-batch/val images, confusion matrices, curves). Nothing
was deleted, only moved, so the losing runs and the raw logs are still on
disk if you need to re-check the mAP50 comparison later.

## 5. Dependency fix (torchvision mismatch)

`pip install -r requirements.txt` originally failed two ways on this
machine, both now fixed in `requirements.txt`:

- Unpinned `torch` pulled the full CUDA build (multi-GB `nvidia-*` wheels)
  and blew past a 3GB `/tmp` quota mid-download. Fixed by installing from
  the CPU wheel index.
- `ultralytics` then pulled a plain-PyPI `torchvision` built against a
  different torch ABI, giving `RuntimeError: operator torchvision::nms does
  not exist` at inference time. Fixed by pinning
  `torchvision==0.29.0+cpu` and adding `--extra-index-url
  https://download.pytorch.org/whl/cpu` to `requirements.txt` so the pinned
  `+cpu` build actually resolves.

Re-verified with a fresh throwaway venv (not just the already-satisfied
system env): `pip install -r requirements.txt` resolves `torch-2.14.0+cpu`
+ `torchvision-0.29.0+cpu` together, no CUDA wheels, both import cleanly.

## 6. Runtime test results

`app.py` was actually run (not just syntax-checked) — `python app.py`
starts fine, model loads, server responds 200 on `/`.

No clean/unannotated single X-ray survived anywhere in the 3 zips (see §1,
§3) — every surviving image is a training-visualization mosaic with
ground-truth boxes and filenames burned into the pixels. Two tests were run
with what's actually available:

- **Full training mosaic** (`yolo11m_fracture_final/train_batch0.jpg`, a
  3x3 tiled grid) uploaded as-is: pipeline ran end to end, 3 `fracture`
  detections at conf 0.67 / 0.39 / 0.28, all correctly landing on tiles
  showing surgical wire/plate fixation hardware.
- **Single cropped tile** (one 640x640 cell cropped out of that same
  mosaic) uploaded as a stand-in for a real single-image upload: 1
  `fracture` detection at **conf 0.86**, box tightly matching the
  pre-existing ground-truth box. (That 0.86 is this one detection's
  confidence score — coincidentally the same number as the "booster" mAP50
  you remembered, but it's an unrelated metric, not evidence the booster
  run is this checkpoint.)

Both tests used post-op images with implanted fixation hardware, not fresh
untreated fractures — this confirms the app *works*, not that detection
quality is representative of real clinical use. CPU inference ran ~410ms/
image (no GPU in this environment).

## 7. Manual steps before this goes public

- **0.86 mAP50 is not verified.** The promoted `model/best.pt` peaks at
  ~0.74-0.80 mAP50 in its own logs, not 0.86. If you find a surviving copy
  of the `yolo11m_fracture_only` or `run_small_1024` weights/logs elsewhere,
  re-run this comparison before trusting this as your final model.
- **CLAHE params in `app.py` are a guess.** Visually compare its output
  against a sample training image if one survives (e.g. `train_batch0.jpg`
  in the archived run folders) to sanity-check contrast/brightness look
  similar.
- No live `yolo val` was run (no dataset available) — all numbers above are
  read from logs only, not independently re-verified.
- **Test with a real, clean, single upload of your own** — everything
  tested here came from cropped/whole training-visualization mosaics with
  hardware already implanted, not a fresh fracture X-ray a real user would
  upload.
- You still need to `git init` / add / commit yourself; nothing here touched
  git.
