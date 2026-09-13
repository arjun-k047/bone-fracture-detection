# Training Notes

## Dataset

GRAZPEDWRI-DX (pediatric wrist trauma X-rays), exported through Roboflow
into YOLO format. Single `fracture` class used for this run - the raw
dataset has a few other finding categories but I dropped those to keep
the problem simple and because the fracture class had the most examples.

Split was the standard Roboflow train/val/test split, no custom splitting
done on my end.

## Preprocessing

Grayscale conversion + CLAHE (contrast limited adaptive histogram
equalization) before training and before inference, `clipLimit=2.0`,
`tileGridSize=(8,8)`. X-rays are low contrast to begin with so this made
fracture lines noticeably easier to see, even just visually. Didn't run a
proper grid search on the CLAHE params, just eyeballed a few values.

## Model / training setup

- Base: `yolo11m.pt` (Ultralytics pretrained)
- Image size: 640
- Epochs: 60
- Batch size: 8
- Optimizer: AdamW, lr0 = 0.001
- Ran on Google Colab, T4 GPU
- Augmentation mostly left at Ultralytics defaults except mosaic disabled
  and hsv saturation/hue disabled (didn't seem to help on grayscale X-rays,
  and just added noise)

I also tried a `yolo11s` variant at 1024px as a smaller/faster alternative,
and an earlier `yolo11m` run with different CLAHE settings that diverged
partway through and got scrapped. `yolo11m_fracture_final` (below) ended
up the best of what I trained, so that's the one shipped as `model/best.pt`.

## Results

Final epoch (60/60):

| Metric | Value |
|---|---|
| Precision | 0.724 |
| Recall | 0.688 |
| mAP50 | 0.736 |
| mAP50-95 | 0.480 |

Peaked slightly higher mid-training (mAP50 0.797 around epoch 51) but the
checkpoint Ultralytics actually saves as `best.pt` is picked by a combined
fitness score (mostly mAP50-95), not raw mAP50, so it doesn't exactly match
whichever epoch had the highest mAP50 alone.

Training curves, confusion matrix and PR curve are in `results/`:

- `results/results.png` - loss/precision/recall/mAP curves across all 60 epochs
- `results/confusion_matrix.png` - fracture vs background confusion matrix on val
- `results/BoxPR_curve.png` - precision-recall curve
- `results/sample_predictions.jpg` - a batch of validation predictions vs ground truth

## What I'd do differently / next steps

- Tune the CLAHE parameters properly instead of eyeballing them.
- Try adding back the other finding classes instead of collapsing everything
  to a single `fracture` label - might help the model separate fracture
  patterns from other abnormalities.
- Get more epochs in on the yolo11s/1024px run, it was still improving and
  hadn't converged when I stopped it.
- Run inference on a genuinely fresh X-ray (not from the training/val set)
  to get a better feel for real-world performance - most of what I tested
  with during development came from the val set batches, which isn't a
  fair test of generalization.
