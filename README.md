# Bone Fracture Detection (YOLOv11)

Final year project. Fine-tuned a YOLOv11 model to spot fractures on pediatric
wrist X-rays, using the GRAZPEDWRI-DX dataset, and wrapped it in a small
Flask app so it's actually usable instead of just a notebook.

### About this project

Model training, dataset selection, preprocessing, and evaluation are my own
work. I used Claude Code to scaffold the Flask app (routes, templates) and
for general cleanup along the way.

## Why this dataset

GRAZPEDWRI-DX is a public dataset of ~20k pediatric wrist X-rays with bounding
box annotations for fractures and a few other findings, released by the
Medical University of Graz. Went with it because it's one of the few
X-ray sets with proper object-detection style labels instead of just
image-level classification labels.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`, upload an X-ray image, get back the image with
predicted fracture boxes and confidence scores. Don't have an X-ray handy?
`samples/` has 8 you can use to try it out.

## Project layout

```
app.py              Flask app - upload an image, get predictions back
train.py             training script (Colab), kept for reference
model/best.pt        trained weights used by app.py
dataset/             raw GRAZPEDWRI-DX export (gitignored, see below)
samples/             8 sample X-rays for trying the app
results/             confusion matrix, PR curve, training curves, a sample prediction
templates/, static/  frontend for the Flask app
TRAINING_NOTES.md    dataset/training details and results
```

## Results (short version)

Best run: YOLOv11m, 640px, 60 epochs -> **peak mAP50 0.797**. Full numbers,
plots and the training/preprocessing writeup are in
[TRAINING_NOTES.md](TRAINING_NOTES.md).

## Dataset

Trained on [GRAZPEDWRI-DX](https://doi.org/10.1038/s41597-022-01328-z)
(Nagy et al., 2022, *Scientific Data*), a pediatric wrist trauma X-ray
dataset, licensed CC BY 4.0. The raw export isn't committed here (see
`.gitignore`) - grab it from
[Roboflow](https://universe.roboflow.com/ml-u7780/grazpedwri-dx-imszg/dataset/1)
if you need it for retraining. The 8 images in `samples/` were selected and
renamed from this dataset for demo purposes, same license applies.

## Known limitations

- CLAHE preprocessing params in `app.py` (`clipLimit=2.0`, `tileGridSize=8x8`)
  were picked by eye rather than tuned, so results may vary a bit depending
  on input image quality/contrast.
- Only tested against wrist X-rays similar to the training set - not meant
  for any other bone or actual clinical use, this is a student project, not
  a diagnostic tool.
