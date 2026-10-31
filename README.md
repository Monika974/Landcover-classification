# Sentinel-2 Land-Cover Classification

Classifies Sentinel-2 satellite imagery into five land-cover classes using a TensorFlow/Keras
deep-learning pipeline, keeps geographic coordinates so predictions open correctly in QGIS, and
serves the results through a Streamlit dashboard with per-class area statistics.

| ID | Class | Examples |
|----|-------|----------|
| 0 | Water | Rivers, lakes, reservoirs |
| 1 | Vegetation | Forests, grassland, shrubland |
| 2 | Agricultural Land | Cultivated fields |
| 3 | Built-up Area | Buildings, urban settlements |
| 4 | Barren Land | Exposed soil, uncultivated land |

## Pipeline

```
Sentinel-2 imagery --> Google Earth Engine (cloud mask, median composite, WorldCover labels)
        --> Preprocessing (Rasterio/NumPy: indices, 64x64 patches, region-based split)
        --> Models: Random Forest baseline | patch CNN | U-Net   (scikit-learn / TensorFlow-Keras)
        --> Evaluation (confusion matrix, accuracy, precision, recall, F1 on unseen regions)
        --> Georeferenced class map (GeoTIFF + GeoPackage via Rasterio, GeoPandas, Shapely)
        --> QGIS visualisation  +  Streamlit dashboard (classified map + area statistics)
```

## Design decisions

- **Input features (7 channels):** Sentinel-2 bands B2, B3, B4, B8, B11 plus NDVI and NDWI. NDVI/NDWI and the SWIR band help separate vegetation, cropland and water.
- **Labels:** [ESA WorldCover](https://esa-worldcover.org/) v200 (10 m, year 2021), remapped to the five classes in `landcover/config.py`. Imagery is taken from the same year. These labels are model-derived, so treat them as *reference labels*, not field-surveyed ground truth, and say so in your report.
- **Spatially independent evaluation:** every study region belongs to exactly one of train / val / test, so test scores are not inflated by neighbouring-pixel leakage.
- **Three models, compared on the same test regions:**
  - Random Forest on per-pixel values (baseline, no spatial context)
  - Patch CNN: one label per 64x64 patch, so its maps are blocky (scored per patch)
  - U-Net: per-pixel labels, giving a real classified map and accurate area percentages (scored per pixel)
- **Class imbalance:** sqrt-dampened inverse-frequency class weights; the U-Net loss ignores nodata pixels (value 255). Report per-class F1, not only overall accuracy.
- **Known difficulty:** Agricultural Land vs Vegetation is the hardest pair to separate from imagery alone; expect most confusion there.

## Setup

```bash
git clone https://github.com/<your-username>/landcover-classification.git
cd landcover-classification
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Or open `notebooks/colab_quickstart.ipynb` in Google Colab (GPU runtime recommended for training).

## Usage

**1. Choose study regions.** Edit `REGIONS` in `landcover/config.py` (bounding boxes and split). The defaults are examples. Use several regions per split, since a model trained on one scene generalises poorly.

**2. Export data from Earth Engine.** You need a Google Cloud project registered for Earth Engine.
```bash
earthengine authenticate
python scripts/gee_export.py --project YOUR_GCP_PROJECT
```
Wait for the tasks to finish (<https://code.earthengine.google.com/tasks>), then download the GeoTIFFs from the Drive folder `landcover` into `data/raw/`, named `<region>_s2.tif` and `<region>_label.tif`.

**3. Build the patch dataset.**
```bash
python -m landcover.preprocess            # writes data/processed/{train,val,test}.npz
```

**4. Train and evaluate.**
```bash
python -m landcover.baseline_rf                      # Random Forest baseline
python -m landcover.train --model cnn  --epochs 30   # patch CNN
python -m landcover.train --model unet --epochs 40   # U-Net
python -m landcover.evaluate --model cnn
python -m landcover.evaluate --model unet
```
Metrics, confusion matrices and training curves are written to `outputs/`.

**5. Classify a scene.**
```bash
python -m landcover.predict --input data/raw/<test_region>_s2.tif --model-type unet --vectorize
```
Produces `outputs/<region>_classified.tif` (with colour table and the source CRS), an area-statistics CSV and, with `--vectorize`, a GeoPackage of class polygons.

**6. Visualise in QGIS.** Layer > Add Raster Layer > the classified GeoTIFF (colours load automatically), and optionally the `.gpkg` for vector analysis. Place it over an OpenStreetMap or satellite basemap to check results on the ground.

**7. Run the dashboard.**
```bash
streamlit run app/streamlit_app.py
```

## Results

Fill this in after your own runs (scores come from `outputs/*_metrics.json`). Do not report numbers you did not produce.

| Model | Granularity | Overall accuracy | Macro F1 |
|-------|-------------|------------------|----------|
| Random Forest | pixel | TBD | TBD |
| Patch CNN | patch | TBD | TBD |
| U-Net | pixel | TBD | TBD |

## Project structure

```
landcover/        config, indices, labels, preprocess, models, train, evaluate, baseline_rf, predict, metrics
scripts/          gee_export.py (Earth Engine -> Drive)
app/              streamlit_app.py
tests/            pytest suite (labels, indices, patching, prediction, models)
notebooks/        colab_quickstart.ipynb
data/ models/ outputs/   git-ignored working folders
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Limitations

- WorldCover labels have their own errors (especially cropland vs grassland), which cap achievable accuracy.
- A single-year median composite ignores crop seasonality; adding a second-season composite is a natural improvement.
- Vectorising very large scenes is slow; run it on cropped areas.

## License

MIT. Copyright and data: Sentinel-2 (Copernicus, ESA) and ESA WorldCover are used under their open licences; credit them in any publication.
