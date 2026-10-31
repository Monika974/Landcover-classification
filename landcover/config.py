"""Central configuration. Edit REGIONS / DATE_RANGE to change the study area."""
from pathlib import Path  # noqa

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

SEED = 42

# ---- Sentinel-2 inputs -------------------------------------------------------
BANDS = ["B2", "B3", "B4", "B8", "B11"]            # blue, green, red, NIR, SWIR1
FEATURE_NAMES = BANDS + ["NDVI", "NDWI"]
N_FEATURES = len(FEATURE_NAMES)
REFLECTANCE_SCALE = 10000.0                         # S2 SR digital numbers -> 0..1
# ESA WorldCover v200 is a 2021 product, so use 2021 imagery to match the labels.
DATE_RANGE = ("2021-01-01", "2021-12-31")
MAX_CLOUD_PERCENT = 20
SCALE_M = 10

# ---- Classes -----------------------------------------------------------------
CLASS_NAMES = ["Water", "Vegetation", "Agricultural Land", "Built-up Area", "Barren Land"]
CLASS_COLORS = ["#2b83ba", "#1a9641", "#fdae61", "#d7191c", "#c2b280"]
N_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255                                  # nodata / unlabeled pixels

# ESA WorldCover class codes -> our 5 classes
WORLDCOVER_TO_CLASS = {
    10: 1,   # tree cover      -> Vegetation
    20: 1,   # shrubland       -> Vegetation
    30: 1,   # grassland       -> Vegetation
    40: 2,   # cropland        -> Agricultural Land
    50: 3,   # built-up        -> Built-up Area
    60: 4,   # bare / sparse   -> Barren Land
    70: 4,   # snow and ice    -> Barren Land
    80: 0,   # water           -> Water
    90: 1,   # herbaceous wetland -> Vegetation
    95: 1,   # mangroves       -> Vegetation
    100: 1,  # moss and lichen -> Vegetation
}

# ---- Patches -----------------------------------------------------------------
PATCH_SIZE = 64
STRIDE = 64
MIN_VALID_FRACTION = 0.9

# ---- Study regions (EXAMPLES - edit these) ------------------------------------
# bbox = [min_lon, min_lat, max_lon, max_lat]. Each region belongs to exactly one
# split, so the test set is spatially independent of the training data.
# Keep boxes around 0.1-0.2 degrees; at 10 m a 0.2 degree box is ~2000x2000 px.
REGIONS = {
    "pune":     {"bbox": [73.80, 18.45, 74.00, 18.60], "split": "train"},
    "nashik":   {"bbox": [73.70, 19.95, 73.90, 20.10], "split": "train"},
    "kolhapur": {"bbox": [74.15, 16.65, 74.30, 16.78], "split": "val"},
    "nagpur":   {"bbox": [79.00, 21.05, 79.18, 21.20], "split": "test"},
}
