"""Export Sentinel-2 composites and ESA WorldCover labels for every region in config.REGIONS.

One-time setup:   earthengine authenticate
Run:              python scripts/gee_export.py --project YOUR_GCP_PROJECT
Then wait for the tasks (https://code.earthengine.google.com/tasks), download the GeoTIFFs
from the Drive folder and place them in data/raw/ as <region>_s2.tif and <region>_label.tif.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ee  # noqa: E402

from landcover import config  # noqa: E402


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def mask_clouds(img):
    """Mask cloud shadow, clouds and cirrus using the Scene Classification Layer."""
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(ok)


def export_region(name, bbox, folder):
    region = ee.Geometry.Rectangle(bbox)
    crs = utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(*config.DATE_RANGE)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.MAX_CLOUD_PERCENT))
          .map(mask_clouds))
    composite = s2.select(config.BANDS).median().unmask(0).toUint16().clip(region)

    # ESA WorldCover (2021). Model-derived "reference labels", not field-surveyed ground truth.
    labels = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").toUint8().clip(region)

    for img, suffix in ((composite, "s2"), (labels, "label")):
        task = ee.batch.Export.image.toDrive(
            image=img, description=f"{name}_{suffix}", folder=folder,
            fileNamePrefix=f"{name}_{suffix}", region=region, scale=config.SCALE_M,
            crs=crs, maxPixels=1e10, fileFormat="GeoTIFF")
        task.start()
        print(f"started {name}_{suffix}  ({crs}, n_images={s2.size().getInfo() if suffix == 's2' else '-'})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    p.add_argument("--folder", default="landcover", help="Google Drive folder for exports")
    a = p.parse_args()
    ee.Initialize(project=a.project)
    for name, info in config.REGIONS.items():
        export_region(name, info["bbox"], a.folder)


if __name__ == "__main__":
    main()
