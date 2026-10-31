"""Streamlit dashboard: original image, classified map and land-cover area statistics.

    streamlit run app/streamlit_app.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from landcover import config  # noqa: E402
from landcover.predict import (area_stats, classify_raster, load_model,  # noqa: E402
                               pixel_area_from_profile, write_geotiff)

st.set_page_config(page_title="Land-Cover Classifier", layout="wide")
st.title("Sentinel-2 Land-Cover Classification")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    return load_model(path)


def colorize(class_map):
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(config.CLASS_COLORS):
        lut[i] = [int(c[j:j + 2], 16) for j in (1, 3, 5)]
    return lut[class_map]


def true_color(feats):
    rgb = feats[..., [2, 1, 0]]
    lo, hi = np.percentile(rgb[rgb > 0], (2, 98)) if (rgb > 0).any() else (0, 1)
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def downsample(img, max_side=1200):
    step = max(1, int(np.ceil(max(img.shape[:2]) / max_side)))
    return img[::step, ::step]


with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model", ["unet", "cnn"],
                              help="U-Net gives a pixel-level map; the CNN gives a blocky 64x64 patch map.")
    model_path = st.text_input("Model file", str(config.MODEL_DIR / f"{model_type}.keras"))
    uploaded = st.file_uploader("Sentinel-2 GeoTIFF (B2,B3,B4,B8,B11)", type=["tif", "tiff"])
    local = sorted(p.name for p in config.RAW_DIR.glob("*_s2.tif"))
    chosen = st.selectbox("...or a file from data/raw", ["-"] + local)
    run = st.button("Classify", type="primary")

if run:
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}. Train one first (see README).")
        st.stop()
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(uploaded.getvalue())
        tmp.close()
        src = tmp.name
    elif chosen != "-":
        src = str(config.RAW_DIR / chosen)
    else:
        st.warning("Upload a GeoTIFF or choose one from data/raw.")
        st.stop()

    with st.spinner("Classifying..."):
        class_map, profile, feats = classify_raster(src, get_model(model_path), model_type)
        stats = area_stats(class_map, pixel_area_from_profile(profile))
        out = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        out.close()
        write_geotiff(class_map, profile, out.name)
        tif_bytes = Path(out.name).read_bytes()
    st.session_state["result"] = (feats, class_map, stats, tif_bytes)

if "result" in st.session_state:
    feats, class_map, stats, tif_bytes = st.session_state["result"]
    c1, c2 = st.columns(2)
    c1.subheader("Original (true colour)")
    c1.image(downsample(true_color(feats)), use_container_width=True)
    c2.subheader("Classified map")
    c2.image(downsample(colorize(class_map)), use_container_width=True)

    legend = " &nbsp; ".join(f"<span style='color:{c}'>&#9632;</span> {n}"
                             for n, c in zip(config.CLASS_NAMES, config.CLASS_COLORS))
    st.markdown(legend, unsafe_allow_html=True)

    st.subheader("Area by land-cover class")
    t1, t2 = st.columns([1, 1])
    t1.dataframe(stats[["class", "area_km2", "percent"]].round(2), hide_index=True, use_container_width=True)
    t2.bar_chart(stats.set_index("class")["percent"])
    st.download_button("Download classified GeoTIFF", tif_bytes, "classified.tif", "image/tiff")
else:
    st.info("Choose a model and a Sentinel-2 GeoTIFF in the sidebar, then click **Classify**.")
