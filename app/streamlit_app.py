# File: app/streamlit_app.py
# Updated: 2025-08-13

import streamlit as st
import sys
import os
from PIL import Image

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.segmentation import SegmentationModel, overlay_masks
from src.models.captioning import CaptioningModel  # <-- FIXED
from src.utils import apply_mask_to_image, crop_box, label_to_name

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

st.set_page_config(layout="wide")
st.title("🖼️ Image Captioning + Instance Segmentation")

with st.sidebar:
    st.markdown("**Settings**")
    score_thresh = st.slider("Detection confidence", 0.1, 0.95, 0.5, 0.05)
    top_k = st.slider("Show top K instances", 1, 10, 6, 1)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    # Load uploaded image
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    # Save uploaded image to outputs folder
    uploaded_path = os.path.join("outputs", uploaded.name)
    img.save(uploaded_path)

    with st.spinner("Loading models (first time may download weights)…"):
        seg = SegmentationModel(score_thresh=score_thresh)
        cap = CaptioningModel()  # <-- FIXED

    instances = seg.predict(img)
    if not instances:
        st.warning("No instances detected above threshold.")
    else:
        overlay = overlay_masks(img, instances[:top_k])
        st.image(overlay, caption="Overlay", use_container_width=True)

        # Save overlay image
        overlay_path = os.path.join("outputs", f"overlay_{uploaded.name}")
        overlay.save(overlay_path)

        # Global caption
        global_caption = cap.generate(img)
        st.subheader("Global caption")
        st.write(global_caption)

        # Save global caption to text file
        caption_path = os.path.join("outputs", f"caption_{os.path.splitext(uploaded.name)[0]}.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(global_caption)

        # Show & Save per-object crops
        st.subheader("Top instances")
        cols = st.columns(min(top_k, 3))
        for i, inst in enumerate(instances[:top_k]):
            with cols[i % len(cols)]:
                name = label_to_name(inst["label"])
                if inst["mask"] is None:
                    crop = crop_box(img, inst["box"])
                else:
                    masked = apply_mask_to_image(img, inst["mask"])
                    crop = crop_box(masked, inst["box"])
                st.image(crop, caption=f"{name} ({inst['score']:.2f})", use_container_width=True)
                
                # Caption for this object
                obj_caption = cap.generate(crop)
                st.caption(obj_caption)

                # Save cropped object image
                crop_filename = f"crop_{i+1}_{name}_{uploaded.name}"
                crop_path = os.path.join("outputs", crop_filename)
                crop.save(crop_path)

                # Save object caption
                obj_caption_file = os.path.join("outputs", f"caption_crop_{i+1}_{name}_{os.path.splitext(uploaded.name)[0]}.txt")
                with open(obj_caption_file, "w", encoding="utf-8") as f:
                    f.write(obj_caption)
else:
    st.info("Upload an image to begin.")
