import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from streamlit_plotly_events2 import plotly_events
import tifffile

# ====== PATHS ======
csv_path = "../../../data/datasets/EmbeddingProjector/streamlit_data/copernicusfm/embedding_metadata_spectralearth7bands_activefire.csv"
patch_csv_path = "../../../data/datasets/PatchCore/results/copernicusfm/spectralearth7bands/random/0.1_patches.csv"
mask_root = "../../../data/datasets/ActiveFire/masks"

# ====== Load metadata ======
df = pd.read_csv(csv_path)

# ===== Streamlit setup =====
st.set_page_config(layout="wide")
st.title("UMAP Embedding Explorer")


# =====================================================
#  CACHE THE PATCH CSV (GLOBAL NORMALIZATION INCLUDED)
# =====================================================
@st.cache_data(show_spinner=True)
def load_patch_csv(path):
    df = pd.read_csv(path)
    global_min = df["patch_score"].min()
    global_max = df["patch_score"].max()
    return df, global_min, global_max


patch_df, patch_min, patch_max = load_patch_csv(patch_csv_path)


# =====================================================
#  HEATMAP GENERATION — uses cached CSV + global min/max
# =====================================================
def generate_heatmap_on_the_fly(patch_df, global_min, global_max, image_name):
    sub = patch_df[patch_df["image_name"] == image_name]
    if sub.empty:
        raise ValueError(f"No data found for image: {image_name}")

    # Global normalization
    if global_max == global_min:
        sub["normalized_score"] = 0.5
    else:
        sub["normalized_score"] = (sub["patch_score"] - global_min) / (global_max - global_min)

    patch_scores = sub.sort_values("patch_index")["normalized_score"].values
    num_patches = len(patch_scores)
    side_len = int(np.sqrt(num_patches))

    if side_len * side_len != num_patches:
        raise ValueError(f"Patch count {num_patches} is not a perfect square.")

    # Reshape into grid
    heatmap = patch_scores.reshape((side_len, side_len))

    # Render as PNG
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(image_name, fontsize=8)
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf


# =====================================================
#  GROUND TRUTH MASK LOADING
# =====================================================
def load_ground_truth_mask(label):
    priority = ["intersection", "voting", "Kumar-Roy", "Murphy", "Schroeder"]

    for m in priority:
        candidate = label.replace("p", f"{m}_p") + ".tif"
        full_path = os.path.join(mask_root, candidate)

        if os.path.exists(full_path):
            try:
                arr = tifffile.imread(full_path)

                # Fix shape (256,256,1) → (256,256)
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)

                # Normalize binary masks
                if arr.max() <= 1:
                    arr = (arr * 255).astype(np.uint8)

                return arr, m, candidate
            except Exception as e:
                st.warning(f"⚠️ Failed to read mask {candidate}: {e}")

    return None, None, None


# =====================================================
#  LEFT PANEL — UMAP VIEW + SEARCH
# =====================================================
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("UMAP Projection")

    selected_label = None

    if not df.empty:
        df["uid"] = df["image_path"]
        unique_splits = df["split"].unique()
        color_sequence = px.colors.qualitative.Set3 * (len(unique_splits) // 12 + 1)

        fig = px.scatter(
            df, x="x", y="y", color="split", hover_data=["label"],
            title="UMAP Projection of Feature Embeddings",
            color_discrete_sequence=color_sequence[:len(unique_splits)]
        )

        fig.update_layout(
            height=900, width=900,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        selected_points = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            override_height=900, override_width=900,
        )
    else:
        st.warning("⚠️ No data available.")
        selected_points = []

    # SEARCH BOX
    with st.expander("🔍 Search for a sample by label", expanded=False):
        user_input = st.text_input(
            "Enter image label (e.g., LC08_L1TP_166067_20200816_20200816_01_RT_p00291)"
        )
        if user_input:
            if user_input in df["label"].values:
                selected_label = user_input
            else:
                st.error("❌ Label not found.")

    # If not from search, use click
    if not selected_label and selected_points:
        pt = selected_points[0]
        curve_number = pt["curveNumber"]
        point_index = pt["pointIndex"]
        split_name = df["split"].unique()[curve_number]
        row = df[df["split"] == split_name].reset_index(drop=True).iloc[point_index]
        selected_label = row["label"]


# =====================================================
#  RIGHT PANEL — RGB + HEATMAP + MASK
# =====================================================
with right_col:
    st.subheader("Selected Image + Heatmap + GT Mask")

    if selected_label:
        row = df[df["label"] == selected_label].iloc[0]
        image_path = row["image_path"]
        split = row["split"]

        col1, col2 = st.columns(2)

        # -------- RGB IMAGE ----------
        with col1:
            st.markdown("#### RGB Image")
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                st.image(img, caption=f"{selected_label} ({split})", use_container_width=False)
            else:
                st.error("❌ Image not found.")

        # -------- HEATMAP ----------
        with col2:
            st.markdown("#### Anomaly Heatmap")
            try:
                heatmap_buf = generate_heatmap_on_the_fly(
                    patch_df, patch_min, patch_max, selected_label + ".pt"
                )
                st.image(heatmap_buf, caption=f"Heatmap: {selected_label}", use_container_width=False)
            except Exception as e:
                st.warning(f"⚠️ Cannot generate heatmap: {e}")

        # -------- MASK ----------
        st.markdown("#### Ground Truth Mask")

        mask, mask_type, mask_filename = load_ground_truth_mask(selected_label)
        if mask is not None:
            st.image(mask, caption=f"GT Mask ({mask_type} - {mask_filename})", clamp=True)
        else:
            st.info("ℹ️ No mask available for this sample.")