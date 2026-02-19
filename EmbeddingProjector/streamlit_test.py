import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from streamlit_plotly_events2 import plotly_events
import torch

# ====== PATHS ======
csv_path = "../../../data/datasets/EmbeddingProjector/streamlit_data/embedding_metadata_floga.csv"
patch_csv_path = "../../../data/datasets/PatchCore/results/floga/copernicuspretrainbankfloganormal_patches.csv"
mask_root = "../../../data/datasets/FLOGA/FLOGA_GT"
burn_csv_path = "../../../data/datasets/FLOGA/floga_burn_pixels.csv"

# ====== Load metadata ======
df = pd.read_csv(csv_path)

# ===== Streamlit setup =====
st.set_page_config(layout="wide")
st.title("FLOGA – UMAP Embedding Explorer (Pre/Post Fire Analysis)")


# =====================================================
#  CACHE THE PATCH CSV (GLOBAL NORMALIZATION INCLUDED)
# =====================================================
@st.cache_data(show_spinner=True)
def load_patch_csv(path):
    df_p = pd.read_csv(path)
    global_min = df_p["patch_score"].min()
    global_max = df_p["patch_score"].max()
    return df_p, global_min, global_max


patch_df, patch_min, patch_max = load_patch_csv(patch_csv_path)


# =====================================================
#  HEATMAP GENERATION
# =====================================================
def generate_heatmap_on_the_fly(patch_df, global_min, global_max, image_name):

    sub = patch_df[patch_df["image_name"] == image_name]
    if sub.empty:
        raise ValueError(f"No data found for image: {image_name}")

    sub = sub.copy()
    if global_max == global_min:
        sub["normalized_score"] = 0.5
    else:
        sub["normalized_score"] = (sub["patch_score"] - global_min) / (global_max - global_min)

    patch_scores = sub.sort_values("patch_index")["normalized_score"].values
    num_patches = len(patch_scores)
    side_len = int(np.sqrt(num_patches))

    if side_len * side_len != num_patches:
        raise ValueError(f"Patch count {num_patches} is not a perfect square.")

    heatmap = patch_scores.reshape((side_len, side_len))

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
def load_ground_truth_mask(base_name):

    filename = base_name + "_mask.pt"
    full_path = os.path.join(mask_root, filename)

    if not os.path.exists(full_path):
        return None, None

    try:
        tensor = torch.load(full_path)
        mask = tensor.cpu().numpy()

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        mask = (mask > 0.5).astype("uint8") * 255
        return mask, filename

    except Exception as e:
        st.error(f"Failed to load mask {filename}: {e}")
        return None, None


# =====================================================
#  LOAD RGB IMAGE
# =====================================================
def load_rgb_image_from_df(label, df_meta):

    row = df_meta[df_meta["label"] == label]
    if row.empty:
        return None, None

    image_path = row.iloc[0]["image_path"]

    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None, image_path

    img = Image.open(image_path)
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return img, image_path


# =====================================================
#   LOAD LIST OF BURNED SAMPLES (SORTED BY BURN PIXELS)
# =====================================================
@st.cache_data
def load_burn_samples(path):
    if not os.path.exists(path):
        return []
    
    df_burn = pd.read_csv(path)

    # Ensure required columns exist
    if "sample_name" not in df_burn.columns or "num_burn_pixels" not in df_burn.columns:
        st.error("burn_pixels CSV must contain sample_name and num_burn_pixels columns.")
        return []

    # Sort by num_burn_pixels DESCENDING
    df_burn = df_burn.sort_values("num_burn_pixels", ascending=False)

    # Return sorted list of sample names
    return df_burn["sample_name"].tolist()

burn_samples = load_burn_samples(burn_csv_path)


# =====================================================
#  LEFT PANEL — UMAP VIEW + FILTERING + SEARCH
# =====================================================
left_col, right_col = st.columns([3, 4])

with left_col:
    st.subheader("UMAP Projection")

    selected_label = None

    # -----------------------------------------------------
    #   FILTER UMAP BASED ON BURNED SAMPLES
    # -----------------------------------------------------
    st.markdown("### 🔥 Burned Samples Filter")
    show_only_burned = st.checkbox("Show only burned samples in UMAP", value=False)

    if show_only_burned:
        burned_set = set(burn_samples)

        def is_burned_floga(lbl):
            base = lbl.replace("_pre", "").replace("_post", "")
            return base in burned_set

        # Keep any Copernicus split
        is_copernicus = df["split"].str.contains("copernicus", case=False, na=False)

        is_burned = df["label"].apply(is_burned_floga)

        df_vis = df[is_copernicus | is_burned].copy()
        if df_vis.empty:
            st.warning("⚠ No burned samples found — only Copernicus visible.")
    else:
        df_vis = df.copy()

    # -----------------------------------------------------
    # UMAP SCATTER PLOT (uses df_vis!)
    # -----------------------------------------------------
    if not df_vis.empty:
        df_vis = df_vis.assign(uid=df_vis["image_path"])
        unique_splits_vis = df_vis["split"].unique()  # store for click-mapping

        color_sequence = px.colors.qualitative.Set3 * (len(unique_splits_vis) // 12 + 1)

        fig = px.scatter(
            df_vis,
            x="x",
            y="y",
            color="split",
            hover_data=["label"],
            title="UMAP Projection of FLOGA Embeddings",
            color_discrete_sequence=color_sequence[:len(unique_splits_vis)],
        )

        fig.update_layout(
            height=900,
            width=900,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )

        selected_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=900,
            override_width=900,
        )
    else:
        st.warning("⚠️ No data available after filtering.")
        selected_points = []
        unique_splits_vis = np.array([])

    # -----------------------------------------------------
    # SEARCH BOX
    # -----------------------------------------------------
    with st.expander("🔍 Search for a sample by label", expanded=False):
        user_input = st.text_input(
            "Enter FLOGA sample label (e.g., sample00000082_0_2017_pre)"
        )
        if user_input:
            if user_input in df_vis["label"].values:
                selected_label = user_input
            else:
                st.error("❌ Label not found in visible UMAP.")

    # -----------------------------------------------------
    #  DROPDOWN OF BURNED SAMPLES (SORTED)
    # -----------------------------------------------------
    with st.expander("🔥 View samples with burned pixels", expanded=False):
        selected_burn = st.selectbox(
            "Select a burned sample:",
            [""] + burn_samples,   # already sorted
            index=0
        )
        if selected_burn != "":
            selected_label = selected_burn + "_post"

    # -----------------------------------------------------
    # HANDLE CLICK SELECTION (correct split-aware mapping)
    # -----------------------------------------------------
    if not selected_label and selected_points:
        pt = selected_points[0]
        curve_number = pt["curveNumber"]
        point_index = pt["pointIndex"]

        if len(unique_splits_vis) > 0 and curve_number < len(unique_splits_vis):
            split_name = unique_splits_vis[curve_number]
            subset = df_vis[df_vis["split"] == split_name].reset_index(drop=True)

            if 0 <= point_index < len(subset):
                row = subset.iloc[point_index]
                selected_label = row["label"]


# =====================================================
#  RIGHT PANEL – PRE/POST RGB + HEATMAPS + MASK
# =====================================================
with right_col:
    st.subheader("Pre/Post Visualization, Heatmaps, and GT Mask")

    if selected_label:

        base_name = selected_label.replace("_pre", "").replace("_post", "")
        pre_label = base_name + "_pre"
        post_label = base_name + "_post"

        pre_img, pre_path = load_rgb_image_from_df(pre_label, df)
        post_img, post_path = load_rgb_image_from_df(post_label, df)

        row1_col1, row1_col2 = st.columns(2)

        # -------- PRE RGB ----------
        with row1_col1:
            st.markdown("### PRE Event RGB")
            if pre_img is not None:
                st.image(pre_img, caption=f"{pre_label}\n{pre_path}")
            else:
                st.error(f"Missing PRE RGB image for {pre_label} (path: {pre_path})")

        # -------- POST RGB ----------
        with row1_col2:
            st.markdown("### POST Event RGB")
            if post_img is not None:
                st.image(post_img, caption=f"{post_label}\n{post_path}")
            else:
                st.error(f"Missing POST RGB image for {post_label} (path: {post_path})")

        # --- Heatmaps ---
        st.markdown("---")
        st.markdown("## Patch-Level Anomaly Heatmaps")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("### PRE Heatmap")
            try:
                buf_pre = generate_heatmap_on_the_fly(
                    patch_df, patch_min, patch_max, pre_label + ".pt"
                )
                st.image(buf_pre, caption=f"PRE Heatmap – {pre_label}")
            except Exception as e:
                st.error(f"Cannot load PRE heatmap for {pre_label}: {e}")

        with row2_col2:
            st.markdown("### POST Heatmap")
            try:
                buf_post = generate_heatmap_on_the_fly(
                    patch_df, patch_min, patch_max, post_label + ".pt"
                )
                st.image(buf_post, caption=f"POST Heatmap – {post_label}")
            except Exception as e:
                st.error(f"Cannot load POST heatmap for {post_label}: {e}")

        # --- GT MASK ---
        st.markdown("---")
        st.markdown("## Ground Truth Mask")

        mask, mask_file = load_ground_truth_mask(base_name)

        if mask is not None:
            mask_vis = mask.astype("float32") / 255.0
            st.image(mask_vis, caption=f"GT Mask ({mask_file})", clamp=True)
        else:
            st.info(f"ℹ️ No ground truth mask available for base name: {base_name}")

    else:
        st.info("Click a point in the UMAP or search by label to start.")