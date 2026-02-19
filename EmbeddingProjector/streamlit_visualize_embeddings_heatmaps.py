import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from streamlit_plotly_events2 import plotly_events

# ====== PATHS ======
csv_path = "../../../data/datasets/EmbeddingProjector/streamlit_data/copernicusfm/embedding_metadata_copernicus_spectralearth7bands_activefire.csv"
patch_csv_path = "../../../data/datasets/PatchCore/results/copernicusfm/mixed/copernicuspretrainbankspectralearth7bandsnormalbalanced.csv"  

# ====== Load metadata ======
df = pd.read_csv(csv_path)

# ===== Streamlit setup =====
st.set_page_config(layout="wide")
st.title("UMAP Embedding Explorer")

# ===== Columns Layout =====
left_col, right_col = st.columns([3, 2])

def generate_heatmap_on_the_fly(csv_path, image_name):
    df = pd.read_csv(csv_path)
    df = df[df["image_name"] == image_name]
    if df.empty:
        raise ValueError(f"No data found for image: {image_name}")

    # Normalize scores globally
    global_min = df["patch_score"].min()
    global_max = df["patch_score"].max()
    if global_max == global_min:
        df["normalized_score"] = 0.5
    else:
        df["normalized_score"] = (df["patch_score"] - global_min) / (global_max - global_min)

    patch_scores = df.sort_values("patch_index")["normalized_score"].values
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
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf

with left_col:
    st.subheader("UMAP Projection")

    if not df.empty:
        df["uid"] = df["image_path"]
        unique_splits = df['split'].unique()
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
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=900,
            override_width=900,
        )
    else:
        st.warning("⚠️ No data to display.")
        selected_points = []

with right_col:
    st.subheader("Selected Image + Heatmap")

    if selected_points:
        pt = selected_points[0]
        curve_number = pt["curveNumber"]
        point_index = pt["pointIndex"]

        split_name = df["split"].unique()[curve_number]
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        selected_row = split_df.iloc[point_index]

        image_path = selected_row["image_path"]
        label = selected_row["label"]
        split = selected_row["split"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RGB Image")
            if os.path.exists(image_path):
                img = Image.open(image_path)
                max_size = (512, 512)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                st.image(img, caption=f"{label} ({split})", use_container_width=False)
            else:
                st.error(f"❌ Image not found: {image_path}")

        with col2:
            st.markdown("#### Anomaly Heatmap")
            try:
                heatmap_buf = generate_heatmap_on_the_fly(patch_csv_path, label + ".pt")
                st.image(heatmap_buf, caption=f"Heatmap: {label}", use_container_width=False)
            except Exception as e:
                st.warning(f"⚠️ Could not generate heatmap: {e}")