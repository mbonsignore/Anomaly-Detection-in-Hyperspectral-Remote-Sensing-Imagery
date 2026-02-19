import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from streamlit_plotly_events2 import plotly_events

# ===== Load metadata =====
csv_path = "../../../data/datasets/EmbeddingProjector/streamlit_data/hyperfree/embedding_metadata_spectralearth7bands_activefire.csv"
df = pd.read_csv(csv_path)

# ===== Streamlit setup =====
st.set_page_config(layout="wide")
st.title("UMAP Embedding Explorer")

# ===== UMAP + Image Viewer Columns =====
left_col, right_col = st.columns([3, 2])  # Plot 3/5 width, Image 2/5

with left_col:
    st.subheader("UMAP Projection")

    if not df.empty:
        # Use image_path as unique identifier
        df["uid"] = df["image_path"]

        # Assign consistent color per split
        unique_splits = df['split'].unique()
        num_colors = len(unique_splits)
        color_sequence = px.colors.qualitative.Set3
        while len(color_sequence) < num_colors:
            color_sequence += color_sequence

        # Plot with uid as part of hover_data
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="split",
            hover_data=["label"],
            title="UMAP Projection of Feature Embeddings",
            color_discrete_sequence=color_sequence[:num_colors]
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
        st.warning("⚠️ No data to display. Please check your CSV file.")
        selected_points = []

# ===== Show image on the right =====
with right_col:
    st.subheader("Selected Image")

    if selected_points:
        pt = selected_points[0]
        curve_number = pt["curveNumber"]
        point_index = pt["pointIndex"]

        # Get the split name corresponding to the curveNumber
        split_name = unique_splits[curve_number]

        # Filter only that split
        split_df = df[df["split"] == split_name].reset_index(drop=True)

        # Get the correct row
        selected_row = split_df.iloc[point_index]

        image_path = selected_row["image_path"]
        label = selected_row["label"]
        split = selected_row["split"]

        if os.path.exists(image_path):
            img = Image.open(image_path)
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            st.image(img, caption=f"{label} ({split})", use_container_width=False)
        else:
            st.error(f"❌ Image not found: {image_path}")

'''
streamlit run streamlit_visualize_embeddings.py
'''