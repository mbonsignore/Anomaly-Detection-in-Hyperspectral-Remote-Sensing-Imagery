#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG – EDIT THESE PATHS / NAMES FOR YOUR SETUP
# ============================================================

# UMAP CSVs for the two backbones you want to compare
CSV_MODEL_1 = "../../../data/datasets/EmbeddingProjector/streamlit_data/copernicusfm/embedding_metadata_floga_test_semisupervised.csv"             #embedding_metadata_floga_test.csv
CSV_MODEL_2 = "../../../data/datasets/EmbeddingProjector/streamlit_data/copernicusfm/embedding_metadata_floga_test_finetuning_semisupervised_fourthrun.csv"  #embedding_metadata_floga_test_finetuning_centerloss.csv

MODEL_1_NAME = "Copernicus-FM (baseline)"
MODEL_2_NAME = "Copernicus-FM (finetuned semisupervised fourth run)"

# FLOGA test label CSV: sample_name, folder, label (0=normal, 1=anomalous)
LABEL_CSV_PATH = "../../../data/datasets/FLOGA/finetuning data/floga_splits/images_test_semisupervised.csv"                                     #images_test.csv

# ============================================================
# UTILITIES
# ============================================================

@st.cache_data(show_spinner=True)
def load_embedding_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {"label", "x", "y"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")
    return df


@st.cache_data(show_spinner=True)
def load_label_map(path: str):
    """
    Load images_test.csv with columns:
      - sample_name (e.g. sample00000082_0_2017_pre / _post)
      - folder (FLOGA_PRE or FLOGA_POST)
      - label (0 normal, 1 anomalous)
    Returns dict: sample_name -> int(label)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label CSV not found: {path}")

    df = pd.read_csv(path)
    expected = {"sample_name", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    # Ensure labels are ints
    df["label"] = df["label"].astype(int)
    label_map = dict(zip(df["sample_name"], df["label"]))
    return label_map


def add_anomaly_column(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    """
    Add boolean column 'is_anom' from images_test.csv mapping:
    df['label'] == sample_name
    label_map[sample_name] ∈ {0,1}
    """
    df = df.copy()

    def get_flag(lbl: str) -> bool:
        return bool(label_map.get(str(lbl), 0))

    df["is_anom"] = df["label"].apply(get_flag)
    return df


def filter_by_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """
    view ∈ {"All", "Pre only", "Post only"}
    Based on suffix in 'label' (_pre / _post).
    """
    df = df.copy()
    if view == "Pre only":
        return df[df["label"].str.endswith("_pre")]
    elif view == "Post only":
        return df[df["label"].str.endswith("_post")]
    else:
        return df


def compute_embedding_distance_stats(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compute L2 distance in UMAP space between model1 and model2
    for samples with the same label (inner join).
    Returns:
      - overall mean/std
      - per-class (anomalous vs normal) mean/std if 'is_anom' exists.
    """
    merged = pd.merge(
        df1[["label", "x", "y"] + (["is_anom"] if "is_anom" in df1.columns else [])]
            .rename(columns={"x": "x1", "y": "y1", "is_anom": "is_anom1"}),
        df2[["label", "x", "y"] + (["is_anom"] if "is_anom" in df2.columns else [])]
            .rename(columns={"x": "x2", "y": "y2", "is_anom": "is_anom2"}),
        on="label",
        how="inner",
    )

    if merged.empty:
        return None

    dx = merged["x1"].to_numpy() - merged["x2"].to_numpy()
    dy = merged["y1"].to_numpy() - merged["y2"].to_numpy()
    dist = np.sqrt(dx**2 + dy**2)

    result = {
        "n_common": int(len(merged)),
        "mean_dist": float(np.mean(dist)),
        "std_dist": float(np.std(dist)),
    }

    # Per-class stats if both have is_anom
    if "is_anom1" in merged.columns and "is_anom2" in merged.columns:
        merged["is_anom"] = merged["is_anom1"] & merged["is_anom2"]

        for flag, name in [(False, "normal"), (True, "anomalous")]:
            subset = dist[merged["is_anom"].to_numpy() == flag]
            if subset.size > 0:
                result[f"{name}_mean_dist"] = float(np.mean(subset))
                result[f"{name}_std_dist"] = float(np.std(subset))
            else:
                result[f"{name}_mean_dist"] = float("nan")
                result[f"{name}_std_dist"] = float("nan")

    return result


def make_color_map_for_split(df_all: pd.DataFrame):
    """Create a stable color map for 'split' values across both models."""
    splits = sorted(df_all["split"].dropna().unique().tolist())
    palette = px.colors.qualitative.Dark24
    if len(splits) > len(palette):
        repeats = len(splits) // len(palette) + 1
        palette = (palette * repeats)[: len(splits)]
    return {s: c for s, c in zip(splits, palette)}


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(layout="wide")
st.title("🔍 UMAP Comparison – Two Backbone Versions on FLOGA Test Set")

# ---------- Load data ----------
with st.spinner("Loading embeddings..."):
    df1 = load_embedding_csv(CSV_MODEL_1)
    df2 = load_embedding_csv(CSV_MODEL_2)

    df1["model"] = MODEL_1_NAME
    df2["model"] = MODEL_2_NAME

    # Ensure 'split' exists (some embeddings may already have it)
    if "split" not in df1.columns:
        df1["split"] = "model1"
    if "split" not in df2.columns:
        df2["split"] = "model2"

with st.spinner("Loading FLOGA test labels from images_test.csv..."):
    label_map = load_label_map(LABEL_CSV_PATH)
    df1 = add_anomaly_column(df1, label_map)
    df2 = add_anomaly_column(df2, label_map)

    # Just to reassure ourselves:
    n_anom_1 = int(df1["is_anom"].sum())
    n_anom_2 = int(df2["is_anom"].sum())
    st.info(f"Detected anomalous samples from images_test.csv: "
            f"{n_anom_1} (model 1 view), {n_anom_2} (model 2 view)")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Controls")

view_mode = st.sidebar.radio(
    "View",
    options=["All", "Pre only", "Post only"],
    index=0,
)

color_mode = st.sidebar.radio(
    "Color by",
    options=["split", "anomaly label (images_test.csv)", "none"],
    index=0,
)

point_size = st.sidebar.slider("Point size", min_value=3, max_value=15, value=6, step=1)
opacity = st.sidebar.slider("Point opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05)

highlight_label = st.sidebar.text_input(
    "Highlight specific sample (label)",
    value="",
    help="Example: sample00000082_0_2017_post",
)

subset_mode = st.sidebar.radio(
    "Subset",
    options=["All", "Normal only", "Anomalous only"],
    index=0,
)

# Apply view filters
df1_vis = filter_by_view(df1, view_mode)
df2_vis = filter_by_view(df2, view_mode)

# Apply anomaly subset
if subset_mode != "All":
    flag = (subset_mode == "Anomalous only")
    df1_vis = df1_vis[df1_vis["is_anom"] == flag]
    df2_vis = df2_vis[df2_vis["is_anom"] == flag]

# Build color info
color_arg_1 = None
color_arg_2 = None
color_discrete_map = None

if color_mode == "split":
    all_for_color = pd.concat([df1_vis, df2_vis], ignore_index=True)
    color_discrete_map = make_color_map_for_split(all_for_color)
    color_arg_1 = "split"
    color_arg_2 = "split"

elif color_mode == "anomaly label (images_test.csv)":
    color_discrete_map = {
        False: "#1f77b4",  # blue – normal
        True: "#d62728",   # red – anomalous
    }
    color_arg_1 = "is_anom"
    color_arg_2 = "is_anom"

else:
    color_arg_1 = None
    color_arg_2 = None
    color_discrete_map = None


# ============================================================
# MAIN LAYOUT – TWO UMAPS SIDE BY SIDE
# ============================================================
col_left, col_right = st.columns(2)

# ---------- MODEL 1 ----------
with col_left:
    st.subheader(f"UMAP – {MODEL_1_NAME}")

    df_plot = df1_vis.copy()
    if highlight_label:
        df_plot["__highlight__"] = (df_plot["label"] == highlight_label)
    else:
        df_plot["__highlight__"] = False

    fig1 = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=color_arg_1,
        hover_data=["label", "split"] if "split" in df_plot.columns else ["label"],
        color_discrete_map=color_discrete_map,
        title=MODEL_1_NAME,
    )

    fig1.update_traces(
        marker=dict(size=point_size, opacity=opacity),
        selector=dict(mode="markers"),
    )

    if highlight_label and df_plot["__highlight__"].any():
        df_high = df_plot[df_plot["__highlight__"]]
        fig1.add_scatter(
            x=df_high["x"],
            y=df_high["y"],
            mode="markers",
            marker=dict(size=point_size + 4, color="black", symbol="x"),
            name="highlight",
        )

    fig1.update_layout(
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.1),
    )

    st.plotly_chart(fig1, use_container_width=True)

# ---------- MODEL 2 ----------
with col_right:
    st.subheader(f"UMAP – {MODEL_2_NAME}")

    df_plot = df2_vis.copy()
    if highlight_label:
        df_plot["__highlight__"] = (df_plot["label"] == highlight_label)
    else:
        df_plot["__highlight__"] = False

    fig2 = px.scatter(
        df_plot,
        x="x",
        y="y",
        color=color_arg_2,
        hover_data=["label", "split"] if "split" in df_plot.columns else ["label"],
        color_discrete_map=color_discrete_map,
        title=MODEL_2_NAME,
    )

    fig2.update_traces(
        marker=dict(size=point_size, opacity=opacity),
        selector=dict(mode="markers"),
    )

    if highlight_label and df_plot["__highlight__"].any():
        df_high = df_plot[df_plot["__highlight__"]]
        fig2.add_scatter(
            x=df_high["x"],
            y=df_high["y"],
            mode="markers",
            marker=dict(size=point_size + 4, color="black", symbol="x"),
            name="highlight",
        )

    fig2.update_layout(
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.1),
    )

    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# QUANTITATIVE COMPARISON – DISTANCE STATS
# ============================================================
st.markdown("---")
st.subheader("📊 Quantitative comparison between embeddings")

stats = compute_embedding_distance_stats(df1_vis, df2_vis)

if stats is None:
    st.warning("Not enough common labels between the two views after filtering.")
else:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Common samples", stats["n_common"])
    col_b.metric("Mean L2 distance (all)", f"{stats['mean_dist']:.4f}")
    col_c.metric("Std L2 distance (all)", f"{stats['std_dist']:.4f}")

    if "normal_mean_dist" in stats:
        st.markdown("**Per-class distance statistics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Normal samples**")
            st.write(f"- mean distance: `{stats['normal_mean_dist']:.4f}`")
            st.write(f"- std distance: `{stats['normal_std_dist']:.4f}`")
        with col2:
            st.write("**Anomalous samples**")
            st.write(f"- mean distance: `{stats['anomalous_mean_dist']:.4f}`")
            st.write(f"- std distance: `{stats['anomalous_std_dist']:.4f}`")


# ============================================================
# OPTIONAL: SPREAD / CLUSTER COMPACTNESS
# ============================================================
st.markdown("---")
st.subheader("📌 Spread of normal vs anomalous embeddings (per model)")


def spread_stats(df: pd.DataFrame, name: str):
    if "is_anom" not in df.columns:
        st.write(f"Model **{name}**: no anomaly info available.")
        return

    res = {}
    for flag, label in [(False, "normal"), (True, "anomalous")]:
        sub = df[df["is_anom"] == flag]
        if sub.empty:
            res[label] = None
            continue
        xs = sub["x"].to_numpy()
        ys = sub["y"].to_numpy()
        r = np.sqrt(xs**2 + ys**2)
        res[label] = {
            "n": len(sub),
            "x_std": float(np.std(xs)),
            "y_std": float(np.std(ys)),
            "radial_std": float(np.std(r)),
        }

    st.write(f"**{name}**")
    for cls in ["normal", "anomalous"]:
        stats_cls = res.get(cls)
        if stats_cls is None:
            st.write(f"- {cls}: no samples")
        else:
            st.write(
                f"- {cls}: n={stats_cls['n']}, "
                f"x_std={stats_cls['x_std']:.3f}, "
                f"y_std={stats_cls['y_std']:.3f}, "
                f"radial_std={stats_cls['radial_std']:.3f}"
            )


col_s1, col_s2 = st.columns(2)
with col_s1:
    spread_stats(df1_vis, MODEL_1_NAME)
with col_s2:
    spread_stats(df2_vis, MODEL_2_NAME)