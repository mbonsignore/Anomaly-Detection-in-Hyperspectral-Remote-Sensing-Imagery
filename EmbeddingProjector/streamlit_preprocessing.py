#!/usr/bin/env python3
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

# ---- Common paths ----
embedding_root = "../../../data/datasets/hyperfree/Extracted Features"
rgb_root = "../../../data/datasets/EmbeddingProjector/RGB"
output_csv = "../../../data/datasets/EmbeddingProjector/streamlit_data/copernicusfm/embedding_metadata.csv"

# ----------------------------------------------------------------
# MODE 1: LEGACY MODE (multiple splits, no CSV)
# ----------------------------------------------------------------
USE_CSV_MODE = True  # <- set to True for FLOGA+CSV mode

splits = {
    # "africa": "features activefire africa",
    # "europe": "features activefire europe",
    # "oceania": "features activefire oceania",
    # "north_america": "features activefire north america",
    # "south_america": "features activefire south america",
    # "asia": "features activefire asia",
    # "copernicus_pretrain": "features copernicus pretrain",
    # "floga_pre": "features floga pre",
    # "floga_post": "features floga post",
    "spectralearth_7bands": "features spectralearth 7 bands",
    "spectralearth": "features spectralearth",
    "spectralearth_7bands_srf": "features spectralearth 7 bands srf",
}

# ----------------------------------------------------------------
# MODE 2: CSV MODE (single folder + labels CSV, e.g. FLOGA)
# ----------------------------------------------------------------
# When USE_CSV_MODE = True, these are used:

features_folder = (
    "../../../data/datasets/Copernicus-FM/Extracted Features/finetuning/features floga test semisupervised fourth run"
)
labels_csv = (
    "../../../data/datasets/FLOGA/finetuning data/floga_splits/images_test_semisupervised.csv"
)

# Assumption: RGB images for FLOGA are organized as:
#   <rgb_root>/FLOGA_PRE/<sample_name>.png (or .jpg / .jpeg)
#   <rgb_root>/FLOGA_POST/<sample_name>.png ...
# where "folder" column in CSV is either FLOGA_PRE or FLOGA_POST.
# Adjust this if your structure is different.


# ============================================================
# STORAGE
# ============================================================
records = []
all_features = []


# ============================================================
# HELPER: ADD ONE SAMPLE
# ============================================================
def add_sample_record(
    split_name: str,
    base_name: str,
    image_dir: str,
    feature_path: str,
    anomaly_label: int | None = None,
    event: str | None = None,
):
    """
    Add one sample to records + all_features, if both feature and image exist.
    """

    # Find RGB image
    image_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            image_path = os.path.abspath(candidate)
            break
    if image_path is None:
        return  # skip if no RGB

    # Load feature
    try:
        tensor = torch.load(feature_path)  # e.g. [256,4,4]
        flat = tensor.reshape(-1).numpy().astype(np.float32)
    except Exception as e:
        print(f"⚠️ Could not load {feature_path}: {e}")
        return

    rec = {
        "split": split_name,   # e.g., "spectralearth_7bands" or "floga_pre"
        "label": base_name,    # what you display in Streamlit
        "image_path": image_path,
    }

    if event is not None:
        rec["event"] = event  # "pre" / "post"

    if anomaly_label is not None:
        rec["anomaly_label"] = int(anomaly_label)

    records.append(rec)
    all_features.append(flat)


# ============================================================
# MODE 1: LEGACY MULTI-SPLIT MODE
# ============================================================
if not USE_CSV_MODE:
    for split_name, split_folder in splits.items():
        embedding_dir = os.path.join(embedding_root, split_folder)
        image_dir = os.path.join(rgb_root, f"{split_name.capitalize()}_RGB")
        print(f"🔍 Processing {split_name}...")

        if not os.path.exists(embedding_dir):
            print(f"⚠️ Missing folder: {embedding_dir}")
            continue

        feature_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith(".pt")])

        if split_name == "copernicus_pretrain":
            # Keep every 3rd sample only to reduce density
            feature_files = feature_files[::3]

        # Further reduction to speed up UMAP
        feature_files = feature_files[::2]  # Keep every 2nd sample

        for f in tqdm(feature_files, desc=f"{split_name}"):
            base = f.replace(".pt", "")
            feature_path = os.path.join(embedding_dir, f)
            add_sample_record(
                split_name=split_name,
                base_name=base,
                image_dir=image_dir,
                feature_path=feature_path,
            )

# ============================================================
# MODE 2: CSV + SINGLE FOLDER MODE (e.g., FLOGA)
# ============================================================
else:
    print("🔍 Running in CSV mode (single features folder + labels CSV)")

    if not os.path.isdir(features_folder):
        raise SystemExit(f"❌ Features folder does not exist: {features_folder}")
    if not os.path.isfile(labels_csv):
        raise SystemExit(f"❌ Labels CSV does not exist: {labels_csv}")

    df = pd.read_csv(labels_csv)

    expected_cols = {"sample_name", "folder", "label"}
    if not expected_cols.issubset(df.columns):
        raise SystemExit(
            f"CSV {labels_csv} must contain columns {sorted(expected_cols)}, found {list(df.columns)}"
        )

    # Optional: subsample rows for speed (e.g., keep every 2nd)
    #df = df.iloc[::2].copy()

    print(f"📄 Loaded {len(df)} rows from {labels_csv}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FLOGA CSV rows"):
        sample_name = row["sample_name"]  # e.g., "xxx_pre" or "yyy_post"
        folder = row["folder"]           # "FLOGA_PRE" or "FLOGA_POST"
        label = int(row["label"])        # 0 / 1

        base = sample_name  # feature + RGB base name

        # Determine event from suffix
        if base.endswith("_pre"):
            event = "pre"
            split_name = "floga_pre"   # will appear in Streamlit as a separate split
        elif base.endswith("_post"):
            event = "post"
            split_name = "floga_post"
        else:
            event = "unknown"
            split_name = "floga_unknown"

        feature_path = os.path.join(features_folder, base + ".pt")
        if not os.path.exists(feature_path):
            # feature for this sample not extracted → skip
            continue

        

        # Assume RGBs are stored as rgb_root/<folder>/<sample_name>.png
        image_dir = os.path.join(rgb_root, folder.lower().capitalize() + "_RGB")

        add_sample_record(
            split_name=split_name,
            base_name=base,
            image_dir=image_dir,
            feature_path=feature_path,
            anomaly_label=label,
            event=event,
        )


# ============================================================
# UMAP + SAVE
# ============================================================

# If nothing was loaded, bail out early
if len(all_features) == 0:
    raise SystemExit("❌ No feature vectors loaded. Check paths and configuration.")

# Convert to numpy array
all_features = np.stack(all_features).astype(np.float32)
print(f"✅ Loaded {all_features.shape[0]} feature vectors (dim={all_features.shape[1]})")

# ======= RUN UMAP =======
print("🧠 Running UMAP dimensionality reduction...")

scaled = StandardScaler().fit_transform(all_features)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=50,
    min_dist=0.1,
    random_state=42,
)
embedding = reducer.fit_transform(scaled)

print("✅ UMAP completed")

# Assign UMAP coordinates
for i, (x, y) in enumerate(embedding):
    records[i]["x"] = float(x)
    records[i]["y"] = float(y)

# ========= SAVE ==========
df_out = pd.DataFrame.from_records(records)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_out.to_csv(output_csv, index=False)

print(f"✅ CSV saved to: {output_csv}")
print(f"✅ Columns: {list(df_out.columns)}")
print(f"✅ Total samples: {len(df_out)}")