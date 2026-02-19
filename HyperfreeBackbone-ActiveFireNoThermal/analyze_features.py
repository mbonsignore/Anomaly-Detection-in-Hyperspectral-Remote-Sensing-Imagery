import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.metrics import pairwise_distances

# ======= CONFIGURATION =======
MODE = "compare"  # Set to "single" or "compare"

# Single dataset path
FEATURE_DIR_SINGLE = "Extracted Features/features hyperseg mixed backbone"

# Comparison dataset paths
FEATURE_DIR_NORMAL = "Extracted Features/features hyperseg mixed backbone"
FEATURE_DIR_ANOMALOUS = "Extracted Features/features activefire mixed backbone oceania"

# ======= UTILS =======

def load_features(feature_dir):
    feature_list = []
    filenames = sorted([file for file in os.listdir(feature_dir) if file.endswith(".pt")])
    for file in filenames:
        path = os.path.join(feature_dir, file)
        tensor = torch.load(path)
        feature_list.append(tensor)
    return feature_list, filenames

def plot_stats(tensor, name="Feature"):
    print(f"🔍 {name}")
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Min: {tensor.min().item():.4f}")
    print(f"Max: {tensor.max().item():.4f}")
    print(f"Mean: {tensor.mean().item():.4f}")
    print(f"Std: {tensor.std().item():.4f}")
    print()

def visualize_channels(tensor, channels=[0, 1, 2, 10, 50, 100, 200]):
    upsampled = F.interpolate(tensor.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)

    selected_channels = [c for c in channels if c < upsampled.shape[0]]
    n = len(selected_channels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, c in enumerate(selected_channels):
        im = axs[i].imshow(upsampled[c].cpu().numpy(), cmap='viridis')
        axs[i].set_title(f"Channel {c}")
        axs[i].axis("off")
        fig.colorbar(im, ax=axs[i])

    for j in range(len(selected_channels), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Feature Maps - Selected Channels", fontsize=16, y=1.02)
    plt.savefig("feature_maps.png", bbox_inches='tight')
    plt.show()

def plot_energy_map(tensor):
    energy = tensor.norm(dim=0)  # shape [H, W]
    plt.imshow(energy.cpu(), cmap='inferno')
    plt.title("Feature Energy Map (Channel Norm)")
    plt.colorbar()
    plt.show()

def feature_flatten_list(feature_list):
    # stack flattened features into a single 2D array [n_samples, n_features]
    return np.stack([f.reshape(-1).cpu().numpy() for f in feature_list])

def tsne_visualization(flat_features, labels=None,  title="t-SNE of Extracted Features"):
    tsne = TSNE(n_components=2, perplexity=min(50, len(flat_features)-1), init='random', random_state=42)
    reduced = tsne.fit_transform(flat_features)
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels if labels is not None else 'blue', alpha=0.6, cmap='coolwarm')
    plt.title(title)
    plt.grid()
    plt.show()

def pca_variance_plot(flat_features, title="PCA: Explained Variance"):
    pca = PCA(n_components=min(50, flat_features.shape[1]))
    pca.fit(flat_features)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title)
    plt.grid()
    plt.show()

def compute_distance_matrix(flat1, flat2):
    return pairwise_distances(flat1, flat2, metric="cosine")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if MODE == "single":
        feature_list, filenames = load_features(FEATURE_DIR_SINGLE)

        if not feature_list:
            print("❌ No .pt files found. Check FEATURE_DIR.")
            exit()

        extracted_feature_number = 0

        print(f"✅ Loaded {len(feature_list)} feature tensors from {FEATURE_DIR_SINGLE}")

        print("🔢 First 5 feature files loaded:")
        for f in filenames[:5]:
            print(f)

        # Analyze first feature map
        plot_stats(feature_list[extracted_feature_number], filenames[extracted_feature_number])
        visualize_channels(feature_list[extracted_feature_number])
        plot_energy_map(feature_list[extracted_feature_number])
        
        # Global feature analysis
        flat = feature_flatten_list(feature_list)
        tsne_visualization(flat)
        pca_variance_plot(flat)
    elif MODE == "compare":
        features_norm, _ = load_features(FEATURE_DIR_NORMAL)
        features_anom, _ = load_features(FEATURE_DIR_ANOMALOUS)
        print(f"✅ Loaded {len(features_norm)} normal and {len(features_anom)} anomalous feature tensors")

        flat_norm = feature_flatten_list(features_norm)
        flat_anom = feature_flatten_list(features_anom)

        # t-SNE comparison
        flat_all = np.vstack((flat_norm, flat_anom))
        labels = [0] * len(flat_norm) + [1] * len(flat_anom)
        tsne_visualization(flat_all, labels, title="t-SNE: Normal vs Anomalous")

        # PCA comparison
        pca_variance_plot(flat_norm, title="PCA: Normal Features")
        pca_variance_plot(flat_anom, title="PCA: Anomalous Features")

        # Cosine distance matrix
        dists = compute_distance_matrix(flat_norm, flat_anom)
        print(f"\n📐 Cosine Distance Matrix Shape: {dists.shape}")
        print(f"Mean Cosine Distance between normal and anomalous: {dists.mean():.4f}")
        print(f"Std Deviation of Distances: {dists.std():.4f}")

    else:
        raise ValueError("❌ Invalid MODE. Use 'single' or 'compare'")