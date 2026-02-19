#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path


def bytes_to_mb(x):
    return x / (1024 ** 2)


def inspect_state_dict(state_dict):
    total_params = 0
    total_bytes = 0
    dtype_count = {}

    largest_tensors = []

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue

        numel = tensor.numel()
        element_size = tensor.element_size()  # bytes per element

        total_params += numel
        total_bytes += numel * element_size

        dtype = str(tensor.dtype)
        dtype_count[dtype] = dtype_count.get(dtype, 0) + numel

        largest_tensors.append((name, numel * element_size))

    largest_tensors.sort(key=lambda x: x[1], reverse=True)

    print("\n=== CHECKPOINT PARAMETER SUMMARY ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter memory (from tensors): {bytes_to_mb(total_bytes):.2f} MB")

    print("\nDtype distribution:")
    for dtype, count in dtype_count.items():
        print(f"  {dtype}: {count:,} parameters")

    print("\nTop 5 largest tensors:")
    for name, size in largest_tensors[:5]:
        print(f"  {name} -> {bytes_to_mb(size):.2f} MB")

    return total_params, total_bytes


def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch .pth checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to .pth file")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{ckpt_path} not found")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] File size on disk: {bytes_to_mb(ckpt_path.stat().st_size):.2f} MB")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Case 1: checkpoint contains backbone_state_dict
    if isinstance(checkpoint, dict):
        if "backbone_state_dict" in checkpoint:
            print("[INFO] Found 'backbone_state_dict' in checkpoint.")
            state_dict = checkpoint["backbone_state_dict"]
        elif "state_dict" in checkpoint:
            print("[INFO] Found 'state_dict' in checkpoint.")
            state_dict = checkpoint["state_dict"]
        else:
            # Might already be a raw state_dict
            print("[INFO] Assuming checkpoint is a raw state_dict.")
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    inspect_state_dict(state_dict)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
