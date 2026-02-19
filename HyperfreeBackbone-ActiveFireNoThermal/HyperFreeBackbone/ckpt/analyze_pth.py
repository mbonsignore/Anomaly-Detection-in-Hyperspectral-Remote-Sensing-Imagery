import torch

# Load the checkpoint file
ckpt_path = "./HyperFreeBackbone/ckpt/HyperFree-b.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")  # Use map_location="cpu" to be safe

# Print the keys in the checkpoint
print("🔑 Top-level keys in the checkpoint:")
print(ckpt.keys())

# If it's a state_dict directly (e.g., from torch.save(model.state_dict()))
if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    print("\n🧠 Checkpoint is a pure state_dict with the following layers:\n")
    for k in ckpt:
        print(k)

# If it has nested keys (e.g., 'model', 'optimizer', 'epoch', etc.)
elif 'state_dict' in ckpt:
    print("\n📦 Checkpoint contains a nested state_dict (e.g., from training):\n")
    for k in ckpt['state_dict']:
        print(k)

# If it uses some other key (e.g., 'model_state_dict')
elif 'model_state_dict' in ckpt:
    print("\n📦 Checkpoint contains model_state_dict:\n")
    for k in ckpt['model_state_dict']:
        print(k)

else:
    print("\n⚠️ Checkpoint structure is unusual. Keys:")
    for k in ckpt:
        print(f" - {k}")