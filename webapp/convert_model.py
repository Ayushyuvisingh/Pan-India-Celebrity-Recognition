import torch

checkpoint = torch.load(
    "model/best.pt",
    map_location="cpu"
)

torch.save(
    checkpoint["model"],
    "model/best_inference.pt"
)

print("Done")