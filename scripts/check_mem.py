"""Quick script to check GPU memory usage after model load and encoder offload."""
import torch
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.loader import load_cambrian

tokenizer, model, ip, ctx = load_cambrian(model_path="nyu-visionx/cambrian-8b", gpu_ids=[0])
print(f"After load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:   {torch.cuda.memory_reserved()/1e9:.2f} GB")

inner = getattr(model, "model", model)
towers = getattr(inner, "vision_tower_aux_list", None)
if towers:
    for t in towers:
        t.cpu()
    torch.cuda.empty_cache()

print(f"After offload: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:      {torch.cuda.memory_reserved()/1e9:.2f} GB")
