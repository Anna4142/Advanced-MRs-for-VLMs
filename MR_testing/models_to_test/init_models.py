import json
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_llava_model(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_id = config["model_id"]
    torch_dtype = getattr(torch, config["torch_dtype"])
    device_map = config["device_map"]

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map)

    return {
        "model": model,
        "processor": processor
    }