import random
import os
import json
from PIL import Image
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

def get_random_image(images_dir):
    """Select a random image from the directory."""
    print("Selecting a random image from the directory...")
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    if not images:
        raise ValueError("No images found in the specified directory.")

    selected_image = random.choice(images)
    image_path = os.path.join(images_dir, selected_image)
    print(f"Selected image: {image_path}")
    return image_path

def main():
    # Paths
    images_dir = "/workspaces/Advanced-MRs-for-VLMs/data/coco/val2017"
    llava_config_path = "/workspaces/Advanced-MRs-for-VLMs/MR_testing/models_to_test/llava_config.json"

    # Load LLaVa model
    print("Loading LLaVa model...")
    llava_evaluator = load_llava_model(llava_config_path)
    model = llava_evaluator["model"]
    processor = llava_evaluator["processor"]
    print("LLaVa model loaded.")

    # Print device information
    device = model.device
    print(f"Model is running on device: {device}")

    # Get a random image
    image_path = get_random_image(images_dir)
    image = Image.open(image_path)

    # Prepare prompts
    prompts = [
        "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
        "USER: <image>\nPlease describe this image\nASSISTANT:",
    ]

    # Select a random prompt
    prompt = random.choice(prompts)
    print(f"Selected prompt: {prompt}")

    # Prepare inputs
    print("Preparing inputs for the model...")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    print("Inputs prepared.")

    # Generate
    print("Generating output using the model...")
    generate_ids = model.generate(**inputs, max_new_tokens=30)
    result = processor.batch_decode(generate_ids, skip_special_tokens=True)
    print("Output generated.")

    # Print result
    print(f"Result: {result}")

if __name__ == "__main__":
    main()