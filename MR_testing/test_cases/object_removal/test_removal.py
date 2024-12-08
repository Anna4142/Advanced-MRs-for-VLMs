from MR_testing.test_cases.object_removal.base_removal import BaseRemovalTest
from pathlib import Path
import json
import argparse
from datetime import datetime
from MR_testing.models_to_test.init_models import load_llava_model
import torch
import random
import os

class RemovalTestRunner:
    def __init__(self, config_path: str, llm_evaluator=None, model=None, processor=None, remove_script=None, sam_weights=None):
        print("Initializing RemovalTestRunner...")
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print("Configuration loaded.")
        
        # Initialize with provided evaluator (will be LLaVA)
        self.llm_evaluator = llm_evaluator
        
        # Initialize base test
        self.base_test = BaseRemovalTest(
            config=self.config['default_params'],
            llm_evaluator=self.llm_evaluator,
            model=model,
            processor=processor,
            remove_script=remove_script,
            sam_weights=sam_weights
        )
        print("BaseRemovalTest initialized.")

    def get_random_image_and_annotations(self, images_dir, annotations_path):
        """Select a random image from the directory and get its annotations."""
        print("Selecting a random image from the directory...")
        # List all images in the directory
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if not images:
            raise ValueError("No images found in the specified directory.")

        # Select a random image
        selected_image = random.choice(images)
        image_path = os.path.join(images_dir, selected_image)
        print(f"Selected image: {image_path}")

        # Get the image ID from the file name (assuming the file name is the image ID)
        image_id = int(os.path.splitext(selected_image)[0])
        print(f"Image ID: {image_id}")

        # Get coordinates from annotations
        point_coords = self.base_test.get_coordinates_from_annotations(annotations_path, image_id)
        print(f"Point coordinates: {point_coords}")

        return image_path, point_coords, image_id

    def run_test(self, images_dir: str, annotations_path: str, selected_variations: list = None):
        """Run removal variation tests."""
        print("Running removal variation tests...")
        if not self.llm_evaluator:
            print("Warning: No LLM evaluator provided. Please initialize with LLaVA before running tests.")
            return

        # Get a random image and its annotations
        image_path, point_coords, image_id = self.get_random_image_and_annotations(images_dir, annotations_path)

        # Get variations to test
        variations = self.config['variations']
        if selected_variations:
            variations = {k: v for k, v in variations.items() if k in selected_variations}

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nStarting Removal Variation Tests")
        print(f"Image: {image_path}")
        print(f"Testing {len(variations)} removal variations")
        print("-" * 50)

        # Test each variation
        for variation_type, variation_config in variations.items():
            print(f"\nTesting {variation_type} conditions:")
            print(f"Description: {variation_config['description']}")
            
            test_case = {
                'test_type': 'removal',
                'image_path': image_path,
                'point_coords': point_coords,
                'prompt': variation_config['prompt'],
                'target_object': variation_config['target_object']
            }
            
            result = self.base_test.execute_test(test_case)
            
            if result['status'] == 'success':
                results.append({
                    'variation_type': variation_type,
                    'result': result
                })
                print(f"Original scene: {result['original_scene']}")
                print(f"After removal: {result['result_scene']}")
                print(f"Removal success: {'✓' if result['removal_success'] else '✗'}")
            else:
                print(f"Failed: {result['error']}")

        # Save results
        results_dir = Path("results/removal_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"removal_test_results_{timestamp}.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'image_path': image_path,
                'variations_tested': list(variations.keys()),
                'results': results
            }, f, indent=2)
        print(f"Results saved to: {results_dir}/removal_test_results_{timestamp}.json")

        # Print summary
        print("\nTest Summary:")
        print("-" * 50)
        successful_tests = sum(1 for r in results if r['result']['removal_success'])
        print(f"Total variations tested: {len(results)}")
        print(f"Successful removal: {successful_tests}/{len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Run removal variation tests")
    parser.add_argument("--images_dir", required=True, help="Path to directory containing images")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations file")
    parser.add_argument("--config", default="/workspaces/Advanced-MRs-for-VLMs/MR_testing/test_cases/object_removal/configs/removal_config.json",
                       help="Path to removal test configuration")
    parser.add_argument("--variations", nargs="+", 
                       help="Specific removal variations to test")

    args = parser.parse_args()

    # Load LLaVa model
    print("Loading LLaVa model...")
    llava_evaluator = load_llava_model("/workspaces/Advanced-MRs-for-VLMs/MR_testing/models_to_test/llava_config.json")
    model = llava_evaluator["model"]
    processor = llava_evaluator["processor"]
    remove_script = "/workspaces/Advanced-MRs-for-VLMs/remove_anything.py"
    sam_weights = "/workspaces/Advanced-MRs-for-VLMs/weights/mobile_sam.pt"
    print("LLaVa model loaded.")
    
    # Create and run tests
    print("Creating RemovalTestRunner...")
    tester = RemovalTestRunner(args.config, llava_evaluator, model, processor, remove_script, sam_weights)
    print("Running tests...")
    tester.run_test(args.images_dir, args.annotations, args.variations)

if __name__ == "__main__":
    main()