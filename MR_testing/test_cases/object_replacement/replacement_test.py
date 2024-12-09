from MR_testing.test_cases.replacement.base_replacement import BaseReplacementTest
from pathlib import Path
import json
import argparse
from datetime import datetime
from MR_testing.models_to_test.init_models import load_llava_model
import torch
import random
import os

class ReplacementTestRunner:
    """Runner for replacement variation tests."""
    
    def __init__(self, config_path: str, llm_evaluator=None, model=None, processor=None, replace_script=None, sam_weights=None):
        print("Initializing ReplacementTestRunner...")
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print("Configuration loaded.")
        
        # Initialize with provided evaluator
        self.llm_evaluator = llm_evaluator
        
        # Initialize base test
        self.base_test = BaseReplacementTest(
            config=self.config['default_params'],
            llm_evaluator=self.llm_evaluator,
            model=model,
            processor=processor,
            replace_script=replace_script,
            sam_weights=sam_weights
        )
        print("BaseReplacementTest initialized.")

    def get_random_image_and_annotations(self, images_dir, annotations_path):
        """Select a random image from directory with appropriate annotations."""
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if not images:
            raise ValueError("No images found in the specified directory.")

        # Load annotations
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Try to find suitable image
        for _ in range(len(images)):
            selected_image = random.choice(images)
            image_id = int(os.path.splitext(selected_image)[0])
            
            # Check if image has required annotations
            image_annotations = [ann for ann in annotations['annotations'] 
                               if ann['image_id'] == image_id]
            
            if image_annotations:
                image_path = os.path.join(images_dir, selected_image)
                point_coords = self.base_test.get_coordinates_from_annotations(
                    annotations_path, image_id
                )
                return image_path, point_coords, image_id

        raise ValueError("No suitable images found with required annotations.")

    def run_test(self, images_dir: str, annotations_path: str, selected_variations: list = None):
        """Run replacement variation tests."""
        if not self.llm_evaluator:
            print("Warning: No LLM evaluator provided.")
            return

        # Get random image and annotations
        image_path, point_coords, image_id = self.get_random_image_and_annotations(
            images_dir, annotations_path
        )

        # Get variations to test
        variations = self.config['variations']
        if selected_variations:
            variations = {k: v for k, v in variations.items() if k in selected_variations}

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nStarting Replacement Tests")
        print(f"Image: {image_path}")
        print(f"Testing {len(variations)} variations")

        # Test each variation
        for variation_type, variation_config in variations.items():
            print(f"\nTesting {variation_type}:")
            print(f"Description: {variation_config['description']}")
            
            test_case = {
                'test_type': 'replacement',
                'image_path': image_path,
                'point_coords': point_coords,
                'prompt': variation_config['prompt'],
                'aspects_to_compare': variation_config.get('aspects_to_compare', ['all']),
                'aspects_to_verify': variation_config.get('aspects_to_verify', ['style', 'context'])
            }
            
            result = self.base_test.execute_test(test_case)
            
            if result['status'] == 'success':
                results.append({
                    'variation_type': variation_type,
                    'result': result
                })
                print("Test successful:")
                for aspect, comparison in result['scene_comparisons'].items():
                    print(f"\n{aspect.upper()} comparison:")
                    print(f"Original: {comparison['original'][:100]}...")
                    print(f"Result: {comparison['result'][:100]}...")
            else:
                print(f"Failed: {result['error']}")

        # Save results
        self._save_results(results, timestamp, image_path)
        self._print_summary(results)

    def _save_results(self, results, timestamp, image_path):
        """Save test results to file."""
        results_dir = Path("results/replacement_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"replacement_results_{timestamp}.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'image_path': str(image_path),
                'variations_tested': [r['variation_type'] for r in results],
                'results': results
            }, f, indent=2)

    def _print_summary(self, results):
        """Print test summary."""
        print("\nTest Summary:")
        print("-" * 50)
        successful_tests = sum(1 for r in results 
                             if all(v for k, v in r['result'].get('scene_comparisons', {}).items()))
        print(f"Total variations tested: {len(results)}")
        print(f"Successful replacements: {successful_tests}/{len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Run replacement variation tests")
    parser.add_argument("--images_dir", required=True, 
                       help="Path to directory containing images")
    parser.add_argument("--annotations", required=True,
                       help="Path to COCO annotations file")
    parser.add_argument("--config",
                       default="configs/replacement_config.json",
                       help="Path to replacement test configuration")
    parser.add_argument("--variations", nargs="+",
                       help="Specific replacement variations to test")

    args = parser.parse_args()

    # Load LLaVa model
    print("Loading LLaVa model...")
    llava_evaluator = load_llava_model()
    model = llava_evaluator["model"]
    processor = llava_evaluator["processor"]
    replace_script = "replace_anything.py"
    sam_weights = "weights/mobile_sam.pt"
    
    # Run tests
    tester = ReplacementTestRunner(
        args.config, llava_evaluator, model, processor, replace_script, sam_weights
    )
    tester.run_test(args.images_dir, args.annotations, args.variations)

if __name__ == "__main__":
    main()