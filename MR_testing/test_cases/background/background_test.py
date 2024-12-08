# test_weather.py
# test_weather.py
from MR_testing.test_cases.background.base_background import BaseBackgroundTest
from pathlib import Path
import json
import argparse
from datetime import datetime
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

class WeatherTest:
    def __init__(self, config_path: str, llm_evaluator=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize with provided evaluator (will be LLaVA)
        self.llm_evaluator = llm_evaluator
        
        # Initialize base test
        self.base_test = BaseBackgroundTest(
            config=self.config['default_params'],
            llm_evaluator=self.llm_evaluator
        )

    def run_test(self, image_path: str, point_coords: tuple, selected_variations: list = None):
        """Run weather variation tests."""
        if not self.llm_evaluator:
            print("Warning: No LLM evaluator provided. Please initialize with LLaVA before running tests.")
            return

        # Get variations to test
        variations = self.config['variations']
        if selected_variations:
            variations = {k: v for k, v in variations.items() if k in selected_variations}

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nStarting Weather Variation Tests")
        print(f"Image: {image_path}")
        print(f"Testing {len(variations)} weather variations")
        print("-" * 50)

        # Test each variation
        for weather_type, weather_config in variations.items():
            print(f"\nTesting {weather_type} conditions:")
            print(f"Description: {weather_config['description']}")
            
            test_case = {
                'test_type': 'weather',
                'image_path': image_path,
                'point_coords': point_coords,
                'prompt': weather_config['prompt']
            }
            
            result = self.base_test.execute_test(test_case)
            
            if result['status'] == 'success':
                results.append({
                    'weather_type': weather_type,
                    'result': result
                })
                print(f"Original object: {result['original_object']}")
                print(f"After weather change: {result['result_object']}")
                print(f"Identity preserved: {'✓' if result['identity_preserved'] else '✗'}")
            else:
                print(f"Failed: {result['error']}")

        # Save results
        results_dir = Path("results/weather_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"weather_test_results_{timestamp}.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'image_path': image_path,
                'variations_tested': list(variations.keys()),
                'results': results
            }, f, indent=2)

        # Print summary
        print("\nTest Summary:")
        print("-" * 50)
        successful_tests = sum(1 for r in results if r['result']['identity_preserved'])
        print(f"Total variations tested: {len(results)}")
        print(f"Successful identity preservation: {successful_tests}/{len(results)}")
        print(f"Results saved to: {results_dir}/weather_test_results_{timestamp}.json")

def main():
    parser = argparse.ArgumentParser(description="Run weather variation tests")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--config", default="test_cases/background/configs/weather_config.json",
                       help="Path to weather test configuration")
    parser.add_argument("--coords", nargs=2, type=int, default=[750, 500],
                       help="Point coordinates (x y)")
    parser.add_argument("--variations", nargs="+", 
                       choices=["rainy", "snowy", "foggy", "sunny", "stormy"],
                       help="Specific weather variations to test")

    args = parser.parse_args()

    # Initialize LLaVa
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    llava_evaluator = {
        "model": model,
        "processor": processor
    }
    
    # Create and run tests
    tester = WeatherTest(args.config, llava_evaluator)
    tester.run_test(args.image, tuple(args.coords), args.variations)

if __name__ == "__main__":
    main()