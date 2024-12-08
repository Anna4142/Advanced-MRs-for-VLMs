from MR_testing.test_cases.object_removal.base_removal import BaseRemovalTest
from pathlib import Path
import json
import argparse
from datetime import datetime
import torch
import random
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class AreaGroundTruth:
    """Ground truth for area-based removal."""
    areas: List[float]  # Sorted areas from largest to smallest
    expected_areas: List[float]  # Areas after removing largest
    instance_ids: List[int]  # Corresponding instance IDs
    coords: List[Tuple[int, int]]  # Center coordinates for each instance

class AreaBasedTestRunner:
    def __init__(self, config_path: str, llm_evaluator=None, model=None, processor=None, remove_script=None, sam_weights=None):
        print("Initializing AreaBasedTestRunner...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.llm_evaluator = llm_evaluator
        self.model = model
        self.processor = processor
        
        self.base_test = BaseRemovalTest(
            config=self.config['default_params'],
            llm_evaluator=self.llm_evaluator,
            model=model,
            processor=processor,
            remove_script=remove_script,
            sam_weights=sam_weights
        )
        print("BaseRemovalTest initialized.")

    def get_area_based_instances(self, annotations_path: str, image_id: int, target_object: str, min_instances: int = 3) -> Optional[AreaGroundTruth]:
        """Get instances sorted by area with ground truth."""
        print(f"Getting area-based instances for image {image_id}")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        # Get category ID for target object
        categories = {cat['name'].lower(): cat['id'] for cat in coco_data['categories']}
        category_id = categories.get(target_object.lower())
        if not category_id:
            return None

        # Get instances of target category
        instances = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_id and ann['category_id'] == category_id:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]  # width * height
                center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                instances.append((area, ann['id'], center))

        if len(instances) < min_instances:
            print(f"Found only {len(instances)} instances, need at least {min_instances}")
            return None

        # Sort by area
        instances.sort(reverse=True)  # Largest first
        
        return AreaGroundTruth(
            areas=[area for area, _, _ in instances],
            expected_areas=[area for area, _, _ in instances[1:]],  # Remove largest
            instance_ids=[id for _, id, _ in instances],
            coords=[center for _, _, center in instances]
        )

    def verify_area_removal(self, original_image: Path, result_image: Path, target_object: str) -> Dict[str, Any]:
        """Verify area-based removal using LLaVA."""
        # Prompt LLaVA to count and describe size relationships
        size_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"List all {target_object}s from largest to smallest. Give ONLY ordered numbers and descriptions of their locations."}
                ]
            }
        ]

        # Get descriptions
        def get_llava_description(image_path: Path) -> str:
            prompt = self.processor.apply_chat_template(size_prompt, add_generation_prompt=True)
            inputs = self.processor(images=image_path, text=prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=200)
            return self.processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

        original_desc = get_llava_description(original_image)
        result_desc = get_llava_description(result_image)

        return {
            'original_description': original_desc,
            'result_description': result_desc,
            'largest_removed': 'largest' not in result_desc.lower()
        }

    def run_test(self, images_dir: str, annotations_path: str, selected_variations: list = None):
        """Run area-based removal tests."""
        print("Running area-based removal tests...")
        
        # Get a random image
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if not images:
            raise ValueError("No images found in the directory.")

        found_valid_image = False
        for _ in range(len(images)):  # Try all images if needed
            selected_image = random.choice(images)
            image_id = int(os.path.splitext(selected_image)[0])
            image_path = os.path.join(images_dir, selected_image)
            
            # Get ground truth for areas
            ground_truth = self.get_area_based_instances(
                annotations_path,
                image_id,
                self.config['target_object']
            )
            
            if ground_truth:
                found_valid_image = True
                break

        if not found_valid_image:
            print("No images found with sufficient instances.")
            return

        print(f"Selected image: {image_path}")
        print(f"Found {len(ground_truth.areas)} instances")
        print(f"Areas (sorted): {ground_truth.areas}")

        # Run removal test on largest instance
        test_case = {
            'test_type': 'area_based',
            'image_path': image_path,
            'point_coords': ground_truth.coords[0],  # Coordinates of largest instance
            'prompt': f"Remove the largest {self.config['target_object']}",
            'target_object': self.config['target_object']
        }

        result = self.base_test.execute_test(test_case)
        
        if result['status'] == 'success':
            # Verify removal using LLaVA
            verification = self.verify_area_removal(
                Path(image_path),
                Path(result['result_path']),
                self.config['target_object']
            )
            
            # Save results
            results_dir = Path("results/area_removal_tests")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(results_dir / f"area_test_results_{timestamp}.json", 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'image_path': image_path,
                    'ground_truth': {
                        'original_areas': ground_truth.areas,
                        'expected_areas': ground_truth.expected_areas
                    },
                    'result': result,
                    'verification': verification
                }, f, indent=2)

            print("\nTest Summary:")
            print("-" * 50)
            print(f"Largest area removal successful: {verification['largest_removed']}")
            print(f"Original description: {verification['original_description']}")
            print(f"Result description: {verification['result_description']}")

def main():
    parser = argparse.ArgumentParser(description="Run area-based removal tests")
    parser.add_argument("--images_dir", required=True, help="Path to directory containing images")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations file")
    parser.add_argument("--config", default="configs/area_removal_config.json",
                       help="Path to area removal test configuration")

    args = parser.parse_args()

    # Load LLaVa model
    print("Loading LLaVa model...")
    from MR_testing.models_to_test.init_models import load_llava_model
    llava_evaluator = load_llava_model("/workspaces/Advanced-MRs-for-VLMs/MR_testing/models_to_test/llava_config.json")
    model = llava_evaluator["model"]
    processor = llava_evaluator["processor"]
    remove_script = "/workspaces/Advanced-MRs-for-VLMs/remove_anything.py"
    sam_weights = "/workspaces/Advanced-MRs-for-VLMs/weights/mobile_sam.pt"
    
    # Run tests
    tester = AreaBasedTestRunner(args.config, llava_evaluator, model, processor, remove_script, sam_weights)
    tester.run_test(args.images_dir, args.annotations)