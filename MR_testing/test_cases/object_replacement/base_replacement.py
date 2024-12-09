from MR_testing.test_cases.base_metamorphic import BaseMetamorphicTest
from PIL import Image
from typing import Dict, Any, Tuple, List
from pathlib import Path
import subprocess
import torch
import json

class BaseReplacementTest(BaseMetamorphicTest):
    """Base class for replacement variation tests."""
    
    def __init__(self, config: Dict[str, Any], llm_evaluator, model, processor, replace_script, sam_weights):
        super().__init__(config, llm_evaluator)
        self.replace_script = replace_script
        self.sam_weights = sam_weights
        self.model = model
        self.processor = processor

    def run_replacement(self, 
                       image_path: Path, 
                       point_coords: Tuple[int, int],
                       prompt: str,
                       output_dir: Path) -> Dict[str, Any]:
        """Execute replace_anything.py with given parameters."""
        print(f"Running replacement with prompt: {prompt}")
        cmd = [
            "python", self.replace_script,
            "--input_img", str(image_path),
            "--coords_type", "key_in",
            "--point_coords", str(point_coords[0]), str(point_coords[1]),
            "--point_labels", "1",
            "--text_prompt", prompt,
            "--output_dir", str(output_dir),
            "--sam_model_type", "vit_h",
            "--sam_ckpt", self.sam_weights
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result_path = output_dir / "replaced_with_mask_0.png"
            
            if result_path.exists():
                return {
                    'status': 'success',
                    'output_path': result_path,
                    'output_image': Image.open(result_path)
                }
            return {
                'status': 'failed', 
                'error': 'Output image not found'
            }
        except subprocess.CalledProcessError as e:
            return {
                'status': 'failed', 
                'error': str(e)
            }

    def get_scene_description(self, image: Image.Image, focus: str = "all") -> str:
        """Get scene description focusing on specific aspects."""
        focus_prompts = {
            "all": "Describe all objects and their relationships in the scene.",
            "style": "Describe the visual style and aesthetic of the scene.",
            "context": "Describe how objects relate to and interact with each other.",
            "lighting": "Describe the lighting conditions and shadows in the scene."
        }
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": focus_prompts[focus]},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

    def compare_scenes(self, original: Image.Image, result: Image.Image, aspects: List[str] = None) -> Dict[str, Any]:
        """Compare original and result images across multiple aspects."""
        aspects = aspects or ["all", "style", "context", "lighting"]
        comparisons = {}
        
        for aspect in aspects:
            original_desc = self.get_scene_description(original, aspect)
            result_desc = self.get_scene_description(result, aspect)
            comparisons[aspect] = {
                'original': original_desc,
                'result': result_desc
            }
            
        return comparisons

    def execute_test(self, test_case: Dict) -> Dict[str, Any]:
        """Execute the replacement test and return results."""
        image_path = Path(test_case['image_path'])
        results_dir = Path(f"results/replacement/{test_case['test_type']}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get original scene description
        original_img = Image.open(image_path)
        original_desc = self.get_scene_description(original_img)

        # Apply replacement
        result = self.run_replacement(
            image_path,
            test_case['point_coords'],
            test_case['prompt'],
            results_dir
        )

        if result['status'] == 'success':
            comparisons = self.compare_scenes(
                original_img, 
                result['output_image'],
                test_case.get('aspects_to_compare')
            )
            
            return {
                'status': 'success',
                'original_path': str(image_path),
                'result_path': str(result['output_path']),
                'original_scene': original_desc,
                'scene_comparisons': comparisons
            }
        else:
            return {
                'status': 'failed',
                'error': result['error']
            }

    def verify_relations(self, original: Image.Image, result: Image.Image, test_case: Dict) -> Dict[str, bool]:
        """Verify relations between original and result image."""
        aspects_to_verify = test_case.get('aspects_to_verify', ['style', 'context', 'lighting'])
        comparisons = self.compare_scenes(original, result, aspects_to_verify)
        
        # Simple verification - check if key descriptors are preserved
        verifications = {}
        for aspect in aspects_to_verify:
            original_desc = comparisons[aspect]['original'].lower()
            result_desc = comparisons[aspect]['result'].lower()
            
            # Count common significant words
            original_words = set(original_desc.split())
            result_words = set(result_desc.split())
            common_words = original_words.intersection(result_words)
            
            verifications[f'{aspect}_preserved'] = len(common_words) / len(original_words) > 0.5
            
        return verifications