from MR_testing.test_cases.base_metamorphic import BaseMetamorphicTest
from PIL import Image
from typing import Dict, Any, Tuple, List
from pathlib import Path
import torch
from MR_testing.test_cases.replacement.base_replacement import BaseReplacementTest

class CausalInteractionTest(BaseReplacementTest):
    """Test class for causal interaction replacements."""

    def __init__(self, config: Dict[str, Any], llm_evaluator, model, processor, replace_script, sam_weights):
        super().__init__(config, llm_evaluator, model, processor, replace_script, sam_weights)
        self.interaction_types = {
            'lighting': ['lamp', 'candle', 'flashlight', 'bulb'],
            'heating': ['stove', 'heater', 'fireplace', 'radiator'],
            'watering': ['sprinkler', 'hose', 'watering_can', 'fountain'],
            'supporting': ['table', 'shelf', 'stand', 'bracket']
        }

    def verify_relations(self, original: Image.Image, result: Image.Image, test_case: Dict) -> Dict[str, bool]:
        """Override to implement causal interaction verification."""
        # Get detailed descriptions focusing on interactions
        original_interaction = self._get_interaction_description(
            original, test_case['original_object']
        )
        result_interaction = self._get_interaction_description(
            result, test_case['replacement_object']
        )
        
        # Verify causal relationship maintenance
        interaction_verification = self._verify_causal_relationship(
            original_interaction,
            result_interaction,
            test_case['original_object'],
            test_case['replacement_object']
        )
        
        return {
            'causal_maintained': interaction_verification['maintained'],
            'effect_strength': interaction_verification['effect_strength'],
            'interaction_quality': interaction_verification['quality']
        }

    def _get_interaction_description(self, image: Image.Image, object_name: str) -> str:
        """Get description focusing on causal interactions."""
        interaction_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Describe specifically how the {object_name} affects or interacts with other objects or people in the scene. What is its causal effect? What changes does it create?"}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(interaction_prompt, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

    def _verify_causal_relationship(self, 
                                  original_desc: str,
                                  result_desc: str,
                                  original_obj: str,
                                  replacement_obj: str) -> Dict[str, Any]:
        """Verify if causal relationship is maintained."""
        # Find interaction category
        interaction_type = None
        for category, objects in self.interaction_types.items():
            if original_obj in objects and replacement_obj in objects:
                interaction_type = category
                break
                
        # Extract effect verbs and descriptors
        original_effects = self._extract_effect_descriptors(original_desc)
        result_effects = self._extract_effect_descriptors(result_desc)
        
        # Check effect strength
        effect_strength = self._compare_effect_strength(
            original_desc, result_desc
        )
        
        # Analyze interaction quality
        quality_metrics = self._analyze_interaction_quality(result_desc)
        
        return {
            'maintained': bool(set(original_effects) & set(result_effects)),
            'effect_strength': effect_strength,
            'quality': quality_metrics,
            'interaction_type': interaction_type
        }

    def _extract_effect_descriptors(self, text: str) -> List[str]:
        """Extract words describing effects from text."""
        effect_words = {
            'lighting': ['illuminates', 'brightens', 'lights up', 'shines'],
            'heating': ['heats', 'warms', 'increases temperature'],
            'watering': ['waters', 'moistens', 'hydrates', 'sprays'],
            'supporting': ['holds', 'supports', 'bears', 'carries']
        }
        
        found_effects = []
        for effect_list in effect_words.values():
            found_effects.extend([effect for effect in effect_list 
                                if effect in text.lower()])
        return found_effects

    def _compare_effect_strength(self, original: str, result: str) -> str:
        """Compare the strength of effects between descriptions."""
        intensity_markers = {
            'high': ['strongly', 'brightly', 'significantly', 'completely'],
            'medium': ['moderately', 'adequately', 'partially'],
            'low': ['weakly', 'dimly', 'slightly', 'barely']
        }
        
        def get_intensity(text):
            for level, markers in intensity_markers.items():
                if any(marker in text.lower() for marker in markers):
                    return level
            return 'medium'
            
        original_strength = get_intensity(original)
        result_strength = get_intensity(result)
        
        return f"{original_strength} -> {result_strength}"

    def _analyze_interaction_quality(self, text: str) -> Dict[str, bool]:
        """Analyze quality aspects of the interaction description."""
        return {
            'has_spatial_context': any(word in text.lower() 
                                     for word in ['above', 'below', 'next to', 'between']),
            'has_intensity_info': any(word in text.lower() 
                                    for word in ['strong', 'weak', 'moderate']),
            'has_effect_detail': any(word in text.lower() 
                                   for word in ['because', 'causes', 'leads to']),
            'has_temporal_info': any(word in text.lower() 
                                   for word in ['when', 'while', 'after', 'during'])
        }

    def execute_test(self, test_case: Dict) -> Dict[str, Any]:
        """Override to implement causal interaction specific testing."""
        # First run basic replacement test
        result = super().execute_test(test_case)
        
        if result['status'] == 'success':
            # Add causal interaction specific verification
            original_img = Image.open(test_case['image_path'])
            result_img = Image.open(result['result_path'])
            
            causal_verification = self.verify_relations(
                original_img,
                result_img,
                test_case
            )
            
            result['causal_verification'] = causal_verification
            
        return result