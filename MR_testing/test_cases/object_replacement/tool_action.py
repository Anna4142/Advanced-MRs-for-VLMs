from MR_testing.test_cases.base_metamorphic import BaseMetamorphicTest
from PIL import Image
from typing import Dict, Any, Tuple, List
from pathlib import Path
import torch
from MR_testing.test_cases.replacement.base_replacement import BaseReplacementTest

class ToolActionTest(BaseReplacementTest):
    """Test class for tool-action replacements."""

    def __init__(self, config: Dict[str, Any], llm_evaluator, model, processor, replace_script, sam_weights):
        super().__init__(config, llm_evaluator, model, processor, replace_script, sam_weights)
        self.tool_categories = {
            'cutting': ['knife', 'scissors', 'shears', 'cutter'],
            'writing': ['pen', 'pencil', 'marker', 'chalk'],
            'cleaning': ['broom', 'vacuum', 'mop', 'duster'],
            'cooking': ['pan', 'pot', 'wok', 'skillet']
        }

    def verify_relations(self, original: Image.Image, result: Image.Image, test_case: Dict) -> Dict[str, bool]:
        """Override to implement tool-action verification."""
        # Get detailed descriptions focusing on tool usage
        original_usage = self._get_tool_usage_description(
            original, test_case['original_object']
        )
        result_usage = self._get_tool_usage_description(
            result, test_case['replacement_object']
        )
        
        # Verify action maintenance
        action_verification = self._verify_tool_action(
            original_usage,
            result_usage,
            test_case['original_object'],
            test_case['replacement_object']
        )
        
        return {
            'action_maintained': action_verification['maintained'],
            'effectiveness': action_verification['effectiveness'],
            'tool_usage': action_verification['usage_quality']
        }

    def _get_tool_usage_description(self, image: Image.Image, tool_name: str) -> str:
        """Get description focusing on how tool is being used."""
        usage_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Describe exactly how the {tool_name} is being used. What action is being performed with it? How effectively is it being used? What is being affected by its use?"}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(usage_prompt, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=200)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].split("ASSISTANT:")[-1].strip()

    def _verify_tool_action(self,
                           original_desc: str,
                           result_desc: str,
                           original_tool: str,
                           replacement_tool: str) -> Dict[str, Any]:
        """Verify if tool action is maintained."""
        # Find tool category
        tool_category = None
        for category, tools in self.tool_categories.items():
            if original_tool in tools and replacement_tool in tools:
                tool_category = category
                break
        
        # Extract action verbs
        original_actions = self._extract_action_verbs(original_desc, tool_category)
        result_actions = self._extract_action_verbs(result_desc, tool_category)
        
        # Check effectiveness
        effectiveness = self._analyze_effectiveness(result_desc)
        
        # Analyze usage quality
        usage_quality = self._analyze_tool_usage(result_desc)
        
        return {
            'maintained': bool(set(original_actions) & set(result_actions)),
            'effectiveness': effectiveness,
            'usage_quality': usage_quality,
            'tool_category': tool_category
        }

    def _extract_action_verbs(self, text: str, category: str = None) -> List[str]:
        """Extract action verbs specific to tool category."""
        action_verbs = {
            'cutting': ['cut', 'slice', 'chop', 'trim'],
            'writing': ['write', 'draw', 'mark', 'sketch'],
            'cleaning': ['clean', 'sweep', 'wipe', 'dust'],
            'cooking': ['cook', 'fry', 'sauté', 'heat']
        }
        
        if category:
            verbs = action_verbs.get(category, [])
        else:
            verbs = [v for verbs in action_verbs.values() for v in verbs]
            
        return [verb for verb in verbs if verb in text.lower()]

    def _analyze_effectiveness(self, text: str) -> str:
        """Analyze tool usage effectiveness from description."""
        effectiveness_markers = {
            'high': ['effectively', 'efficiently', 'well', 'successfully'],
            'moderate': ['adequately', 'sufficiently', 'reasonably'],
            'low': ['struggled', 'difficult', 'awkward', 'poorly']
        }
        
        for level, markers in effectiveness_markers.items():
            if any(marker in text.lower() for marker in markers):
                return level
        return 'moderate'

    def _analyze_tool_usage(self, text: str) -> Dict[str, bool]:
        """Analyze quality aspects of tool usage description."""
        return {
            'proper_technique': any(word in text.lower() 
                                  for word in ['correctly', 'properly', 'appropriately']),
            'user_proficiency': any(word in text.lower() 
                                  for word in ['skillfully', 'expertly', 'comfortably']),
            'task_completion': any(word in text.lower() 
                                 for word in ['completed', 'finished', 'achieved']),
            'tool_control': any(word in text.lower() 
                              for word in ['controlled', 'precise', 'accurate'])
        }

    def execute_test(self, test_case: Dict) -> Dict[str, Any]:
        """Override to implement tool-action specific testing."""
        # First run basic replacement test
        result = super().execute_test(test_case)
        
        if result['status'] == 'success':
            # Add tool-action specific verification
            original_img = Image.open(test_case['image_path'])
            result_img = Image.open(result['result_path'])
            
            tool_verification = self.verify_relations(
                original_img,
                result_img,
                test_case
            )
            
            result['tool_verification'] = tool_verification
            
        return result