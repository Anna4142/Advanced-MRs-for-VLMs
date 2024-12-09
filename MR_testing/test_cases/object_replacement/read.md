# Object Replacement Testing Framework

A framework for testing Visual Language Models (VLMs) using object replacement Metamorphic Relations (MRs). Tests how VLMs understand and describe scene changes when objects are replaced while maintaining semantic relationships.



## Metamorphic Relations

### 1. Causal Interaction Replacement (MR20)
Tests replacement of objects involved in causal interactions.

#### Example Test Cases:
- Lamp → Candle (lighting interaction)
- Fan → AC (cooling interaction)
- Heater → Fireplace (heating interaction)

#### Verification Points:
- Causal relationship maintained
- Effect consistency
- Interaction plausibility

### 2. Tool-Action Replacement (MR21)
Tests replacement of tools used in specific actions.

#### Example Test Cases:
- Knife → Scissors (cutting)
- Pen → Pencil (writing)
- Broom → Vacuum (cleaning)

#### Verification Points:
- Action preservation
- Tool effectiveness
- User adaptation required

### 3. Cause-Effect Replacement (MR22)
Tests replacement of objects that cause specific effects.

#### Example Test Cases:
- Sprinkler → Watering Can (plant watering)
- Remote → Smartphone (device control)
- Switch → Button (activation)

#### Verification Points:
- Effect maintenance
- Cause appropriateness
- Outcome consistency

## Usage

### 1. Running Tests
```python
from MR_testing.test_cases.object_replacement.causal_replacement import CausalReplacementTest

# Initialize test
test = CausalReplacementTest(
    config=config,
    llm_evaluator=llm_evaluator,
    model=model,
    processor=processor,
    replace_script="path/to/replace_anything.py",
    sam_weights="path/to/sam_weights.pt"
)

# Run test
result = test.execute_test({
    'test_type': 'causal',
    'image_path': 'path/to/image.jpg',
    'point_coords': (x, y),
    'original_object': 'lamp',
    'replacement_object': 'candle',
    'prompt': 'Replace the lamp with a candle'
})
```

### 2. Configuration
```json
{
    "causal_replacement": {
        "description": "Test causal interaction replacements",
        "valid_pairs": [
            {
                "original": "lamp",
                "replacements": ["candle", "flashlight"],
                "interaction": "illumination"
            }
        ],
        "verification_aspects": [
            "causal_maintenance",
            "effect_consistency",
            "interaction_plausibility"
        ]
    }
}
```

### 3. Adding New Test Cases
1. Create new test class inheriting from BaseReplacementTest
2. Implement verification methods
3. Add configuration in configs directory
4. Update main test runner

## MLLM Verification

Each test type uses specific prompts for verification:

### Causal Interaction:
```python
"Describe how the {replacement_object} interacts with other objects/people. Does it produce the same effects as the {original_object}?"
```

### Tool Action:
```python
"Describe how the {replacement_object} is used for the task. Is it as effective as the {original_object}?"
```

### Cause Effect:
```python
"What effect does the {replacement_object} produce? Compare it to the effect of the {original_object}."
```

## Evaluation Metrics

1. **Relationship Preservation**
   - Causal consistency
   - Action maintenance
   - Effect similarity

2. **Scene Coherence**
   - Spatial relationships
   - Context appropriateness
   - Physical plausibility

3. **Functional Equivalence**
   - Purpose fulfillment
   - Effectiveness
   - Usability



```
@article{object_replacement_testing,
    title={Testing Visual Language Models through Object Replacement Relations},
    year={2024}
}
```