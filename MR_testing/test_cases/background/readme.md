# Background Models and Background Tests

The background models and background tests are designed to evaluate the robustness of object detection models under various weather conditions. The process involves selecting random images from a dataset, applying weather variations, and verifying if the object identity is preserved after the variations are applied.

## Components

1. **BaseMetamorphicTest**:
    - An abstract base class that defines the structure for metamorphic tests. It includes abstract methods `execute_test` and `verify_relations` that must be implemented by subclasses.

2. **BaseBackgroundTest**:
    - A concrete class that inherits from `BaseMetamorphicTest`. It focuses on background variation tests, specifically testing object identity under different weather conditions.
    - Implements methods to run image replacements, get object descriptions using LLaVa, compare object identities, and extract coordinates from COCO annotations.

3. **WeatherTestRunner**:
    - A class that manages the execution of weather variation tests. It selects random images from a dataset, applies weather variations, and verifies if the object identity is preserved.

4. **Configuration Files**:
    - `weather_config.json`: Defines the different weather variations and their corresponding prompts and descriptions.
    - `llava_config.json`: Configuration for loading the LLaVa model.

## Workflow

1. **Initialization**:
    - The `WeatherTestRunner` class is initialized with a configuration file and an LLaVa evaluator.
    - The `BaseBackgroundTest` class is initialized with default parameters and the LLaVa evaluator.

2. **Random Image Selection**:
    - The `get_random_image_and_annotations` method selects a random image from the specified directory and retrieves its annotations from the COCO annotations file.

3. **Weather Variation Tests**:
    - The `run_test` method iterates over the specified weather variations.
    - For each variation, the `execute_test` method of the `BaseBackgroundTest` class is called.
    - The `run_replacement` method applies the weather variation to the image using the `replace_anything.py` script.
    - The `compare_identity` method verifies if the object identity is preserved after the variation is applied.

4. **Results**:
    - The results are saved in the `results/weather_tests` directory as a JSON file.
    - The generated images are saved in the `results/background/weather` directory.