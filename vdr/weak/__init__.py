"""Train-only weak step labels for the controller.

The teacher reads Nutrition5K train ingredient lists and train nutrition.
It does not read rgb_test or depth_test, and it does not call the VLM.
"""

from .teacher import (
    TEACHER_VERSION,
    TRAIN_SPLIT,
    TestSplitRefused,
    UnsafeOutputPath,
    generate_dish_steps,
    generate_split,
    pool_of_dish,
    refuse_test_split,
)

__all__ = [
    "TEACHER_VERSION",
    "TRAIN_SPLIT",
    "TestSplitRefused",
    "UnsafeOutputPath",
    "generate_dish_steps",
    "generate_split",
    "pool_of_dish",
    "refuse_test_split",
]
