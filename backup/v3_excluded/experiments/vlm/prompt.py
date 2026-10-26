"""The one vision prompt shared by every unfinished look-at-image experiment.

The prompt has no weight, no dataset name, and no sample id.
"""
from __future__ import annotations

from mmcfa.runtime import sha256_text

PROMPT = """Look at this food photograph and answer from the image only.

Return one JSON object with exactly these keys:
- food_name: a short English name for the food
- form: one of rigid, prepared, liquid, stacked
- references: an array, possibly empty, of visible objects that have a standard physical size. Each item has object (coin, card, utensil, or other), bbox_1000, size_mm, and conf. bbox_1000 is [x0, y0, x1, y1] with the image width and height each scaled to 1000. size_mm is a positive number only when you know that object's standard size; a Chinese 1-yuan coin is 25 mm. conf is from 0 to 1.
- weight_guess_g: a positive number, your estimate of the edible food weight in grams
- scene: a short phrase describing the setting
- out_of_domain: true when this is a casual real-world photograph that a model trained on overhead cafeteria trays would not be calibrated for, such as a restaurant table, packaged food, a handheld angle, or several meals in one frame. false for an overhead tray, a controlled laboratory photo of one food, or a clean render of one food.

There is no weight printed in the task text. Use an empty references array when no sized object is visible.
"""

PROMPT_SHA256 = sha256_text(PROMPT)
