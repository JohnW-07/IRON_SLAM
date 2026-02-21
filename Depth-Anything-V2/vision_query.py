"""
vision_query.py

Sends the site photo + list of object categories (from as_designed.json) to a vision LLM.
The LLM returns pixel bounding boxes for each visible object.
Outputs as_built.json.

SAMPLE PROMPT (used internally):
  See VISION_PROMPT below — swap in your preferred API call.
"""

import json
import os
import base64


# ── Prompt template ─────────────────────────────────────────────────────────
VISION_PROMPT = """You are an expert construction site inspector AI.

I am giving you a photo of a construction site. I also have a list of object categories 
that should be present based on the architectural 3D model (As-Designed).

Your task:
1. Look at the image carefully.
2. For each category below, identify if the object is VISIBLE in the image.
3. If visible, return its pixel bounding box [x_min, y_min, x_max, y_max] and a short label.
4. If NOT visible, mark it as "not_visible".

Object categories to look for:
{categories}

Respond ONLY with valid JSON in exactly this format:
{{
  "objects": [
    {{
      "category": "HVAC_Duct",
      "visible": true,
      "bbox_pixels": [120, 80, 340, 200],
      "label": "Rectangular duct on ceiling",
      "confidence": 0.9
    }},
    {{
      "category": "ElectricalPanel",
      "visible": false,
      "bbox_pixels": null,
      "label": null,
      "confidence": 0.0
    }}
  ]
}}

Image dimensions will be provided separately. Base all pixel coordinates on the actual image size.
"""

# ── Main function ────────────────────────────────────────────────────────────

def query_and_save(image_path, as_designed_path, output_path):
    """
    Query a vision LLM with the site image + object categories.
    Falls back to a mock response if no API key is configured.
    """
    with open(as_designed_path, 'r') as f:
        as_designed = json.load(f)

    categories = list(as_designed.keys())

    result = call_llm_vision(image_path, categories)

    # Merge result with as_designed data
    as_built = {}
    for obj in result.get("objects", []):
        cat = obj["category"]
        as_built[cat] = {
            "visible": obj.get("visible", False),
            "bbox_pixels": obj.get("bbox_pixels"),
            "label": obj.get("label"),
            "confidence": obj.get("confidence", 0.0)
        }

    with open(output_path, 'w') as f:
        json.dump(as_built, f, indent=2)

    visible_count = sum(1 for v in as_built.values() if v["visible"])
    print(f"  Found {visible_count}/{len(categories)} objects visible in image")
    return as_built


def call_llm_vision(image_path, categories):
    try:
        return _call_gemini(image_path, categories)
    except Exception as e:
        print(f"  WARNING: Gemini API call failed ({e}), using mock response")
        return _mock_response(categories)

def _call_gemini(image_path, categories):
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # free tier model

    with open(image_path, 'rb') as f:
        image_data = f.read()

    import PIL.Image
    import io
    pil_image = PIL.Image.open(io.BytesIO(image_data))

    prompt = VISION_PROMPT.format(categories=json.dumps(categories, indent=2))

    response = model.generate_content([prompt, pil_image])
    response_text = response.text.strip()

    # Strip markdown fences if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    return json.loads(response_text.strip())


def _mock_response(categories):
    """
    Mock LLM response for testing without an API key.
    Simulates finding ~half the objects with plausible pixel boxes.
    """
    import random
    random.seed(42)

    objects = []
    for i, cat in enumerate(categories[:20]):  # limit to first 20 for mock
        visible = random.random() > 0.4
        objects.append({
            "category": cat,
            "visible": visible,
            "bbox_pixels": [
                random.randint(50, 300),
                random.randint(50, 200),
                random.randint(350, 600),
                random.randint(250, 400)
            ] if visible else None,
            "label": f"Detected {cat}" if visible else None,
            "confidence": round(random.uniform(0.6, 0.95), 2) if visible else 0.0
        })

    print("  (Using MOCK vision response — set ANTHROPIC_API_KEY for real results)")
    return {"objects": objects}


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python vision_query.py image.jpg as_designed.json as_built.json")
        sys.exit(1)
    query_and_save(sys.argv[1], sys.argv[2], sys.argv[3])