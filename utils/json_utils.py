# -----------------------------
# SAFE JSON LOADER
# -----------------------------
import json
import re


def safe_json_loads(text):
    if not text or not text.strip():
        return None

    text = text.strip()

    # remove markdown code fences if present
    text = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    