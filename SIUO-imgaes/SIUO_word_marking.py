import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


MODEL = "gpt-4.1-mini"

# This script, siuo_gen.json, and the images/ folder should all be in the same directory.
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "siuo_gen.json"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_JSON = BASE_DIR / "siuo_gen_word_labels.json"


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix in (".jpg", ".jpeg"):
            mime_type = "image/jpeg"
        elif suffix == ".webp":
            mime_type = "image/webp"
        else:
            raise ValueError(f"Unsupported image type: {image_path.suffix}")

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Tries to parse a JSON object from the model output.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def mark_word(question: str, word_index: int | None) -> str:
    words = question.split()
    if word_index is None or word_index < 0 or word_index >= len(words):
        return question

    marked = []
    for i, w in enumerate(words):
        if i == word_index:
            marked.append(f"[{w}]")
        else:
            marked.append(w)
    return " ".join(marked)


def label_pair(client: OpenAI, item: Dict[str, Any], image_dir: Path) -> Dict[str, Any]:
    image_name = item["image"]
    question = item["question"]
    image_path = image_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    data_url = image_to_data_url(image_path)

    system_prompt = (
        "You are labeling a multimodal safety dataset for research. "
        "Given an image and a text prompt, choose exactly ONE word in the prompt "
        "that is most strongly associated with the unsafe implication suggested by the image. "
        "Return only valid JSON."
    )

    user_prompt = f"""
Question ID: {item.get('question_id')}
Category: {item.get('category', '')}
Prompt: {question}

Return JSON with exactly these fields:
- unsafe_word: string or null
- word_index: integer or null   (0-based index into the prompt words split on spaces)
- confidence: number from 0 to 1
- note: short string

Rules:
- Pick the single best word, not a phrase or span.
- If no single word is clearly best, choose the most relevant word anyway.
- Do not include any extra keys.
""".strip()

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            },
        ],
    )

    parsed = extract_json_object(response.output_text)

    word_index = parsed.get("word_index")
    parsed["question_id"] = item.get("question_id")
    parsed["image"] = image_name
    parsed["question"] = question
    parsed["category"] = item.get("category", "")
    parsed["marked_prompt"] = mark_word(question, word_index)

    return parsed


def main() -> None:
    client = OpenAI()

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing JSON file: {INPUT_JSON}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {IMAGE_DIR}")

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of examples.")

    results: List[Dict[str, Any]] = []

    for i, item in enumerate(data, start=1):
        try:
            print(f"[{i}/{len(data)}] Processing question_id={item.get('question_id')}")
            results.append(label_pair(client, item, IMAGE_DIR))
        except Exception as e:
            results.append(
                {
                    "question_id": item.get("question_id"),
                    "image": item.get("image"),
                    "question": item.get("question"),
                    "category": item.get("category"),
                    "error": str(e),
                }
            )

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved labels to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()