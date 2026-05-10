import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


MODEL = "gpt-4.1-mini"

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_JSON = BASE_DIR / "siuo_gen.json"
LABELS_JSON = BASE_DIR / "siuo_gen_word_labels.json"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_JSON = BASE_DIR / "siuo_gen_word_labels_fixed.json"

SUCCESS_SLEEP_SECONDS = 22
RATE_LIMIT_SLEEP_SECONDS = 60


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
    client = OpenAI(api_key = "")

    if not ORIGINAL_JSON.exists():
        raise FileNotFoundError(f"Missing original JSON file: {ORIGINAL_JSON}")
    if not LABELS_JSON.exists():
        raise FileNotFoundError(f"Missing labels JSON file: {LABELS_JSON}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {IMAGE_DIR}")

    with ORIGINAL_JSON.open("r", encoding="utf-8") as f:
        original_data = json.load(f)

    with LABELS_JSON.open("r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    if not isinstance(original_data, list):
        raise ValueError("Expected siuo_gen.json to be a list.")
    if not isinstance(labeled_data, list):
        raise ValueError("Expected siuo_gen_word_labels.json to be a list.")

    original_by_id = {}
    for item in original_data:
        qid = item.get("question_id")
        if qid is not None:
            original_by_id[qid] = item

    fixed_results: List[Dict[str, Any]] = []

    for i, row in enumerate(labeled_data, start=1):
        if "error" not in row:
            fixed_results.append(row)
            continue

        qid = row.get("question_id")
        source_item = original_by_id.get(qid, row)

        print(f"[{i}/{len(labeled_data)}] Reprocessing question_id={qid}")

        success = False
        last_error = None

        for attempt in range(6):
            try:
                new_row = label_pair(client, source_item, IMAGE_DIR)
                fixed_results.append(new_row)
                success = True
                time.sleep(SUCCESS_SLEEP_SECONDS)
                break
            except Exception as e:
                last_error = str(e)
                msg = last_error.lower()

                if "rate_limit" in msg or "429" in msg or "requests per min" in msg:
                    wait_seconds = RATE_LIMIT_SLEEP_SECONDS + (attempt * 10)
                    print(f"Rate limit hit for question_id={qid}. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue

                fixed_results.append(
                    {
                        "question_id": source_item.get("question_id"),
                        "image": source_item.get("image"),
                        "question": source_item.get("question"),
                        "category": source_item.get("category"),
                        "error": last_error,
                    }
                )
                success = True
                break

        if not success:
            fixed_results.append(
                {
                    "question_id": source_item.get("question_id"),
                    "image": source_item.get("image"),
                    "question": source_item.get("question"),
                    "category": source_item.get("category"),
                    "error": last_error or "Failed after multiple retries.",
                }
            )

        with OUTPUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(fixed_results, f, indent=2, ensure_ascii=False)

    print(f"Saved fixed labels to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()