# generate_safe_prompts_with_image.py
# pip install openai tqdm

import json
import time
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# 🔑 API key
client = OpenAI()

# 📁 Paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "siuo_gen.json"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_JSON = BASE_DIR / "siuo_gen_safe_aligned.json"

# 🧠 Prompt
SYSTEM_PROMPT = (
    "You are a precise text editor.\n"
    "Given an image and a question that are unsafe together:\n"
    "Your task is to minimally edit the question so that:\n"
    "- The combination becomes SAFE\n"
    "- The edited question remains consistent with the image\n"
    "Strict rules:\n"
    "- Change as few words as possible (ideally 1–3 words)\n"
    "- Do NOT rewrite the whole sentence\n"
    "- Do NOT add new concepts not grounded in the image\n"
    "- Preserve the original sentence structure\n"
    "- Keep wording as close as possible\n"
    "- Only modify the part that makes it unsafe\n"
    "Output rules:\n"
    "- Output ONLY the edited question\n"
    "- No explanation\n"
    "- No extra text"
)

# 🖼️ Encode image
def encode_image(image_path):
    ext = Path(image_path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, mime

# 🔁 Robust API call with retry
def rewrite_with_retry(image_path, question, max_retries=5):
    base64_image, mime = encode_image(image_path)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Original question:\n{question}",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime};base64,{base64_image}",
                            },
                        ],
                    },
                ],
                temperature=0.2,
            )

            return response.output[0].content[0].text.strip()

        except Exception as e:
            err_str = str(e)

            # 🚫 Rate limit handling
            if "429" in err_str:
                wait_time = 10 * (attempt + 1)
                print(f"⏳ 429 error. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            # ❌ Other errors → retry a bit then fail
            print(f"⚠️ Error: {e}")
            time.sleep(5)

    # fallback if all retries fail
    return question


def main():
    # 📥 Load JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data["data"]

    results = []

    for i, item in enumerate(tqdm(records)):
        image_path = IMAGE_DIR / item["image"]
        question = item["question"]

        safe_q = rewrite_with_retry(str(image_path), question)

        new_item = dict(item)
        new_item["safe_question"] = safe_q
        results.append(new_item)

        print(f"[{i+1}/{len(records)}] done")

        # ✅ 22 second delay after SUCCESS
        time.sleep(22)

    # 📤 Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()