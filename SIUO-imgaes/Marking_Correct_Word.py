import json
import re
from pathlib import Path


# Input/output files
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "siuo_gen_word_labels.json"
OUTPUT_JSON = BASE_DIR / "siuo_gen_word_labels_corrected.json"


def clean_word(word: str) -> str:
    """
    Normalize words for matching.
    Removes punctuation and lowercases.
    """
    return re.sub(r"[^\w]", "", word).lower()


def find_word_index(question: str, unsafe_word: str):
    """
    Finds the first matching word index in the question.
    """
    words = question.split()
    target = clean_word(unsafe_word)

    for i, w in enumerate(words):
        if clean_word(w) == target:
            return i

    return None


def mark_word(question: str, word_index):
    """
    Surrounds the selected word with brackets.
    """
    words = question.split()

    if word_index is None:
        return question

    if word_index < 0 or word_index >= len(words):
        return question

    words[word_index] = f"[{words[word_index]}]"
    return " ".join(words)


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSON}")

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    corrected = []

    for item in data:
        # Skip entries with errors
        if "error" in item:
            corrected.append(item)
            continue

        question = item.get("question", "")
        unsafe_word = item.get("unsafe_word")

        if unsafe_word is None:
            corrected.append(item)
            continue

        # Recompute correct index
        correct_index = find_word_index(question, unsafe_word)

        # Update fields
        item["word_index"] = correct_index
        item["marked_prompt"] = mark_word(question, correct_index)

        corrected.append(item)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)

    print(f"Saved corrected file to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()