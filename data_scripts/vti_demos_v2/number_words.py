"""Number-word helpers and FALSE_COUNT boundaries for demos_v2.

Named ``number_words`` (not ``numbers``) so it does not shadow the Python
stdlib ``numbers`` module — Anthropic/Pydantic import that during client init.
"""

from __future__ import annotations

from . import config

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def number_to_word(n: int) -> str:
    """Spell out a non-negative integer (sufficient for counts ≤ 20, and beyond)."""
    if n < 0:
        raise ValueError(f"number_to_word expects n >= 0, got {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _TENS[tens]
        return f"{_TENS[tens]}-{_ONES[ones]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return f"{_ONES[hundreds]} hundred"
        return f"{_ONES[hundreds]} hundred {number_to_word(rest)}"
    raise ValueError(f"number_to_word not implemented for n={n}")


def word_to_number(word: str) -> int | None:
    """Inverse of number_to_word for the small vocabulary we emit (0–20)."""
    w = word.strip().lower().replace("-", " ")
    mapping = {number_to_word(i): i for i in range(21)}
    return mapping.get(w) or mapping.get(w.replace(" ", "-"))


def false_count(n: int) -> int:
    return int(config.FALSE_COUNT(n))


def false_count_at_most(n: int) -> int:
    """False upper-bound for ``at_most`` mode: max(1, min(n-2, (2*n)//3))."""
    return max(1, min(n - 2, (2 * n) // 3))
