"""从回复中的标点提取用于候选评分的语气线索。"""

import re


MAX_PUNCTUATION_BONUS = 0.06

_NON_SPEECH = re.compile(
    r"```[\s\S]*?(?:```|$)|`[^`\n]*`|https?://\S+"
    r'|“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"'
)


def punctuation_emotion_weights(
    reply: str, emotion_priors: list[str] | None = None,
) -> dict[str, float]:
    """返回有上限的分类加分；情绪先验用于消解感叹号和波浪号的歧义。"""
    text = _NON_SPEECH.sub(" ", reply or "")
    # Markdown 引用中的标点属于被引用者。
    text = re.sub(r"(?m)^\s*>[^\n]*", " ", text)
    if not any(char.isalnum() for char in text):
        return {}
    priors = set(emotion_priors or [])
    weights: dict[str, float] = {}

    def boost(values: dict[str, float], *, contextual: bool = False) -> None:
        for category, value in values.items():
            if contextual and priors and category not in priors:
                continue
            weights[category] = min(
                MAX_PUNCTUATION_BONUS, max(weights.get(category, 0.0), value),
            )

    questions = re.findall(r"[?？]+", text)
    if questions:
        repeated = any(len(run) >= 2 for run in questions)
        boost({"confused": 0.06 if repeated else 0.03})
    if re.search(r"(?:[?？]+[!！]+|[!！]+[?？]+)", text):
        boost({"surprised": 0.06, "confused": 0.03})

    exclamations = re.findall(r"[!！]+", text)
    if exclamations:
        strong = any(len(run) >= 2 for run in exclamations)
        bonus = 0.05 if strong else 0.025
        boost({"happy": bonus, "excitement": bonus, "surprised": bonus}, contextual=True)
        if "angry" in priors:
            boost({"angry": bonus})

    if re.search(r"\.{3,}|…+", text):
        boost({"dumb": 0.04, "sigh": 0.03, "embarrassed": 0.02})
    if re.search(r"[~～]+", text):
        boost({"happy": 0.025, "love": 0.03, "shy": 0.02}, contextual=True)
    return weights
