from dataclasses import dataclass
import re


@dataclass
class TaskResult:
    label: str
    method: str
    confidence: float
    raw: str = ""


KEYWORDS = {
    "SOH": ["寿命", "rul", "capacity", "退化", "健康"],
    "FD": ["故障", "fault", "短路", "过充"],
    "AD": ["异常", "anomaly", "outlier", "变点"]
}


def rule_router(text: str):

    text = text.lower()

    scores = {k: 0 for k in KEYWORDS}

    for k, kws in KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[k] += 1

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return None, 0.0, scores

    return best, float(scores[best]), scores


class TaskRouter:

    def __init__(self, llm=None):
        self.llm = llm

    def route(self, text: str, use_llm=True):

        label, score, raw = rule_router(text)

        if label:
            return TaskResult(label, "rule", score, str(raw))

        if use_llm and self.llm is not None:
            return self.llm_route(text)

        return TaskResult("SOH", "default", 0.1, str(raw))