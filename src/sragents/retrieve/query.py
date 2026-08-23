"""Canonical retrieval query construction shared by retrieval and reranking."""

from sragents.prompts import build_prompt


def build_retrieval_query(instance: dict) -> str:
    """Render exactly the text indexed by every retrieval baseline."""
    system, user = build_prompt(instance)
    parts = [user]
    if system:
        parts.append(system)
    if instance.get("dataset") == "toolqa":
        from sragents.toolqa.fewshots import TOOLQA_EXAMPLES
        parts.append(TOOLQA_EXAMPLES)
    return "\n".join(parts)
