"""长期记忆：跨会话保存用户画像/偏好/事实，JSON 持久化 + LLM 抽取。

短期记忆由 Agent 状态里的 messages（对话历史）承担；这里实现跨会话的长期记忆。
"""
from __future__ import annotations

import json
from typing import Any

from ..config import settings
from ..llm import llm

_EXTRACT_PROMPT = (
    "你是用户画像记忆抽取器。根据对话内容，更新用户的长期记忆。\n"
    "记忆结构（JSON）：{\"facts\": [\"事实\"], \"preferences\": {\"键\": \"值\"}, \"notes\": [\"备注\"]}。\n"
    "规则：只保留明确、稳定、用户主动表达的信息；不要记录单次临时的诉求；不要杜撰。\n"
    "请输出合并后的完整记忆 JSON。"
)


class Memory:
    """单个用户的长期记忆。"""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self.path = settings.memory_dir / f"{user_id}.json"

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"facts": [], "preferences": {}, "notes": []}

    def save(self, memory: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def extract(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """用 LLM 从对话中抽取并合并长期记忆。未配置 LLM 时原样返回。"""
        existing = self.load()
        if not settings.llm_ready:
            return existing
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content', '')[:500]}" for m in messages if m.get("content")
        )
        user_msg = (
            f"现有记忆：\n{json.dumps(existing, ensure_ascii=False, indent=2)}\n\n"
            f"对话内容：\n{transcript}"
        )
        try:
            updated = llm.generate_json(
                [
                    {"role": "system", "content": _EXTRACT_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
            )
            if isinstance(updated, dict):
                return updated
        except Exception:
            pass
        return existing
