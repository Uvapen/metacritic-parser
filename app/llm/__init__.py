"""Клиент LLM для саммари по отзывам."""

from app.llm.client import LLMClient, LLMResult
from app.llm.prompts import (
    CRITIC_SUMMARY_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    USER_SUMMARY_SYSTEM_PROMPT,
    build_critic_summary_prompt,
    build_summary_prompt,
    build_user_summary_prompt,
)

__all__ = [
    "LLMClient",
    "LLMResult",
    "SUMMARY_SYSTEM_PROMPT",
    "CRITIC_SUMMARY_SYSTEM_PROMPT",
    "USER_SUMMARY_SYSTEM_PROMPT",
    "build_summary_prompt",
    "build_critic_summary_prompt",
    "build_user_summary_prompt",
]
