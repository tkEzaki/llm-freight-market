"""LLM backend layer for RQ1.

Public surface:
    from src.llm import make_backend, available_backends, DEFAULT_MODELS
    from src.llm.base import LLMBackend, LLMResponse, LLMRequestError
    from src.llm.cache import ResponseCache
    from src.llm.prompts import build_shipper_prompt
"""
from .registry import DEFAULT_MODELS, available_backends, make_backend  # noqa: F401
