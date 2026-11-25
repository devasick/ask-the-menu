from app.config import Settings
from app.llm.base import LLMClient


def make_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_backend == "fake":
        from app.llm.fake import FakeLLMClient

        return FakeLLMClient()
    from app.llm.providers import DirectLLMClient

    return DirectLLMClient(settings)
