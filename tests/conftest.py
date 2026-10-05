import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field

os.environ.update(
    CREWAI_DISABLE_TELEMETRY="true", CREWAI_TRACING_ENABLED="false", OTEL_SDK_DISABLED="true"
)

import pytest
from crewai.llms.base_llm import BaseLLM
from langchain_tavily._utilities import TavilyExtractAPIWrapper, TavilySearchAPIWrapper
from pydantic import Field

from social_editor.models import Critique, Draft
from social_editor.platforms import load_platforms


@dataclass
class CallLog:
    prompts: list[str] = field(default_factory=list)
    in_flight: int = 0
    peak_in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, messages) -> None:
        with self.lock:
            self.prompts.append(json.dumps(messages, ensure_ascii=False))
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        time.sleep(0.05)  # long enough for concurrent crews to overlap
        with self.lock:
            self.in_flight -= 1


class FakeLLM(BaseLLM):
    """Answers each structured-output request with canned JSON; no network.

    CrewAI shallow-copies an agent's LLM at kickoff, so all copies share one CallLog.
    """

    fail_for: str | None = None
    log: CallLog = Field(default_factory=CallLog)

    def call(self, messages, from_task=None, from_agent=None, response_model=None, **_):
        self.log.record(messages)
        platform = from_agent.role.removesuffix(" Content Editor")
        if platform == self.fail_for:
            raise RuntimeError("rate limited")
        if response_model is Critique:
            return json.dumps({"key_points": ["Lead with the prize", "Add a deadline"]})

        stage = "Final" if from_task.description.startswith("Revise") else "Draft"
        draft = json.dumps({"text": f"{stage} {platform} post", "hashtags": ["Logo", "#Design"]})
        if response_model is Draft:
            return draft
        if from_task.tools:  # CrewAI drops the schema while tools are in play (ReAct format)
            if not any(m["role"] == "assistant" for m in messages):  # no search yet
                return (
                    "Thought: Check which hashtags are in use\nAction: hashtag_search\n"
                    'Action Input: {"query": "logo contest hashtags"}'
                )
            return f"Thought: I now know the final answer\nFinal Answer: {draft}"
        raise AssertionError(f"Unexpected unstructured call: {from_task.description[:40]}")

    async def acall(self, messages, **kwargs):
        return await asyncio.to_thread(self.call, messages, **kwargs)

    def supports_function_calling(self) -> bool:
        return False


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM(model="fake")


@pytest.fixture
def platforms():
    return load_platforms()


@pytest.fixture
def tavily(monkeypatch):
    """Real langchain-tavily tools with only the HTTP call replaced."""
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    calls = []

    def search(self, query, **_):
        calls.append(query)
        return {"results": [{"title": "Logo contests", "content": "#LogoDesign 12k posts"}]}

    def extract(self, urls, **_):
        calls.append(urls)
        return {"results": [{"url": urls[0], "raw_content": "MetalBoys logo contest. " * 2000}]}

    monkeypatch.setattr(TavilySearchAPIWrapper, "raw_results", search)
    monkeypatch.setattr(TavilyExtractAPIWrapper, "raw_results", extract)
    return calls
