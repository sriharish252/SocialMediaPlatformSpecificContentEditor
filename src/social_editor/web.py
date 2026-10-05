"""The app's two LangChain integrations, both from langchain-tavily. They switch on when
TAVILY_API_KEY is set; without it the app rewrites pasted text only.

- read_article: TavilyExtract turns a pasted link into clean article text for the crews.
- hashtag_search: TavilySearch, handed to the editor as a CrewAI tool so it picks hashtags people
  are using this month rather than ones it remembers.
"""

import os
import re

from crewai.tools.base_tool import Tool
from langchain_tavily import TavilyExtract, TavilySearch
from pydantic import BaseModel, Field

MAX_ARTICLE_CHARS = 20_000

_URL = re.compile(r"https?://\S+")


def enabled() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def is_url(text: str) -> bool:
    return bool(_URL.fullmatch(text.strip()))


def read_article(url: str) -> str:
    result = TavilyExtract(format="text").invoke({"urls": [url.strip()]})
    # The tool reports failures in its result (a message or {"error": ...}) rather than raising.
    if isinstance(result, dict) and result.get("results"):
        return result["results"][0]["raw_content"][:MAX_ARTICLE_CHARS]
    detail = result.get("error", "no readable text") if isinstance(result, dict) else result
    raise ValueError(f"Couldn't read {url}: {detail}")


class _Query(BaseModel):
    query: str = Field(description="What to search for, e.g. 'road bike instagram hashtags'")


def hashtag_search() -> Tool:
    """TavilySearch over the past month, wrapped as a CrewAI tool.

    CrewAI only auto-converts LangChain tools that expose a plain `func`, which TavilySearch
    doesn't, so this wraps its invoke() with a one-field schema the agent can't misuse.
    """
    search = TavilySearch(max_results=5, time_range="month")
    return Tool(
        name="hashtag_search",
        description=(
            "Search the web (past month) to see which hashtags people currently use for a topic "
            "on a platform."
        ),
        args_schema=_Query,
        func=lambda query: search.invoke({"query": query}),
    )
