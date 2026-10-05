import asyncio

import pytest
from langchain_tavily._utilities import TavilyExtractAPIWrapper

from social_editor import web
from social_editor.crew import build_crew, run_platforms


def test_editor_researches_hashtags_before_drafting(platforms, llm, tavily):
    tools = [web.hashtag_search()]
    [post] = asyncio.run(run_platforms("Hello", [platforms["instagram"]], llm, tools))

    assert tavily == ["logo contest hashtags"]
    assert any("#LogoDesign 12k posts" in prompt for prompt in llm.log.prompts)
    assert post.text == "Final Instagram post"


def test_only_the_draft_task_gets_the_tool(platforms, llm, tavily):
    draft, critique, rewrite = build_crew(platforms["tiktok"], llm, [web.hashtag_search()]).tasks
    assert [t.name for t in draft.tools] == ["hashtag_search"]
    assert "hashtag_search" in draft.description
    assert not critique.tools and not rewrite.tools
    assert "hashtag_search" not in build_crew(platforms["tiktok"], llm).tasks[0].description


def test_read_article_returns_trimmed_text(tavily):
    text = web.read_article(" https://example.com/post ")
    assert tavily == [["https://example.com/post"]]
    assert text.startswith("MetalBoys logo contest.")
    assert len(text) == web.MAX_ARTICLE_CHARS


def test_read_article_raises_when_nothing_is_extracted(tavily, monkeypatch):
    monkeypatch.setattr(TavilyExtractAPIWrapper, "raw_results", lambda self, **_: {"results": []})
    with pytest.raises(ValueError, match=r"Couldn't read https://example\.com"):
        web.read_article("https://example.com")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("https://example.com/a?b=1", True), (" http://x.io \n", True), ("see https://x.io", False)],
)
def test_is_url(text, expected):
    assert web.is_url(text) is expected
