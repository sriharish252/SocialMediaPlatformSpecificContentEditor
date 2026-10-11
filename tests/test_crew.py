import asyncio

from social_editor.crew import build_crew, run_platforms
from social_editor.models import Critique, Draft, PlatformPost


def test_crew_is_editor_critic_rewrite(platforms, llm):
    crew = build_crew(platforms["tiktok"], llm)
    draft, critique, rewrite = crew.tasks
    editor, critic = crew.agents

    assert [t.agent for t in crew.tasks] == [editor, critic, editor]
    assert editor.role == "TikTok Content Editor"
    assert critique.context == [draft]
    assert rewrite.context == [draft, critique]
    assert [t.output_pydantic for t in crew.tasks] == [Draft, Critique, Draft]
    assert all(agent.llm is llm for agent in crew.agents)


def test_platform_rules_reach_the_prompts(platforms, llm):
    linkedin = platforms["linkedin"]
    draft, critique, rewrite = build_crew(linkedin, llm).tasks
    for task in (draft, critique, rewrite):
        assert linkedin.rules in task.description
    assert "{content}" in draft.description


def test_run_returns_one_structured_post_per_platform(platforms, llm):
    chosen = [platforms["instagram"], platforms["linkedin"]]
    posts = asyncio.run(run_platforms("New logo contest, $20k prize", chosen, llm))

    assert [p.platform for p in posts] == ["Instagram", "LinkedIn"]
    instagram = posts[0]
    assert isinstance(instagram, PlatformPost)
    assert instagram.text == "Final Instagram post"
    assert instagram.hashtags == ["#Logo", "#Design"]
    assert instagram.critique == ["Lead with the prize", "Add a deadline"]
    assert instagram.warnings == []
    assert len(llm.log.prompts) == 6


def test_platforms_run_concurrently(platforms, llm):
    asyncio.run(run_platforms("Hello", list(platforms.values()), llm))
    assert llm.log.peak_in_flight == 3


def test_content_with_braces_is_passed_through(platforms, llm):
    asyncio.run(run_platforms("Use code {SAVE20} at checkout", [platforms["tiktok"]], llm))
    assert "Use code {SAVE20} at checkout" in llm.log.prompts[0]


def test_one_failing_platform_does_not_sink_the_rest(platforms, llm):
    llm.fail_for = "TikTok"
    results = asyncio.run(run_platforms("Hello", list(platforms.values()), llm))
    instagram, tiktok, linkedin = results
    assert isinstance(tiktok, Exception)
    assert isinstance(instagram, PlatformPost)
    assert isinstance(linkedin, PlatformPost)
