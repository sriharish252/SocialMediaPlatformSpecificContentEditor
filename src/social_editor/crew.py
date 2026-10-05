"""One crew per platform: the editor drafts, the critic reviews, the editor rewrites.

Prompts here are platform-agnostic; everything platform-specific comes from platforms.toml.
"""

import asyncio

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, ValidationError

from social_editor.models import Critique, Draft, PlatformPost
from social_editor.platforms import Platform, check_post

CRITIC_BACKSTORY = """You have a keen eye for grammar, clarity and flow, you know who each \
platform's audience is, and you judge whether a post will actually land with them. Your \
feedback is clear, concise and actionable, never harsh for the sake of it."""

FACTS = "Keep every fact from the original (names, dates, prices, links) and never invent new ones."


def build_crew(platform: Platform, llm: BaseLLM) -> Crew:
    editor = Agent(
        role=f"{platform.name} Content Editor",
        goal=f"Turn the user's content into a {platform.name} post its audience will engage with",
        backstory=platform.persona,
        llm=llm,
    )
    critic = Agent(
        role="Content Critic",
        goal="Find the changes that will most improve a post for its platform",
        backstory=CRITIC_BACKSTORY,
        llm=llm,
    )

    draft = Task(
        description=(
            f"Rewrite the content below as a {platform.name} post.\n\n"
            f"Platform rules:\n{platform.rules}\n\n{FACTS}\n\n"
            "Content:\n{content}"
        ),
        expected_output=f"A {platform.name} post: the text and, separately, its hashtags.",
        agent=editor,
        output_pydantic=Draft,
    )
    critique = Task(
        description=(
            f"Review the editor's {platform.name} draft against the original content and these "
            f"rules:\n{platform.rules}\n\n"
            "Check platform fit, the opening hook, clarity, spelling and grammar, the call to "
            "action, hashtag choice, and that no facts were changed or invented.\n\n"
            "Original content:\n{content}"
        ),
        expected_output="Three to five specific, actionable points, most important first.",
        agent=critic,
        context=[draft],
        output_pydantic=Critique,
    )
    rewrite = Task(
        description=(
            f"Revise your {platform.name} draft using the critic's feedback. Apply the points "
            "that improve the post and skip any that would break the rules or the facts.\n\n"
            f"Platform rules:\n{platform.rules}\n\n{FACTS}"
        ),
        expected_output=f"The final {platform.name} post: the text and, separately, its hashtags.",
        agent=editor,
        context=[draft, critique],
        output_pydantic=Draft,
    )

    return Crew(
        agents=[editor, critic], tasks=[draft, critique, rewrite], process=Process.sequential
    )


def parse[M: BaseModel](output: TaskOutput, model: type[M]) -> M | None:
    """CrewAI fills .pydantic when the LLM honoured the schema; fall back to parsing raw JSON."""
    if isinstance(output.pydantic, model):
        return output.pydantic
    try:
        return model.model_validate_json(output.raw)
    except ValidationError:
        return None


def to_post(platform: Platform, tasks: list[TaskOutput]) -> PlatformPost:
    _, critique_out, final_out = tasks
    final = parse(final_out, Draft) or Draft(text=final_out.raw, hashtags=[])
    critique = parse(critique_out, Critique) or Critique(key_points=[critique_out.raw])
    post = PlatformPost(
        platform=platform.name,
        text=final.text,
        hashtags=final.hashtags,
        critique=critique.key_points,
    )
    post.warnings = check_post(platform, post)
    return post


async def run_platforms(
    content: str, platforms: list[Platform], llm: BaseLLM
) -> list[PlatformPost | Exception]:
    """Run every platform's crew concurrently. A failure in one doesn't sink the others."""

    async def run(platform: Platform) -> PlatformPost:
        result = await build_crew(platform, llm).akickoff(inputs={"content": content})
        return to_post(platform, result.tasks_output)

    return await asyncio.gather(*(run(p) for p in platforms), return_exceptions=True)
