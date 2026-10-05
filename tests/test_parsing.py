from crewai.tasks.task_output import TaskOutput

from social_editor.crew import parse, to_post
from social_editor.models import Critique, Draft, PlatformPost


def output(raw: str, pydantic=None) -> TaskOutput:
    return TaskOutput(description="d", raw=raw, agent="a", pydantic=pydantic)


def test_hashtags_are_normalised():
    draft = Draft(text="t", hashtags=["#AI", "ai", " Road Bikes ", "", "#"])
    assert draft.hashtags == ["#AI", "#RoadBikes"]


def test_full_text_and_character_count():
    post = PlatformPost(platform="x", text=" Hello ", hashtags=["#a", "#b"], critique=[])
    assert post.full_text == "Hello\n\n#a #b"
    assert post.character_count == len("Hello\n\n#a #b")


def test_full_text_without_hashtags_has_no_trailing_gap():
    post = PlatformPost(platform="x", text="Hello", hashtags=[], critique=[])
    assert post.full_text == "Hello"


def test_parse_prefers_crewai_pydantic():
    draft = Draft(text="t", hashtags=[])
    assert parse(output("ignored", pydantic=draft), Draft) is draft


def test_parse_falls_back_to_raw_json():
    parsed = parse(output('{"key_points": ["Be bolder"]}'), Critique)
    assert parsed == Critique(key_points=["Be bolder"])


def test_parse_returns_none_for_prose():
    assert parse(output("Here is your post!"), Draft) is None


def test_to_post_survives_unstructured_output(platforms):
    tasks = [output("draft"), output("Needs a hook."), output("Plain final post")]
    post = to_post(platforms["linkedin"], tasks)
    assert (post.text, post.hashtags, post.critique) == ("Plain final post", [], ["Needs a hook."])
