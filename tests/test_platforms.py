import pytest

from social_editor.models import PlatformPost
from social_editor.platforms import check_post


def post(text: str, hashtags: int = 0) -> PlatformPost:
    tags = [f"#tag{i}" for i in range(hashtags)]
    return PlatformPost(platform="x", text=text, hashtags=tags, critique=[])


@pytest.mark.parametrize(
    ("key", "max_chars", "max_hashtags"),
    [("instagram", 2200, 5), ("tiktok", 4000, 5), ("linkedin", 3000, 5)],
)
def test_platform_limits(platforms, key, max_chars, max_hashtags):
    platform = platforms[key]
    assert (platform.max_chars, platform.max_hashtags) == (max_chars, max_hashtags)
    assert platform.visible_chars < platform.max_chars


def test_rules_carry_the_limits(platforms):
    for platform in platforms.values():
        assert f"at most {platform.max_chars} characters" in platform.rules
        assert f"at most {platform.max_hashtags}" in platform.rules
        assert platform.tone in platform.rules


def test_post_within_limits_has_no_warnings(platforms):
    assert check_post(platforms["instagram"], post("Short and sweet", hashtags=5)) == []


def test_over_length_post_is_flagged(platforms):
    [warning] = check_post(platforms["instagram"], post("a" * 2201))
    assert "2,201 characters" in warning


def test_hashtags_count_toward_length(platforms):
    assert check_post(platforms["instagram"], post("a" * 2190, hashtags=2))


def test_too_many_hashtags_is_flagged(platforms):
    [warning] = check_post(platforms["instagram"], post("Hi", hashtags=6))
    assert "allows 5" in warning
