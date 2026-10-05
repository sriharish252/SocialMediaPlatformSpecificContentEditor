import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel, PositiveInt

from social_editor.models import PlatformPost

CONFIG = Path(__file__).with_name("platforms.toml")


class Platform(BaseModel):
    key: str
    name: str
    max_chars: PositiveInt
    visible_chars: PositiveInt
    max_hashtags: PositiveInt
    audience: str
    tone: str
    format: str
    hashtags: str
    persona: str

    @property
    def rules(self) -> str:
        return (
            f"- Audience: {self.audience}\n"
            f"- Tone: {self.tone}\n"
            f"- Format: {self.format}\n"
            f"- Length: at most {self.max_chars} characters including hashtags. Only the first "
            f"{self.visible_chars} show before 'more', so the hook must land there.\n"
            f"- Hashtags: at most {self.max_hashtags}. {self.hashtags}"
        )


@cache
def load_platforms(path: Path = CONFIG) -> dict[str, Platform]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: Platform(key=key, **values) for key, values in data.items()}


def check_post(platform: Platform, post: PlatformPost) -> list[str]:
    """Hard-limit checks the LLM can't be trusted to do itself."""
    warnings = []
    if post.character_count > platform.max_chars:
        warnings.append(
            f"{post.character_count:,} characters is over {platform.name}'s "
            f"{platform.max_chars:,} limit."
        )
    if len(post.hashtags) > platform.max_hashtags:
        warnings.append(
            f"{len(post.hashtags)} hashtags; {platform.name} allows {platform.max_hashtags}."
        )
    return warnings
