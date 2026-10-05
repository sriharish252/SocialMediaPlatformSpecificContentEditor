from pydantic import BaseModel, Field, computed_field, field_validator


class Draft(BaseModel):
    """What the editor agent returns."""

    text: str = Field(description="The post body, without hashtags")
    hashtags: list[str] = Field(description="Hashtags for the post, each starting with #")

    @field_validator("hashtags")
    @classmethod
    def normalise_hashtags(cls, tags: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for tag in tags:
            word = "".join(tag.split()).lstrip("#")
            if word:
                seen.setdefault(word.lower(), f"#{word}")
        return list(seen.values())


class Critique(BaseModel):
    """What the critic agent returns."""

    key_points: list[str] = Field(
        description="Three to five specific, actionable points, most important first"
    )


class PlatformPost(BaseModel):
    platform: str
    text: str
    hashtags: list[str]
    critique: list[str]
    warnings: list[str] = []

    @computed_field
    @property
    def full_text(self) -> str:
        return "\n\n".join(filter(None, [self.text.strip(), " ".join(self.hashtags)]))

    @computed_field
    @property
    def character_count(self) -> int:
        return len(self.full_text)
