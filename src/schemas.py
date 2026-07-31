"""Output contract.

The whole point of this pipeline is that the model returns a *validated object*,
not free-form text that has to be parsed with regex. These Pydantic models are
passed straight to the OpenAI structured-outputs API as the response schema, so
a malformed response is impossible by construction — and a semantically wrong
one is caught by the validators below.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Product(BaseModel):
    name: str = Field(description="Product name exactly as shown or spoken")
    brand: Optional[str] = Field(
        description="The commercial label the product is sold under — the "
        "manufacturer, seller, or store brand. NOT the author of a book, the "
        "artist of a record, the director of a film, or any other creator: "
        "those belong in `creators`. Null if no brand is stated."
    )
    creators: List[str] = Field(
        description="People credited for making the work rather than selling "
        "it: authors, artists, designers, performers. Empty for ordinary goods "
        "with no named creator."
    )
    price_text: Optional[str] = Field(
        description="Price as it appears on screen or is spoken, e.g. '$29.99'. "
        "Never invent a price."
    )
    attributes: List[str] = Field(
        description="Objective properties stated in the video: colour, size, "
        "material, quantity, capacity, compatibility, page count, duration. "
        "Test before including: could two people disagree about whether it is "
        "true? If yes, it is marketing copy, not an attribute — leave it out. "
        "'12-hour battery', 'waterproof', '320 pages' qualify. 'Modern design', "
        "'high quality', 'perfect choice' do not; those belong in `hooks`. "
        "Return an empty list rather than padding it."
    )
    first_seen_sec: float = Field(description="Timestamp of first appearance")


class Scene(BaseModel):
    start_sec: float
    end_sec: float
    description: str = Field(
        description="One sentence, factual, no adjectives. Written in the "
        "output language named in the system instructions."
    )
    on_screen_text: List[str] = Field(description="Text legible in the frame")


class VideoInsight(BaseModel):
    """Top-level object returned for one video."""

    language: str = Field(description="BCP-47 code of the spoken language")
    summary: str = Field(
        description="Two sentences, factual. Written in the output language "
        "named in the system instructions — not English by default."
    )
    content_type: Literal[
        "product_demo", "unboxing", "review", "tutorial", "ad", "vlog", "other"
    ]
    products: List[Product]
    scenes: List[Scene]
    hooks: List[str] = Field(
        description="Verbatim lines used as attention hooks, for ad reuse"
    )
    transcript: str
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-reported confidence. Below 0.6 the caller should "
        "route the video to human review rather than trust the fields.",
    )
