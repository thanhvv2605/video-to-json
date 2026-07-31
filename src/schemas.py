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
    brand: Optional[str] = Field(description="Brand if identifiable, else null")
    price_text: Optional[str] = Field(
        description="Price as it appears on screen or is spoken, e.g. '$29.99'. "
        "Never invent a price."
    )
    attributes: List[str] = Field(
        description="Concrete attributes stated in the video: colour, size, "
        "material, quantity. Claims only — no inference."
    )
    first_seen_sec: float = Field(description="Timestamp of first appearance")


class Scene(BaseModel):
    start_sec: float
    end_sec: float
    description: str = Field(description="One sentence, factual, no adjectives")
    on_screen_text: List[str] = Field(description="Text legible in the frame")


class VideoInsight(BaseModel):
    """Top-level object returned for one video."""

    language: str = Field(description="BCP-47 code of the spoken language")
    summary: str = Field(description="Two sentences, factual")
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
