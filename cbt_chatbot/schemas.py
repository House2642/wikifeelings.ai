"""Pydantic schemas for CBT therapy chatbot."""

from pydantic import BaseModel, Field
from typing import Optional, Any


class Extraction(BaseModel):
    """What the LLM extracts from each user message."""

    # What happened? (external event or circumstance)
    situation: Optional[str] = None

    # What are they thinking? (automatic thought or belief)
    thought: Optional[str] = None

    # Which cognitive distortions are present?
    distortions: list[str] = Field(
        default=[],
        description="From: catastrophizing, mind_reading, fortune_telling, should_statements, all_or_nothing, personalization, overgeneralization, labeling"
    )

    # How do they feel?
    feeling: Optional[str] = Field(
        default=None,
        description="From: anxious, sad, angry, ashamed, guilty, overwhelmed"
    )
    intensity: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="How intense is the feeling 0-10"
    )

    # What are they doing (or not doing)?
    behaviors: list[str] = Field(
        default=[],
        description="From: avoidance, withdrawal, rumination, reassurance_seeking, procrastination, substance_use"
    )

    # Safety - CRITICAL
    crisis_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="0=none, 1=passive thoughts, 2=active ideation, 3=plan, 4=intent/means"
    )


class CBTState(BaseModel):
    """The cognitive model we're building about the user."""

    # Conversation history
    messages: list[dict[str, Any]] = Field(default=[])

    # The cognitive model
    situation: Optional[str] = None
    thought: Optional[str] = None
    feeling: Optional[str] = None
    intensity: Optional[int] = None
    distortions: list[str] = Field(default=[])
    behaviors: list[str] = Field(default=[])

    # What do we have? (for policy decisions)
    has_situation: bool = False
    has_thought: bool = False
    has_feeling: bool = False

    # Safety
    crisis_level: int = 0

    # Current action (set by policy)
    current_action: Optional[str] = None
