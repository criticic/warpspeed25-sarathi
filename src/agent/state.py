"""State for the multi-agent system."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class OverallState(TypedDict):
    """The overall state of the multi-agent system."""

    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    research_required: bool  # Keep for backward compatibility
    agent_type: str  # New field for agent routing
    generate_audio: bool
    audio_output: str | None
    id: str
    detected_language_code: str | None
    detected_script_code: str | None
    needs_translation: bool
    needs_translation_back: bool  # Whether response needs translation back to original language
    source_language: str | None  # For translation agent
    target_language: str | None  # For translation agent
    text_to_convert: str | None  # For translation/TTS agents
    translation_result: str | None  # For storing translation results
    english_answer: str | None  # Store English answer for potential translation
    output_script: str | None  # Preferred output script for translation


class ReflectionState(TypedDict):
    """The state of the reflection node."""

    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    """A search query."""

    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    """The state of the query generation node."""

    search_query: list[Query]


class WebSearchState(TypedDict):
    """The state of the web search node."""

    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    """The output of the search state."""

    running_summary: str = field(default=None)  # Final report
