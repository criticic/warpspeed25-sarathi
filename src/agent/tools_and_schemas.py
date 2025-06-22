"""Pydantic models for the multi-agent system."""

from typing import List

from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    """A list of search queries and a rationale."""

    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    """A reflection on the research process."""

    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class AgentRouting(BaseModel):
    """Routing decision for the user's query to determine which agent to use."""

    agent_type: str = Field(
        description="The type of agent needed: 'research', 'translation', 'text_to_speech', 'direct_answer', 'calendar', 'mailbox'"
    )
    answer: str = Field(
        description="The direct answer to the user's query if agent_type is 'direct_answer', otherwise empty string."
    )
    generate_audio: bool = Field(
        description="Whether the user wants an audio response.", default=False
    )
    source_language: str = Field(
        description="Source language for translation (if agent_type is 'translation')", default=""
    )
    target_language: str = Field(
        description="Target language for translation (if agent_type is 'translation')", default=""
    )
    text_to_convert: str = Field(
        description="Text to convert to speech (if agent_type is 'text_to_speech') or translate (if agent_type is 'translation')", default=""
    )
