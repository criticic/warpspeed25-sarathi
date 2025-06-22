"""Multi-Agent Research and Communication System.

This module implements a LangGraph-based multi-agent system that handles:
- Research queries with web search and reflection
- Language identification and translation
- Text-to-speech conversion
- Intent routing to appropriate specialized agents

The system uses Google Gemini for reasoning and SarvamAI for language services.
"""

import base64
import io
import logging
import os
import re
import uuid

from dotenv import load_dotenv
from google.genai import Client
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydub import AudioSegment
from sarvamai import SarvamAI

from agent.configuration import Configuration
from agent.prompts import (
    answer_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    agent_routing_instructions,
    web_searcher_instructions,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.tools_and_schemas import Reflection, SearchQueryList, AgentRouting
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

log = logging.getLogger("uvicorn.error")
log.setLevel(logging.DEBUG)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

if os.getenv("SARVAM_API_KEY") is None:
    raise ValueError("SARVAM_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# Used for language identification and translation
sarvam_client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))


def save_combined_audio(audio_response, filename: str) -> None:
    """Save audio response from SarvamAI TTS by combining base64 encoded audio chunks.
    
    Args:
        audio_response: Response from SarvamAI TTS containing base64 encoded audio chunks
        filename: Output filename for the combined audio
    """
    if not isinstance(audio_response.audios, list) or not audio_response.audios:
        log.error("No audio data to save or data is not a list.")
        return

    combined_segment = None

    for i, b64_audio_string in enumerate(audio_response.audios):
        try:
            # Decode the base64 string to bytes
            audio_bytes = base64.b64decode(b64_audio_string)

            # Load the WAV data from bytes into an AudioSegment
            # Use io.BytesIO to treat bytes as a file-like object
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            if combined_segment is None:
                combined_segment = segment
            else:
                combined_segment += segment  # pydub overloads + for concatenation
        except Exception as e:
            log.error(f"Error processing audio chunk {i}: {e}")
            # Continue with other chunks
            continue

    if combined_segment:
        try:
            # Export the combined audio to a file
            combined_segment.export(filename, format="wav")
            log.info(f"Successfully saved combined audio to {filename}")
        except Exception as e:
            log.error(f"Error exporting combined audio: {e}")
    else:
        log.error("No audio segments were successfully processed to combine.")


def save_combined_audio_as_mp3(audio_response, filename: str) -> None:
    """Save audio response from SarvamAI TTS as MP3 optimized for WhatsApp.
    
    Args:
        audio_response: Response from SarvamAI TTS containing base64 encoded audio chunks
        filename: Output filename for the combined MP3 audio
    """
    if not isinstance(audio_response.audios, list) or not audio_response.audios:
        log.error("No audio data to save or data is not a list.")
        return

    combined_segment = None

    for i, b64_audio_string in enumerate(audio_response.audios):
        try:
            # Decode the base64 string to bytes
            audio_bytes = base64.b64decode(b64_audio_string)

            # Load the WAV data from bytes into an AudioSegment
            # Use io.BytesIO to treat bytes as a file-like object
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            if combined_segment is None:
                combined_segment = segment
            else:
                combined_segment += segment  # pydub overloads + for concatenation
        except Exception as e:
            log.error(f"Error processing audio chunk {i}: {e}")
            # Continue with other chunks
            continue

    if combined_segment:
        try:
            # Export as MP3 optimized for WhatsApp voice messages
            # WhatsApp prefers: MP3, mono, 16kHz sample rate, 64kbps bitrate
            combined_segment.export(
                filename, 
                format="mp3",
                parameters=["-ac", "1", "-ar", "16000", "-b:a", "64k"]  # mono, 16kHz, 64kbps
            )
            log.info(f"Successfully saved combined MP3 audio to {filename}")
        except Exception as e:
            log.error(f"Error exporting combined MP3 audio: {e}")
    else:
        log.error("No audio segments were successfully processed to combine.")


# Nodes
def language_identification(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that identifies the language of the user's query.
    
    Uses SarvamAI Language Identification API to detect the language and script
    of the user's input message.
    
    Args:
        state: Current graph state containing the user's messages
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including detected language information
    """
    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content')]
    if not user_messages:
        return {
            "detected_language_code": None,
            "detected_script_code": None,
            "needs_translation": False
        }
    
    latest_message = user_messages[-1].content
    
    try:
        # Call SarvamAI Language Identification API using SDK
        response = sarvam_client.text.identify_language(
            input=latest_message
        )
        
        detected_language = response.language_code
        detected_script = response.script_code
        
        # Determine if translation is needed (if not English)
        needs_translation = detected_language is not None and detected_language != "en-IN"
        
        log.info(f"Language identification: detected={detected_language}, script={detected_script}, needs_translation={needs_translation}")
        
        return {
            "detected_language_code": detected_language,
            "detected_script_code": detected_script,
            "needs_translation": needs_translation
        }
        
    except Exception as e:
        log.error(f"Language identification failed: {e}")
        # Default to no translation needed if identification fails
        return {
            "detected_language_code": None,
            "detected_script_code": None,
            "needs_translation": False
        }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    grounding_metadata = response.candidates[0].grounding_metadata
    if grounding_metadata is not None and grounding_metadata.grounding_chunks is not None:
        resolved_urls = resolve_urls(grounding_metadata.grounding_chunks, state["id"])
        # Gets the citations and adds them to the generated text
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [item for citation in citations for item in citation["segments"]]
    else:
        # No grounding metadata or chunks available, use the response text as-is
        modified_text = response.text
        sources_gathered = []

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> str:
    """LangGraph node that evaluates the research findings.

    Determines whether the research is sufficient or if more is needed.

    Args:
        state: Current graph state containing the reflection output
        config: Configuration for the runnable

    Returns:
        "web_research" if more research is needed, "finalize_answer" otherwise
    """
    configurable = Configuration.from_runnable_config(config)
    max_loops = state.get("max_research_loops", configurable.max_research_loops)
    if state["is_sufficient"] or state["research_loop_count"] >= max_loops:
        return "finalize_answer"
    else:
        return "web_research"


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that generates the final answer.

    Args:
        state: Current graph state containing the research topic and summaries
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including the final answer in the messages key
    """
    # Check if this is a direct answer (no research required)
    if not state.get("research_required", True):
        log.info("Processing direct answer in finalize_answer")
        # For direct answers, the message is already in state from agent_router
        existing_messages = state.get("messages", [])
        if existing_messages:
            # Store the answer for potential translation
            latest_message = existing_messages[-1]
            log.info(f"Direct answer content: '{latest_message.content[:200]}...' (length: {len(latest_message.content)})")
            
            # For direct answers, determine if translation is needed
            needs_translation_back = should_translate_back_to_original_language(state)
            
            result_dict = {
                "messages": existing_messages,
                "english_answer": latest_message.content,  # Store for potential translation if needed
                "needs_translation_back": needs_translation_back
            }
            
            # If translation is needed, set up parameters for translation_agent
            if needs_translation_back:
                log.info(f"Setting up translation: source=en-IN, target={state.get('detected_language_code')}")
                result_dict.update({
                    "text_to_convert": latest_message.content,  # The text to translate
                    "source_language": "en-IN",  # Source is English for research answers
                    "target_language": state.get("detected_language_code", "hi-IN"),  # Target is user's original language
                })
            
            return result_dict
        else:
            # Fallback if no messages exist
            return {"messages": []}

    # Research was required, generate answer from research results
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model", configurable.answer_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    log.info(f"Final answer content: '{result.content[:200]}...' (length: {len(result.content)})")

    # Store the English answer for potential translation
    needs_translation_back = should_translate_back_to_original_language(state)
    
    result_dict = {
        "messages": [result],
        "english_answer": result.content,  # Store for translation if needed
        "needs_translation_back": needs_translation_back
    }
    
    # If translation is needed, set up parameters for translation_agent
    if needs_translation_back:
        result_dict.update({
            "text_to_convert": result.content,  # The English text to translate
            "source_language": "en-IN",  # Source is English
            "target_language": state.get("detected_language_code", "hi-IN"),  # Target is user's original language
        })
    
    return result_dict


def smart_text_chunking(text, max_length=2400):
    """Split text into chunks that respect sentence boundaries and stay under max_length."""
    chunks = []
    current_chunk = ""
    
    # Split by sentences first
    sentences = []
    temp = text
    while temp:
        # Find sentence endings
        endings = [temp.find('.'), temp.find('!'), temp.find('?')]
        endings = [e for e in endings if e != -1]
        
        if not endings:
            sentences.append(temp)
            break
            
        next_end = min(endings) + 1
        sentences.append(temp[:next_end])
        temp = temp[next_end:].lstrip()
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:  # If we have content, save it as a chunk
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:  # If sentence itself is too long, split it at spaces
                words = sentence.split(' ')
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            chunks.append(word)  # Single word longer than limit
                    else:
                        current_chunk += (' ' + word if current_chunk else word)
        else:
            current_chunk += (' ' + sentence if current_chunk else sentence)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_audio_output(state: OverallState, config: RunnableConfig) -> OverallState:
    """Generate audio output from the final text response by chunking and combining audio segments."""    
    try:
        # Get the final message content
        if state.get("messages") and len(state["messages"]) > 0:
            final_message = state["messages"][-1].content
        else:
            return {}
        
        # Remove markdown formatting for cleaner audio output
        clean_message = remove_markdown_formatting(final_message)
        
        log.info(f"Generating audio for message: {clean_message}")
        log.info(f"Language code for audio generation: {state['detected_language_code']}")
        
        # Create unique identifier for this audio generation
        unique_id = str(uuid.uuid4())
        
        # Ensure static_audio directory exists
        os.makedirs("static_audio", exist_ok=True)
        
        # Split text into chunks that fit within TTS limits
        text_chunks = smart_text_chunking(clean_message, max_length=1400)
        log.info(f"Split text into {len(text_chunks)} chunks for TTS processing")
        
        audio_segments = []
        temp_files = []
        
        # Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            try:
                log.info(f"Processing chunk {i + 1}/{len(text_chunks)}: {chunk[:10]}...")  
                # Convert chunk to speech
                tts_response = sarvam_client.text_to_speech.convert(
                    text=chunk,
                    target_language_code=state["detected_language_code"],
                    enable_preprocessing=True,
                )

                # log.info(tts_response)
                # save to a text file for debugging
                with open(f"static_audio/chunk_{unique_id}_{i}.txt", "w") as f:
                    import jsonpickle
                    f.write(jsonpickle.encode(tts_response, unpicklable=False))
                
                # Save chunk as WAV
                chunk_wav_filename = f"chunk_{unique_id}_{i}.wav"
                chunk_wav_filepath = os.path.join("static_audio", chunk_wav_filename)
                temp_files.append(chunk_wav_filepath)
                save_combined_audio(tts_response, chunk_wav_filepath)
                
                # Load as AudioSegment
                audio_segment = AudioSegment.from_wav(chunk_wav_filepath)
                audio_segments.append(audio_segment)
                
            except Exception as e:
                log.error(f"Error generating audio for chunk {i}: {e}")
                # Continue with other chunks
                continue
        
        if not audio_segments:
            log.error("No audio segments were successfully generated")
            return {}
        
        # Combine all audio segments
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio = combined_audio + segment
        
        # Export combined audio as MP3 optimized for WhatsApp
        mp3_filename = f"output_{unique_id}.mp3"
        mp3_filepath = os.path.join("static_audio", mp3_filename)
        
        # Export with settings optimized for WhatsApp voice messages
        # WhatsApp prefers: MP3, mono, 16kHz sample rate, 64kbps bitrate
        combined_audio.export(
            mp3_filepath, 
            format="mp3",
            parameters=["-ac", "1", "-ar", "16000", "-b:a", "64k"]  # mono, 16kHz, 64kbps
        )
        
        # Get base URL from config
        base_url = config.get("configurable", {}).get("base_url", "")
        audio_url = f"{base_url}static_audio/{mp3_filename}"
        
        # Clean up temporary WAV files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        log.info(f"Successfully generated combined audio: {audio_url}")
        return {"audio_output": audio_url}
        
    except Exception as e:
        # Log error instead of print
        log.error(f"Error generating audio: {e}")
        return {}


def should_generate_audio(state: OverallState) -> str:
    """Determine whether to generate audio output."""
    if state.get("generate_audio", False) and state.get("detected_language_code"):
        return "generate_audio_output"
    return "__end__"


def should_translate(state: OverallState) -> str:
    """Determine whether to generate audio output directly (translation is handled by translation_agent)."""
    return should_generate_audio(state)


def intelligent_chunking(text, chunk_size=1500):
    """Split a long text into smaller chunks without cutting mid-sentence or mid-URL."""
    chunks = []
    while len(text) > chunk_size:
        # Find the last newline character within the chunk size
        split_index = text.rfind("\n", 0, chunk_size)
        if split_index == -1:
            # If no newline, find the last space
            split_index = text.rfind(" ", 0, chunk_size)
        if split_index == -1:
            # If no space either, force split at chunk_size
            split_index = chunk_size
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()
    chunks.append(text)
    return chunks


def remove_markdown_formatting(text: str) -> str:
    """Remove markdown formatting characters from text for cleaner audio output."""
    # Remove bold and italic formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove code blocks and inline code
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove strikethrough
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text


# New Agent Routing Node
def agent_router(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that routes to appropriate agent based on user intent.

    Uses Gemini 2.0 Flash to determine which agent should handle the user's query.

    Args:
        state: Current graph state containing the user's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including agent_type and relevant parameters
    """
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.agent_router_model,
        temperature=0.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(AgentRouting)

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    formatted_prompt = agent_routing_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
    )

    log.info(f"Agent router received research_topic: '{research_topic[:500]}...'")

    result = structured_llm.invoke(formatted_prompt)

    # Preserve existing generate_audio state if it was previously set to True
    existing_generate_audio = state.get("generate_audio", False)
    generate_audio = result.generate_audio or existing_generate_audio

    log.info(f"Agent routing: agent_type={result.agent_type}, detected_lang={state.get('detected_language_code')}, needs_translation={state.get('needs_translation')}")
    log.info(f"Agent router answer: '{result.answer[:200]}...' (length: {len(result.answer)})")

    return {
        "agent_type": result.agent_type,
        "research_required": result.agent_type == "research",  # For backward compatibility
        "generate_audio": generate_audio,
        "source_language": result.source_language,
        "target_language": result.target_language,
        "text_to_convert": result.text_to_convert,
        "messages": [AIMessage(content=result.answer)]
        if result.agent_type == "direct_answer"
        else [],
    }


def route_to_agent(state: OverallState) -> str:
    """Route to the appropriate agent based on agent_type."""
    agent_type = state.get("agent_type", "direct_answer")
    
    if agent_type == "research":
        return "generate_query"
    elif agent_type == "translation":
        return "translation_agent"
    elif agent_type == "text_to_speech":
        return "text_to_speech_agent"
    elif agent_type == "direct_answer":
        return "finalize_answer"
    else:
        # For future agents like calendar, mailbox
        return f"{agent_type}_agent"


# Standalone Translation Agent
def translation_agent(state: OverallState, config: RunnableConfig) -> OverallState:
    """Standalone translation agent that handles translation requests.
    
    Uses SarvamAI Translation API to translate text between languages.
    
    Args:
        state: Current graph state containing translation parameters
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including translated text
    """
    text_to_translate = state.get("text_to_convert", "")
    source_lang = state.get("source_language", "auto")
    target_lang = state.get("target_language", "en")
    
    log.info(f"Translation agent called: source_lang={source_lang}, target_lang={target_lang}, text={(text_to_translate)}")
    
    if not text_to_translate:
        return {
            "messages": [AIMessage(content="I need text to translate. Please provide the text you want me to translate.")]
        }
    
    try:
        # Handle auto-detection of source language
        if source_lang == "auto":
            # Use language identification to detect source language
            lang_response = sarvam_client.text.identify_language(input=text_to_translate)
            detected_lang = lang_response.language_code or "en-IN"
            source_lang = map_to_sarvam_language_code(detected_lang)
        else:
            # Ensure provided source language is compatible with SarvamAI
            source_lang = map_to_sarvam_language_code(source_lang)
        
        # Ensure target language is also compatible with SarvamAI
        target_lang = map_to_sarvam_language_code(target_lang)
        
        # Get any script preference from state
        target_script = state.get("output_script") or state.get("detected_script_code")
        
        # Check if text needs chunking (SarvamAI has 2000 char limit)
        if len(text_to_translate) > 1800:  # Leave buffer for safety
            # Split text into chunks
            text_chunks = intelligent_chunking(text_to_translate, chunk_size=1800)
            translated_chunks = []
            
            log.info(f"Translating long text in {len(text_chunks)} chunks")
            
            # Translate each chunk
            for i, chunk in enumerate(text_chunks):
                try:
                    # Prepare translation parameters
                    translate_params = {
                        "input": chunk,
                        "source_language_code": source_lang,
                        "target_language_code": target_lang,
                        "speaker_gender": "Male"
                    }
                    
                    # Add output_script if specified and target script is Latin
                    if target_script == "Latn":
                        translate_params["output_script"] = "roman"
                    
                    translate_response = sarvam_client.text.translate(**translate_params)
                    translated_chunks.append(translate_response.translated_text)
                    log.info(f"Successfully translated chunk {i + 1}/{len(text_chunks)}")
                except Exception as chunk_error:
                    log.warning(f"Failed to translate chunk {i + 1}: {chunk_error}")
                    # If a chunk fails, keep the original text for that chunk
                    translated_chunks.append(chunk)
            
            # Combine all translated chunks
            translated_text = ' '.join(translated_chunks)
        else:
            # Text is short enough, translate directly
            # Prepare translation parameters
            translate_params = {
                "input": text_to_translate,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "speaker_gender": "Male"
            }
            
            # Add output_script if specified and target script is Latin
            if target_script == "Latn":
                translate_params["output_script"] = "roman"
            
            translate_response = sarvam_client.text.translate(**translate_params)
            translated_text = translate_response.translated_text
        
        return {
            "translation_result": translated_text,
            "messages": [AIMessage(content=translated_text)]
        }
        
    except Exception as e:
        log.error(f"Translation failed: {e}")
        return {
            "messages": [AIMessage(content=f"Sorry, I couldn't translate that text. Error: {str(e)}")]
        }


# Standalone Text-to-Speech Agent
def text_to_speech_agent(state: OverallState, config: RunnableConfig) -> OverallState:
    """Standalone text-to-speech agent that converts text to audio.
    
    Uses SarvamAI TTS API to convert text to speech. Prioritizes text_to_convert from 
    agent router, falls back to last AI message if needed.
    
    Args:
        state: Current graph state containing text to convert or messages
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including audio output
    """
    # First, try to get text from agent router's text_to_convert field
    text_to_convert = state.get("text_to_convert", "").strip()
    
    # If no text_to_convert from router, fall back to last AI message
    if not text_to_convert:
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content.strip():
                text_to_convert = message.content
                break
    
    language_code = state.get("detected_language_code", "hi-IN")  # Default to Hindi
    
    if not text_to_convert:
        return {
            "messages": [AIMessage(content="I need text to convert to speech. Please provide the text you want me to read aloud.")]
        }
    
    try:
        # Remove markdown formatting for cleaner audio
        clean_text = remove_markdown_formatting(text_to_convert)
        
        # Create unique identifier for this audio generation
        unique_id = str(uuid.uuid4())
        
        # Ensure static_audio directory exists
        os.makedirs("static_audio", exist_ok=True)
        
        # Generate audio
        tts_response = sarvam_client.text_to_speech.convert(
            text=clean_text,
            target_language_code=language_code,
            enable_preprocessing=True,
        )
        
        # Save audio as MP3 optimized for WhatsApp
        mp3_filename = f"tts_output_{unique_id}.mp3"
        mp3_filepath = os.path.join("static_audio", mp3_filename)
        save_combined_audio_as_mp3(tts_response, mp3_filepath)
        
        # Get base URL from config
        base_url = config.get("configurable", {}).get("base_url", "")
        audio_url = f"{base_url}static_audio/{mp3_filename}"
        
        return {
            "audio_output": audio_url,
        }
        
    except Exception as e:
        log.error(f"Text-to-speech conversion failed: {e}")
        return {
            "messages": [AIMessage(content=f"Sorry, I couldn't convert that text to speech. Error: {str(e)}")]
        }


# Specialized translation node for translating responses back to user's original language
def translate_back_to_original(state: OverallState, config: RunnableConfig) -> OverallState:
    """Translate the English response back to the user's original language.
    
    Args:
        state: Current graph state containing the English answer and detected language
        config: Configuration for the runnable
        
    Returns:
        Dictionary with state update including translated response
    """
    english_answer = state.get("english_answer", "")
    target_language = state.get("detected_language_code", "hi-IN")
    
    log.info(f"Translating back to original language: target_language={target_language}, english_answer length={len(english_answer)}")
    
    if not english_answer:
        # Fallback to the last message if english_answer is not available
        messages = state.get("messages", [])
        if messages:
            english_answer = messages[-1].content
    
    if not english_answer:
        log.error("No English answer found to translate back")
        return {}
    
    try:
        # Ensure target language is compatible with SarvamAI
        target_lang = map_to_sarvam_language_code(target_language)
        
        # Get the original script for output formatting
        original_script = state.get("detected_script_code")
        
        # Check if text needs chunking for translation
        if len(english_answer) > 1800:
            # Split text into chunks
            text_chunks = intelligent_chunking(english_answer, chunk_size=1800)
            translated_chunks = []
            
            log.info(f"Translating response back to {target_lang} in {len(text_chunks)} chunks")
            
            # Translate each chunk
            for i, chunk in enumerate(text_chunks):
                try:
                    # Prepare translation parameters
                    translate_params = {
                        "input": chunk,
                        "source_language_code": "en-IN",  # English source
                        "target_language_code": target_lang,
                        "speaker_gender": "Male"
                    }
                    
                    # Add output_script if original script was Latin
                    if original_script == "Latn":
                        translate_params["output_script"] = "roman"
                    
                    translate_response = sarvam_client.text.translate(**translate_params)
                    translated_chunks.append(translate_response.translated_text)
                    log.info(f"Successfully translated chunk {i + 1}/{len(text_chunks)}")
                except Exception as chunk_error:
                    log.warning(f"Failed to translate chunk {i + 1}: {chunk_error}")
                    # If a chunk fails, keep the original text for that chunk
                    translated_chunks.append(chunk)
            
            # Combine all translated chunks
            translated_text = ' '.join(translated_chunks)
        else:
            # Text is short enough, translate directly
            # Prepare translation parameters
            translate_params = {
                "input": english_answer,
                "source_language_code": "en-IN",  # English source
                "target_language_code": target_lang,
                "speaker_gender": "Male"
            }
            
            # Add output_script if original script was Latin
            if original_script == "Latn":
                translate_params["output_script"] = "roman"
            
            translate_response = sarvam_client.text.translate(**translate_params)
            translated_text = translate_response.translated_text
        
        log.info(f"Successfully translated response back to {target_lang}")
        
        # Update the messages with the translated response
        return {
            "messages": [AIMessage(content=translated_text)],
            "translation_result": translated_text
        }
        
    except Exception as e:
        log.error(f"Translation back to original language failed: {e}")
        # If translation fails, return the original English response
        return {
            "messages": [AIMessage(content=english_answer)]
        }


# Future Agent Placeholders
def calendar_agent(state: OverallState, config: RunnableConfig) -> OverallState:
    """Handle calendar-related requests - to be implemented later."""
    return {
        "messages": [AIMessage(content="Calendar functionality is coming soon!")]
    }


def mailbox_agent(state: OverallState, config: RunnableConfig) -> OverallState:
    """Handle mailbox-related requests - to be implemented later."""
    return {
        "messages": [AIMessage(content="Mailbox functionality is coming soon!")]
    }


# Update routing functions to handle new agents
def should_translate_after_agent(state: OverallState) -> str:
    """Determine whether to generate audio after agent processing (translation handled by translation_agent)."""
    return should_generate_audio_after_agent(state)


def should_generate_audio_after_agent(state: OverallState) -> str:
    """Determine whether to generate audio after agent processing."""
    agent_type = state.get("agent_type", "direct_answer")
    
    # Skip audio generation for TTS agent (already handled)
    if agent_type == "text_to_speech":
        return "__end__"
    
    # Check if audio generation is needed for other agents
    if state.get("generate_audio", False) and state.get("detected_language_code"):
        return "generate_audio_output"
    return "__end__"


def route_after_finalize_answer(state: OverallState) -> str:
    """Route after finalize_answer to either translate back to original language or continue to audio."""
    needs_translation_back = state.get("needs_translation_back", False)
    log.info(f"Route after finalize_answer: needs_translation_back={needs_translation_back}")
    
    if needs_translation_back:
        return "translation_agent"
    return should_generate_audio_after_agent(state)


def should_translate_back_to_original_language(state: OverallState) -> bool:
    """Determine if the response needs to be translated back to the user's original language.
    
    Args:
        state: Current graph state containing language detection results
        
    Returns:
        True if translation back to original language is needed, False otherwise
    """
    detected_lang = state.get("detected_language_code")
    needs_translation = state.get("needs_translation", False)
    
    # Translation back to original language is needed if:
    # 1. A non-English language was detected
    # 2. The original language identification indicated translation was needed
    # 3. The detected language is not English variants
    result = detected_lang and needs_translation and not detected_lang.startswith("en")
    
    log.info(f"Should translate back to original: detected_lang={detected_lang}, needs_translation={needs_translation}, result={result}")
    
    return result


def map_to_sarvam_language_code(detected_code: str) -> str:
    """Map detected language codes to SarvamAI supported language codes.
    
    Args:
        detected_code: Language code from language identification
        
    Returns:
        SarvamAI compatible language code
    """
    # SarvamAI supported source language codes
    sarvam_supported = {
        'auto', 'bn-IN', 'en-IN', 'gu-IN', 'hi-IN', 'kn-IN', 'ml-IN', 
        'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'as-IN', 'brx-IN', 
        'doi-IN', 'kok-IN', 'ks-IN', 'mai-IN', 'mni-IN', 'ne-IN', 
        'sa-IN', 'sat-IN', 'sd-IN', 'ur-IN'
    }
    
    # If the detected code is already supported, use it
    if detected_code in sarvam_supported:
        return detected_code
    
    # Common mappings for detected codes that might not match exactly
    language_mappings = {
        'en': 'en-IN',
        'hi': 'hi-IN', 
        'bn': 'bn-IN',
        'gu': 'gu-IN',
        'kn': 'kn-IN',
        'ml': 'ml-IN',
        'mr': 'mr-IN',
        'or': 'od-IN',  # Oriya -> Odia
        'pa': 'pa-IN',
        'ta': 'ta-IN',
        'te': 'te-IN',
        'as': 'as-IN',
        'ne': 'ne-IN',
        'sa': 'sa-IN',
        'ur': 'ur-IN',
        # Add more mappings as needed
    }
    
    # Try to map the base language code (remove country suffix if present)
    base_code = detected_code.split('-')[0].lower()
    if base_code in language_mappings:
        return language_mappings[base_code]
    
    # If we can't map it, default to auto-detection
    log.warning(f"Could not map language code '{detected_code}' to SarvamAI format, using 'auto'")
    return 'auto'


# Create our Multi-Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define all the nodes
builder.add_node("language_identification", language_identification)
builder.add_node("agent_router", agent_router)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("translation_agent", translation_agent)
builder.add_node("text_to_speech_agent", text_to_speech_agent)
builder.add_node("calendar_agent", calendar_agent)
builder.add_node("mailbox_agent", mailbox_agent)
builder.add_node("translate_back_to_original", translate_back_to_original)

builder.add_node("generate_audio_output", generate_audio_output)

# Set the entrypoint as language_identification
builder.add_edge(START, "language_identification")
builder.add_edge("language_identification", "agent_router")

# Route to appropriate agents based on intent
builder.add_conditional_edges(
    "agent_router",
    route_to_agent,
    {
        "generate_query": "generate_query",  # Research agent
        "finalize_answer": "finalize_answer",  # Direct answer
        "translation_agent": "translation_agent",  # Translation agent
        "text_to_speech_agent": "text_to_speech_agent",  # TTS agent
        "calendar_agent": "calendar_agent",  # Future calendar agent
        "mailbox_agent": "mailbox_agent",  # Future mailbox agent
    },
)

# Research agent flow (existing)
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

# Special handling for finalize_answer - it may need translation back to original language
builder.add_conditional_edges(
    "finalize_answer", 
    route_after_finalize_answer, 
    ["translation_agent", "generate_audio_output", "__end__"]
)

# After each agent completes, check for translation/audio needs
agent_endpoints = [
    "translation_agent", 
    "text_to_speech_agent", 
    "calendar_agent", 
    "mailbox_agent"
]

for endpoint in agent_endpoints:
    builder.add_conditional_edges(
        endpoint, 
        should_translate_after_agent, 
        ["generate_audio_output", "__end__"]
    )

# After translation, route to audio generation or end


# Audio generation leads to END
builder.add_edge("generate_audio_output", END)

graph = builder.compile(name="multi-agent-system")
image = graph.get_graph().draw_mermaid_png()
# Save the graph image
with open("agent_graph.png", "wb") as f:
    f.write(image)

