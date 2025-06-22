"""Twilio webhook for the research agent."""

import asyncio
import base64
import logging
import mimetypes
import os
import requests
import time
import uuid
from typing import Tuple

from fastapi import FastAPI, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sarvamai import SarvamAI
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from google import genai
from google.genai import types

from agent.graph import graph

# Get Twilio credentials from environment variables
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
twilio_phone_number = "whatsapp:+14155238886"
sarvam_api_key = os.environ.get("SARVAM_API_KEY")

if not all([account_sid, auth_token, twilio_phone_number, sarvam_api_key]):
    raise ValueError(
        "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and SARVAM_API_KEY must be set"
    )


log = logging.getLogger("uvicorn.error")
log.setLevel(logging.DEBUG)

client = Client(account_sid, auth_token)
sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)


# In-memory store for conversation history
conversation_history = {}

# Track message timestamps and pending tasks for timeout mechanism
message_timestamps = {}
pending_tasks = {}
pending_messages = {}  # Store messages waiting to be processed

app = FastAPI()

# Create a directory for static files
if not os.path.exists("static_audio"):
    os.makedirs("static_audio")

app.mount("/static_audio", StaticFiles(directory="static_audio"), name="static_audio")


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


def process_audio_from_url(url: str) -> tuple[str | None, str | None]:
    """Download an audio file from a URL and transcribe it using SarvamAI."""
    try:
        audio_response = requests.get(url, auth=(account_sid, auth_token))
        audio_response.raise_for_status()
        audio_content = audio_response.content

        # Save the audio content to a temporary file
        with open("temp_audio.ogg", "wb") as f:
            f.write(audio_content)

        # Transcribe the audio file
        with open("temp_audio.ogg", "rb") as f:
            response = sarvam_client.speech_to_text.transcribe(
                file=f, model="saarika:v2.5"
            )
        os.remove("temp_audio.ogg")
        
        log.info(f"Audio transcription completed: '{response.transcript}' (language: {response.language_code})")
        return response.transcript, response.language_code
    except requests.exceptions.RequestException as e:
        log.error(f"Error downloading audio file: {e}")
        return None, None
    except Exception as e:
        log.error(f"Error processing audio file: {e}")
        return None, None


def process_document_from_url(url: str, media_content_type: str | None = None) -> Tuple[str | None, str | None]:
    """Download an image or PDF from a URL and transcribe it using Gemini 2.5 Flash."""
    try:
        # Download the file
        response = requests.get(url, auth=(account_sid, auth_token))
        response.raise_for_status()
        file_content = response.content
        
        # Determine the MIME type
        if media_content_type:
            mime_type = media_content_type
        else:
            mime_type, _ = mimetypes.guess_type(url)
            if not mime_type:
                # Try to determine from response headers
                mime_type = response.headers.get('content-type', '').split(';')[0]
        
        log.info(f"Processing document with MIME type: {mime_type}")
        
        # Check if it's a supported format
        supported_image_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
        supported_pdf_types = ['application/pdf']
        
        if mime_type in supported_image_types:
            # Handle images using LangChain ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
                max_retries=2,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            
            # Convert file content to base64
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            # Create the prompt for image transcription
            image_prompt = """Please carefully analyze and transcribe the content of this image. 
            
Instructions:
- Extract all visible text accurately
- If this is a chart, graph, or diagram, describe what it shows and extract any text labels or data
- If this is a form, extract all fields and their values
- Maintain formatting and structure where possible
- If the image contains multiple languages, indicate which languages are present
- If text is unclear or partially obscured, indicate this in your transcription

Please provide a comprehensive transcription of the image content."""

            # Create message content with the image
            message_content = [
                {"type": "text", "text": image_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{file_base64}"
                    }
                }
            ]
            
            # Send to Gemini for transcription
            response = llm.invoke([{"role": "user", "content": message_content}])
            transcribed_text = response.content
            log.info(f"Image transcription completed: '{transcribed_text[:200]}...' (length: {len(transcribed_text)})")
            
        elif mime_type in supported_pdf_types:
            # Handle PDFs using Google GenAI client directly
            try:
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                
                # Create the prompt for PDF transcription
                pdf_prompt = """Please carefully analyze and transcribe the content of this PDF document.
                
Instructions:
- Extract all text content page by page
- Maintain formatting and structure where possible
- If the document contains tables, preserve table structure
- If the document contains multiple languages, indicate which languages are present
- Include any important visual elements like charts or diagrams with descriptions
- If any text is unclear, indicate this in your transcription

Please provide a comprehensive transcription of the PDF content."""
                
                # Use Google GenAI client to process PDF
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=file_content,
                            mime_type='application/pdf',
                        ),
                        pdf_prompt
                    ]
                )

                log.info("Output from Google GenAI client:", response.text)
                
                transcribed_text = response.text
                log.info(f"PDF transcription completed: '{transcribed_text[:200]}...' (length: {len(transcribed_text)})")
                
            except Exception as pdf_error:
                log.error(f"Error processing PDF with Google GenAI client: {pdf_error}")
                return f"I received a PDF document but encountered an error while processing it: {str(pdf_error)}", None
            
        else:
            log.warning(f"Unsupported media type: {mime_type}")
            return f"I received a file of type {mime_type}, but I can only process images (JPEG, PNG, GIF, WebP) and PDF documents.", None
        
        log.info(f"Document transcription completed, length: {len(transcribed_text)}, first 200 chars: '{transcribed_text[:200]}'")
        return transcribed_text, None  # No language detection for documents yet
            
    except requests.exceptions.RequestException as e:
        log.error(f"Error downloading document: {e}")
        return None, None
    except Exception as e:
        log.error(f"Error processing document: {e}")
        return None, None


def get_media_type_from_url(url: str) -> str | None:
    """Determine media type from URL or by making a HEAD request."""
    try:
        # First try to determine from URL extension
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type:
            return mime_type
        
        # If that fails, make a HEAD request to get content-type
        head_response = requests.head(url, auth=(account_sid, auth_token))
        content_type = head_response.headers.get('content-type', '')
        return content_type.split(';')[0] if content_type else None
    except Exception:
        return None


async def delayed_response_handler(
    from_number: str,
    timeout: float = 7.0,
):
    """Handle delayed response with timeout to batch multiple messages."""
    # Wait for the timeout period
    await asyncio.sleep(timeout)
    
    # Check if this is still the latest message for this user
    current_time = time.time()
    last_message_time = message_timestamps.get(from_number, 0)
    
    # If another message came in during our sleep, don't process
    if current_time - last_message_time < timeout - 0.5:  # 0.5s buffer
        return
    
    # Get all pending messages for this user
    messages_data = pending_messages.get(from_number, [])
    if not messages_data:
        return
    
    # Send the "working on it" message now that we're actually processing
    client.messages.create(
        body="I've received your request and I'm working on it. I'll get back to you shortly.",
        from_=twilio_phone_number,
        to=from_number
    )
    
    # Combine all messages into one query with a space separator
    combined_query = " ".join([msg["query"] for msg in messages_data])
    
    # Use the most recent message's settings for audio and language
    latest_message = messages_data[-1]
    base_url = latest_message["base_url"]
    is_audio = any(msg["is_audio"] for msg in messages_data)  # Audio if any message was audio
    language_code = latest_message["language_code"]
    
    # Clear the pending data for this user
    if from_number in pending_tasks:
        del pending_tasks[from_number]
    if from_number in pending_messages:
        del pending_messages[from_number]
    
    # Process the research with combined query
    await run_research(combined_query, from_number, base_url, is_audio, language_code)


async def run_research(
    query: str,
    from_number: str,
    base_url: str,
    is_audio: bool = False,
    language_code: str | None = None,
):
    """Run the research graph with the user's query."""
    log.info(f"Sending result to {from_number}")
    log.info(f"Twilio phone number: {twilio_phone_number}")

    history = conversation_history.get(from_number, [])
    # If we have a previous state, extract messages and audio preferences
    if isinstance(history, dict):
        messages = history.get("messages", [])
        existing_generate_audio = history.get("generate_audio", False)
        existing_language_code = history.get("language_code")
    else:
        messages = history if isinstance(history, list) else []
        existing_generate_audio = False
        existing_language_code = None
    
    messages.append(HumanMessage(content=query))

    # Build initial state with conversation history and audio preferences
    initial_state = {
        "messages": messages,
        "generate_audio": is_audio or existing_generate_audio,
        "language_code": language_code or existing_language_code,
        "id": str(uuid.uuid4()),  # Generate unique ID for this request
    }
    
    # Pass base_url via config for audio generation
    config = {"configurable": {"base_url": base_url}}
    
    # The graph is asynchronous, so we can await it
    final_state = await graph.ainvoke(initial_state, config=config)

    # Extract the final message content
    if final_state and final_state.get("messages"):
        conversation_history[from_number] = final_state["messages"]
        result = final_state["messages"][-1].content
    else:
        result = "Sorry, I couldn't find an answer to your question."
        conversation_history.setdefault(from_number, []).append(AIMessage(content=result))

    # Chunk the result and send it back to the user
    message_chunks = intelligent_chunking(result)
    for chunk in message_chunks:
        client.messages.create(body=chunk, from_=twilio_phone_number, to=from_number)

    # Check if audio was generated and send it
    if final_state.get("audio_output"):
        try:
            audio_url = final_state["audio_output"]
            log.info(f"Sending audio response to {from_number}: {audio_url}")
            
            # Send the audio file as a WhatsApp voice message
            # For WhatsApp, we need to send it as media_url (not body)
            client.messages.create(
                media_url=[audio_url],
                from_=twilio_phone_number,
                to=from_number
            )
            log.info("Audio successfully sent as WhatsApp voice message")
            
        except Exception as e:
            log.error(f"Error sending audio response: {e}")
            # Fallback: send the URL as text if audio sending fails
            client.messages.create(
                body=f"Audio response generated: {audio_url}",
                from_=twilio_phone_number, 
                to=from_number
            )

    # Store the entire final state to preserve audio preferences
    conversation_history[from_number] = final_state


@app.post("/webhook/twilio")
async def message(
    request: Request,
    From: str = Form(...),
    Body: str = Form(None),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    MediaUrl1: str = Form(None),
    MediaContentType1: str = Form(None),
    MediaUrl2: str = Form(None),
    MediaContentType2: str = Form(None),
    NumMedia: int = Form(0),
):
    """Handle incoming messages from Twilio."""
    log.info(f"Received message from {From}, NumMedia: {NumMedia}")
    query = Body
    is_audio = False
    language_code = None
    processed_content = []

    # Process multiple media files if present
    if NumMedia > 0:
        media_urls = [MediaUrl0, MediaUrl1, MediaUrl2]
        media_types = [MediaContentType0, MediaContentType1, MediaContentType2]
        
        for i in range(min(NumMedia, 3)):  # Process up to 3 media files
            media_url = media_urls[i]
            media_content_type = media_types[i]
            
            if not media_url:
                continue
                
            log.info(f"Processing media {i + 1}: {media_url}, ContentType: {media_content_type}")
            
            # Determine the media type
            media_type = media_content_type or get_media_type_from_url(media_url)
            log.info(f"Detected media type: {media_type}")
            
            if media_type and media_type.startswith('audio/'):
                # Process audio files
                audio_query, audio_language = process_audio_from_url(media_url)
                if audio_query:
                    log.info(f"Adding audio transcription to query: '{audio_query}'")
                    processed_content.append(f"Audio transcription: {audio_query}")
                    is_audio = True
                    if audio_language and not language_code:
                        language_code = audio_language
                        
            elif media_type and (media_type.startswith('image/') or media_type == 'application/pdf'):
                # Process images and PDFs
                doc_query, doc_language = process_document_from_url(media_url, media_type)
                if doc_query:
                    if doc_query.startswith("I received a file of type"):
                        log.info(f"Unsupported document type message: {doc_query}")
                        processed_content.append(doc_query)
                    else:
                        log.info(f"Adding document transcription to query: '{doc_query[:200]}...'")
                        processed_content.append(f"Document transcription: {doc_query}")
                        
            else:
                # Unsupported media type
                processed_content.append(f"Unsupported media file of type '{media_type or 'unknown'}' (I can only process audio files, images, and PDFs)")

    # Combine text message with processed media content
    if processed_content:
        log.info(f"Processed content items: {len(processed_content)}")
        for i, content in enumerate(processed_content):
            log.info(f"Content {i + 1}: {content[:100]}...")
        
        if query:
            query = f"{query}\n\n" + "\n\n".join(processed_content)
            log.info(f"Combined query with original text: '{query[:200]}...' (total length: {len(query)})")
        else:
            query = "\n\n".join(processed_content)
            log.info(f"Query from processed content only: '{query[:200]}...' (total length: {len(query)})")

    log.info(f"Final query to process: '{query[:500]}...' (total length: {len(query)})")

    if not query:
        resp = MessagingResponse()
        resp.message("I'm sorry, I couldn't process your message. Please try again.")
        return Response(content=str(resp), media_type="application/xml")

    if query.strip().lower() == "/new":
        if From in conversation_history:
            del conversation_history[From]
        # Clear any pending tasks for this user
        if From in pending_tasks:
            pending_tasks[From].cancel()
            del pending_tasks[From]
        if From in message_timestamps:
            del message_timestamps[From]
        if From in pending_messages:
            del pending_messages[From]
        resp = MessagingResponse()
        resp.message("New session started. I've cleared our previous conversation.")
        return Response(content=str(resp), media_type="application/xml")

    # Update timestamp for this user
    message_timestamps[From] = time.time()
    
    # Store the message data
    message_data = {
        "query": query,
        "base_url": str(request.base_url),
        "is_audio": is_audio,
        "language_code": language_code,
        "timestamp": time.time()
    }
    
    # Add to pending messages
    if From not in pending_messages:
        pending_messages[From] = []
    pending_messages[From].append(message_data)
    
    # Cancel any existing pending task for this user
    if From in pending_tasks and not pending_tasks[From].done():
        pending_tasks[From].cancel()
    
    # Create a new delayed response task
    task = asyncio.create_task(delayed_response_handler(From))
    pending_tasks[From] = task

    # Return empty response - no immediate acknowledgment
    resp = MessagingResponse()
    return Response(content=str(resp), media_type="application/xml")
