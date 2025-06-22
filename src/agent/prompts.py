from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


agent_routing_instructions = """Your goal is to determine which agent should handle the user's query and extract relevant parameters.

Available Agent Types:
1. "research" - For queries requiring web research (current events, specific facts, recent information)
2. "translation" - For translation requests (e.g., "translate this to Spanish", "how do you say hello in French")
3. "text_to_speech" - For text-to_speech requests (e.g., "read this aloud", "convert this to audio")
4. "direct_answer" - For simple questions, greetings, simple text based queries (summarizing or explaining unless it explaining requires research) or general information that doesn't need research
5. "calendar" - For calendar-related requests (future implementation)
6. "mailbox" - For email-related requests (future implementation)

Instructions:
- Analyze the user's query to understand its intent and determine the most appropriate agent
- IMPORTANT: Pay close attention to the FULL context provided, including any document transcriptions, audio transcriptions, or media content
- If the user asks questions like "What is this?", "Explain this", "Summarize this" and there is document/media content provided, analyze that content to provide a comprehensive answer
- For document analysis, image analysis, or content explanation, use "direct_answer" and provide a detailed response based on the provided content
- Extract relevant parameters based on the agent type
- For translation: identify source and target languages, and the text to translate
- For text_to_speech: identify the text to convert to speech
- Check if the user wants an audio response regardless of the main task
- IMPORTANT: For direct answers, always respond in English as the user's query
- IMPORTANT: For research queries, you can provide the answer in English since it will be translated back to the user's language automatically
- Take care of context and previous interactions, if any
- The current date is {current_date}

Format:
- Format your response as a JSON object with ALL these exact keys:
  - "agent_type": one of the agent types listed above
  - "answer": If agent_type is "direct_answer", provide the answer IN THE SAME LANGUAGE as the user's query based on ALL provided content. For research, leave empty.
  - "generate_audio": boolean, true if user wants audio output
  - "source_language": source language code for translation (empty if not translation)
  - "target_language": target language code for translation (empty if not translation)  
  - "text_to_convert": text to translate or convert to speech (empty if not applicable)

Examples:

User Query: "hello, how are you?"
```json
{{
    "agent_type": "direct_answer",
    "answer": "Hello! I'm doing well, thank you for asking. How can I help you today?",
    "generate_audio": false,
    "source_language": "",
    "target_language": "",
    "text_to_convert": ""
}}
```

User Query: "What is this?" with Document transcription about an IIT certificate
```json
{{
    "agent_type": "direct_answer",
    "answer": "This is a document from the Indian Institute of Technology (BHU) Varanasi, specifically from the School of Materials Science and Technology. It appears to be an official institutional document with the IIT logo and institutional branding.",
    "generate_audio": false,
    "source_language": "",
    "target_language": "",
    "text_to_convert": ""
}}
```

User Query: "नमस्कार, आप कैसे हैं?" (Hindi: Hello, how are you?)
```json
{{
    "agent_type": "direct_answer",
    "answer": "नमस्कार! मैं ठीक हूँ, पूछने के लिए धन्यवाद। आज मैं आपकी कैसे मदद कर सकता हूँ?",
    "generate_audio": false,
    "source_language": "hi-IN",
    "target_language": "",
    "text_to_convert": ""
}}
```

User Query: "आज का मौसम कैसा है?" (Hindi: How is today's weather?)
```json
{{
    "agent_type": "research",
    "answer": "",
    "generate_audio": false,
    "source_language": "hi-IN",
    "target_language": "",
    "text_to_convert": ""
}}
```

User Query: "translate the text 'good morning' to Bengali"
```json
{{
    "agent_type": "translation",
    "answer": "",
    "generate_audio": false,
    "source_language": "en-IN",
    "target_language": "bn-IN", 
    "text_to_convert": "good morning"
}}
```

User Query: "read me the latest news about AI"
```json
{{
    "agent_type": "research",
    "answer": "",
    "generate_audio": true,
    "source_language": "",
    "target_language": "",
    "text_to_convert": ""
}}
```

User Query: "convert this text to speech: Hello world"
```json
{{
    "agent_type": "text_to_speech",
    "answer": "",
    "generate_audio": true,
    "source_language": "",
    "target_language": "",
    "text_to_convert": "Hello world"
}}
```

User Query: {research_topic}
"""


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- IMPORTANT: Generate search queries in ENGLISH even if the original question is in another language. This ensures better search results from web sources.
- Understand the original question (which may be in any language) and create English search queries that will help answer it.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries IN ENGLISH

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings IN ENGLISH.
- Only include the information found in the search results, don't make up any information.
- Note: The original user question may be in any language, but generate your research summary in English as it will be translated if needed.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question (which may be in any language).
- Generate a high-quality answer IN ENGLISH to the user's question based on the provided summaries and the user's question.
- The answer will be automatically translated to the user's original language if needed.
- Do NOT Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST. IT SHOULT NOT BE INCLUDED IN THE ANSWER. INCLUDE THE TEXT "RESEARCH OUTPUT" IN THE ANSWER.

User Context:
- {research_topic}

Summaries:
{summaries}"""
