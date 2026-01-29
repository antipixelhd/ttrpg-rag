# =============================================================================
# Response Generation Module
# =============================================================================
# This module handles generating LLM responses from retrieved chunks.
# It uses strict prompting to minimize hallucinations and ensure the LLM
# only answers based on the provided context.

from openai import OpenAI

from src.config import get_secrets


# =============================================================================
# System Prompt - This is critical for avoiding hallucinations
# =============================================================================
SYSTEM_PROMPT = """
You are an assistant that provides accurate, concise answers about our Dungeons & Dragons campaign using only the information found in the supplied session summary files. Each file represents a distinct segment (or “part”) of a session and is named with explicit part numbers (e.g., `session3_part1.md`, `session3_part2.md`, etc.). These files are in markdown using bullet lists, and their filenames show both the session number and the part number (e.g., “part 1/2/3”), conveying order within a session.
You must:
- Use only information found within these segmented session summaries for your responses; do not draw on outside knowledge or assumptions.
- Always reason through, step by step, the pieces of information drawn from the relevant files

Player characters are: Sai, Selene, Eve, Melli, Kenshi, Vorak, and Aurora.

# Steps
1. For each question, review the content of all provided session summary segment files. Each file is named for a session and a part with a strict order (e.g., session4_part2.md).
2. Identify and collect all parts that correspond to relevant sessions and order them as “part 1,” “part 2,” and so on for each session.
3. Determine which segments (by full filename, including part number) contain answers or evidence for the question.
4. Present your answer in a clearly separated, concise section after all reasoning.
5. If the information is not available in any part, clearly state that the answer cannot be found in the provided materials.

Remember: It is better to say "I don't know" than to provide incorrect information."""


def create_llm_client(config):
    """
    Create an OpenAI client for LLM calls.
    
    Args:
        config: Configuration dictionary (not used, but kept for consistency)
        
    Returns:
        OpenAI: An initialized OpenAI client
    """
    secrets = get_secrets()
    api_key = secrets.get('openai_api_key')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in secrets.yaml")
    
    return OpenAI(api_key=api_key)


def format_context(chunks):
    """
    Format retrieved chunks into a context string for the LLM.
    
    Each chunk is labeled with its source session for reference.
    
    Args:
        chunks: List of chunk dictionaries from retrieval
        
    Returns:
        str: Formatted context string
    """
    if not chunks:
        return "No session notes were found."
    
    context_parts = []
    
    for chunk in chunks:
        # Get chunk metadata
        name = chunk.get('name', 'Unknown')
        content = chunk.get('content', '')
        
        # Format each chunk with a header
        chunk_text = f"[{name}]\n{content}"
        context_parts.append(chunk_text)
    
    return "\n\n---\n\n".join(context_parts)


def build_user_prompt(question, context):
    """
    Build the user message with context and question.
    
    Args:
        question: The user's question
        context: Formatted context from retrieved chunks
        
    Returns:
        str: The complete user message
    """
    return f"""SESSION NOTES:
{context}

---

QUESTION: {question}

ANSWER:"""


def generate_response(question, retrieved_chunks, config, logger=None):
    """
    Generate an LLM response based on retrieved chunks.
    
    This is the main function for response generation. It:
    1. Formats the retrieved chunks into context
    2. Builds a prompt with strict instructions
    3. Calls the LLM with low temperature to minimize hallucinations
    4. Returns the response
    
    Args:
        question: The user's question
        retrieved_chunks: List of chunk dictionaries from retrieval
        config: Configuration dictionary with response settings
        logger: Optional logger for tracking progress
        
    Returns:
        str: The LLM's response
    """
    # Get response settings from config
    model = config.get('response', {}).get('model', 'gpt-5-nano')
    temperature = config.get('response', {}).get('temperature', 0.1)
    max_tokens = config.get('response', {}).get('max_tokens', 500)
    
    message = f"Generating response using {model} (temp={temperature})..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    # Format the context from retrieved chunks
    context = format_context(retrieved_chunks)
    
    # Build the user prompt
    user_prompt = build_user_prompt(question, context)
    
    # Create the LLM client
    client = create_llm_client(config)
    
    # Call the LLM with strict settings
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens
        )
        
        answer = response.choices[0].message.content.strip()
        
        message = "Response generated successfully"
        if logger:
            logger.info(message)
        else:
            print(message)
        
        return answer
        
    except Exception as e:
        error_msg = f"Error generating response: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"Error: {error_msg}")
        raise

