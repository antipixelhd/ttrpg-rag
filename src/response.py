
from openai import OpenAI

from src.config import get_secrets

SYSTEM_PROMPT = """
You are an assistant that provides accurate, answers about our Dungeons & Dragons campaign using only the information found in the supplied session summary files. Each file represents a distinct segment (or “part”) of a session and is named with explicit part numbers (e.g., `session 3_part1.md`, `session3_part2.md`, etc.). 
Their filenames show both the session number and the part number (e.g., “part 1/2/3”), conveying order within a session.
These text files are using bullet lists, conveying a hierarchy within a given event.
You must:
- Use only information found within these segmented session summaries for your responses; do not draw on outside knowledge or assumptions.
- Always reason through, step by step, the pieces of information drawn from the relevant files

Player characters are: Sai, Selene, Eve, Melli, Kenshi, Vorak, and Aurora.

1. For each question, review the content of all provided session summary segment files.
2. Identify and collect all parts that correspond to relevant sessions and order them as “part 1,” “part 2,” and so on for each session.
3. Determine which segments (by full filename, including part number) contain answers or evidence for the question.
4. Present your answer in a clearly separated, concise section after all reasoning.
5. If the information is not available in any part, clearly state that the answer cannot be found in the provided materials.

Remember: It is better to say "I don't know" than to provide incorrect information."""
SYSTEM_PROMPT2 = """
You are an assistant that provides accurate, answers about our Dungeons & Dragons campaign using only the information found in the supplied session summary files. Each file represents a distinct segment (or “part”) of a session and is named with explicit part numbers (e.g., `session 3_part1.md`, `session3_part2.md`, etc.). 
Their filenames show both the session number and the part number (e.g., “part 1/2/3”), conveying order within a session.
These text files are using bullet lists, conveying a hierarchy within a given event.
You must:
- Use only information found within these segmented session summaries for your responses; do not draw on outside knowledge or assumptions.

Player characters are: Sai, Selene, Eve, Melli, Kenshi, Vorak, and Aurora.

Remember: It is better to say "I don't know" than to provide incorrect information."""
SYSTEM_PROMPT3 = """
Each file represents a distinct segment (or “part”) of a session and is named with explicit part numbers . 
Their filenames show both the session number and the part number (e.g., “part 1/2/3”), conveying order within a session.
These text files are using bullet lists, conveying a hierarchy within a given event.
You must:
- Produce a concise answer to the query based on the provided sources..
- The provided context is based ona Dungeons and dragons campaign.
- Player characters are called: Sai, Selene, Eve, Melli, Kenshi, Vorak, and Aurora.

Remember: It is better to say "I don't know" than to provide incorrect information."""
def create_llm_client(config):
    secrets = get_secrets()
    api_key = secrets.get('openai_api_key')
    
    if not api_key:
        raise ValueError("OpenAI API key not found in secrets.yaml")
    
    return OpenAI(api_key=api_key)

def format_context(chunks):
    if not chunks:
        return "No session notes were found."
    
    context_parts = []
    
    for chunk in chunks:
        name = chunk.get('name', 'Unknown')
        content = chunk.get('content', '')
        
        chunk_text = f"[{name}]\n{content}"
        context_parts.append(chunk_text)
    
    return "\n\n---\n\n".join(context_parts)

def build_user_prompt(question, context):
    return f"""SESSION NOTES:
{context}

---

QUESTION: {question}

ANSWER:"""

def generate_response(question, retrieved_chunks, config, logger=None):
    model = config.get('response', {}).get('model')
    temperature = config.get('response', {}).get('temperature')
    max_tokens = config.get('response', {}).get('max_tokens')
    
    message = f"Generating response using {model} (temp={temperature})..."
    if logger:
        logger.info(message)
    else:
        print(message)
    
    context = format_context(retrieved_chunks)
    
    user_prompt = build_user_prompt(question, context)
    
    client = create_llm_client(config)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT3},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        answer = response.choices[0].message.content
        
        if answer is None:
            answer = ""
            warning_msg = "Warning: LLM returned None/empty response"
            if logger:
                logger.warning(warning_msg)
            else:
                print(warning_msg)
        else:
            answer = answer.strip()
        
        message = f"Response generated successfully ({len(answer)} characters)"
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
        
        import traceback
        traceback.print_exc()
        raise

