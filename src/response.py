
from openai import OpenAI

from src.config import get_secrets

SYSTEM_PROMPT = """
You are an assistant that provides accurate, answers about a Dungeons & Dragons campaign using only the information found in the supplied session summary sections
Each section is labeled, showing both the session number and the part number (e.g., "part 1/2/3"), conveying order within a session.
This context is in form of a bullet point lists, conveying a hierarchy within a given event.
You must:
- First translate any context from german to english.
- Produce a concise answer to the query based on the provided sources.
- Player characters are called: Sai, Selene, Eve, Melli, Kenshi, Vorak, and Aurora.
- not mention the Session part number in your answer

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
    return f"SESSION NOTES:\n{context}\n\n---\n\nQUESTION: {question}\n\nANSWER:"

def generate_response(question, retrieved_chunks, config, verbose=False):
    model = config.get('response', {}).get('model')
    temperature = config.get('response', {}).get('temperature')
    max_tokens = config.get('response', {}).get('max_tokens')
    
    print(f"Generating response using {model}...")
    
    context = format_context(retrieved_chunks)
    user_prompt = build_user_prompt(question, context)
    client = create_llm_client(config)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        if answer is None:
            answer = ""
            print("Warning: LLM returned None/empty response")
        else:
            answer = answer.strip()
        
        
        print(f"Response generated (input: {input_tokens} tokens, output: {output_tokens} tokens)")
        
        return {
            'answer': answer,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model': model,
        }
        
    except Exception as e:
        print(f"Error: Error generating response: {e}")
        
        import traceback
        traceback.print_exc()
        raise
