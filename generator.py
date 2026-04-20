import os
from openai import OpenAI

# Setup API Key for Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
) if GROQ_API_KEY else None

MODEL_NAME = 'llama-3.1-8b-instant'

PROMPT_TEMPLATE = """You are the "Student Life Assistant for Rutgers Business School".
Your task is to answer the user's question using ONLY the provided contextual documents below. 

Instructions:
1. Try to answer the question using ONLY the knowledge provided in the Context.
2. If the Context DOES NOT contain the answer (e.g. if asked about an event that isn't listed, or a major that isn't mentioned), EXPLICITLY state: "I don't have information about that in my current database." Do NOT hallucinate.
3. Keep the answer concise and direct.
4. MUST INCLUDE CITATIONS: Cite the source URL or File that your answer came from at the end of the sentence or block, exactly as [Source: <url/file>].

Context:
{context_str}

User Question:
{query}
"""

class RAGGenerator:
    def __init__(self):
        self.model_name = MODEL_NAME

    def generate_answer(self, query, retrieved_chunks):
        if not client:
            return "Please set the OPENAI_API_KEY environment variable to enable answer generation."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"--- Document {i+1} ---\n{chunk['metadata_prefix']}{chunk['text']}\n")
        
        context_str = "\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query=query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during generation: {e}"

if __name__ == "__main__":
    print("Run app.py to interact with the generator.")
