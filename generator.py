import os
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"

PROMPT_TEMPLATE = """You are the "Student Life Assistant for Rutgers Business School".
Your task is to answer the user's question using ONLY the contextual documents below.

Instructions:
1. Carefully scan all provided context chunks before answering.
2. If the answer appears in any chunk, answer using only that information.
3. If multiple chunks are relevant, combine them carefully and cite the supporting source(s).
4. If the answer is only partially available, provide the partial answer and clearly say what is missing.
5. Only say "I don't have information about that in my current database." if none of the chunks contain a relevant answer.
6. Do not hallucinate or use outside knowledge.
7. Keep the answer concise and direct.
8. MUST INCLUDE CITATIONS: Cite the source URL or file exactly as [Source: <url/file>] at the end of each answer paragraph or bullet.

Context:
{context_str}

User Question:
{query}
"""


class RAGGenerator:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_answer(self, query, retrieved_chunks):
        if not self.client:
            return "Please set the OPENAI_API_KEY environment variable to enable answer generation."

        if not retrieved_chunks:
            return "I don't have information about that in my current database."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            metadata_prefix = chunk.get("metadata_prefix", "").strip()
            text = chunk.get("text", "").strip()

            if text:
                context_parts.append(
                    f"--- Document {i+1} ---\n"
                    f"{metadata_prefix}\n"
                    f"{text}\n"
                )

        if not context_parts:
            return "I don't have information about that in my current database."

        context_str = "\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful retrieval-augmented assistant. Use only the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0
            )

            answer = response.choices[0].message.content

            if not answer or not answer.strip():
                return "I don't have information about that in my current database."

            return answer.strip()

        except Exception as e:
            return f"Error during generation: {e}"


if __name__ == "__main__":
    print("Run app.py to interact with the generator.")