import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Groq-compatible OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

MODEL_NAME = "llama3-70b-8192"  # or try mixtral-8x7b-32768 for Mixture of Experts

def ask_question(question, top_chunks):
    prompt = f"""You are a policy analyst. Use the below clauses to answer the question precisely and with reasoning.\n\nRelevant Clauses:\n{chr(10).join(top_chunks)}\n\nQuestion:\n{question}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful policy analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
