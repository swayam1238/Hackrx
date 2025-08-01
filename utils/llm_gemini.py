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
    # Adaptive chunk processing based on total content size
    limited_chunks = []
    total_length = 0
    max_chunk_length = 300  # Increased for better context
    
    # Calculate total available context
    total_chunks_length = sum(len(chunk) for chunk in top_chunks)
    
    # Adaptive context limit based on content size
    if total_chunks_length > 5000:  # Large content
        context_limit = 2500
    elif total_chunks_length > 2000:  # Medium content
        context_limit = 2000
    else:  # Small content
        context_limit = 1500
    
    for chunk in top_chunks:
        if total_length + len(chunk) < context_limit:
            # Use more of each chunk for large files
            chunk_to_use = chunk[:max_chunk_length] if len(chunk) > max_chunk_length else chunk
            limited_chunks.append(chunk_to_use)
            total_length += len(chunk_to_use)
        else:
            break
    
    prompt = f"""Answer this question based on the provided policy clauses:

Question: {question}

Clauses:
{chr(10).join(limited_chunks)}

Provide a direct, concise answer. If information is not in the clauses, state "Not specified"."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a policy analyst. Give direct, concise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # More deterministic
            max_tokens=400,   # Increased for better answers
            timeout=45        # Increased timeout for large files
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract reasoning and confidence from the response
        reasoning = response_text
        confidence = "High" if "clause" in response_text.lower() or "section" in response_text.lower() else "Medium"
        
        # Try to extract relevant clauses from the response
        relevant_clauses = []
        if "clause" in response_text.lower():
            import re
            clause_matches = re.findall(r'clause\s+\d+\.?\d*', response_text.lower())
            relevant_clauses.extend(clause_matches)
        if "section" in response_text.lower():
            import re
            section_matches = re.findall(r'section\s+\d+\.?\d*', response_text.lower())
            relevant_clauses.extend(section_matches)
        
        # If no specific clauses found, provide a general reference
        if not relevant_clauses:
            relevant_clauses = ["Policy clauses"]
        
        return (
            response_text,
            reasoning,
            relevant_clauses,
            confidence
        )
            
    except Exception as e:
        return (
            f"❌ Error generating answer: {str(e)}",
            "Error occurred during processing",
            [],
            "Low"
        )
