import re
import google.generativeai as genai

# --- Configuration ---
# WARNING: Hardcoding API keys is a major security risk. 
# It's highly recommended to use environment variables instead.
genai.configure(api_key="AIzaSyDg6eoWSyMxwzycFHbnGRll9xRP_777PyY")

# The best model for this reasoning task is Gemini 1.5 Pro.
#MODEL_NAME = "gemini-1.5-pro-latest"
# For a faster, more economical option, use Gemini 1.5 Flash.
MODEL_NAME = "gemini-1.5-flash-latest"

# Configure the model generation parameters
generation_config = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_output_tokens": 1024,
}

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config
)


def create_policy_prompt(question: str, retrieved_chunks: list[str]) -> str:
    """
    Creates a high-precision prompt for policy Q&A using the Chain-of-Thought method.
    """
    context_str = "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks))

    # This prompt is well-structured for powerful models like Gemini 1.5 Pro
    prompt = f"""You are a specialized AI assistant acting as a meticulous Insurance Policy Analyst.
Your task is to analyze the provided policy clauses and answer the user's question with precision.

<PolicyClauses>
{context_str}
</PolicyClauses>

<Question>
{question}
</Question>

Follow these instructions precisely:
1.  First, inside a <thinking> block, break down your process. Identify key terms in the question. Scan the <PolicyClauses> to locate the exact text that addresses these terms. Extract the relevant facts and clause numbers.
2.  After your thinking process, provide the final, concise answer directly, with no preamble.
3.  Your answer must be derived *only* from the text in <PolicyClauses>.
4.  If the information is not present, you must state that the answer is not available in the provided text.

<thinking>
(Your reasoning process to find the answer goes here)
</thinking>

(Your final, direct answer goes here)
"""
    return prompt

def ask_question(question: str, top_chunks: list[str]):
    """
    Asks the question to the Gemini model and parses the structured response.
    """
    # --- Chunk Curation ---
    limited_chunks = []
    total_length = 0
    context_limit = 2500  # Simplified limit for clarity

    for chunk in top_chunks:
        if total_length + len(chunk) < context_limit:
            limited_chunks.append(chunk)
            total_length += len(chunk)
        else:
            break
    
    # --- Create the improved prompt ---
    prompt = create_policy_prompt(question, limited_chunks)

    try:
        # --- Call the Gemini API ---
        response = model.generate_content(prompt)

        # Access the response text
        raw_model_output = response.text.strip()

        # --- Post-process the structured <thinking> output (This logic remains the same) ---
        thinking_part = "No <thinking> block found."
        final_answer = raw_model_output

        if "</thinking>" in raw_model_output:
            parts = raw_model_output.split("</thinking>")
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_model_output, re.DOTALL)
            if thinking_match:
                thinking_part = thinking_match.group(1).strip()
            
            final_answer = parts[-1].strip()

        # Extract clauses and determine confidence from the reasoning
        relevant_clauses = re.findall(r'clause|section|\[\d+\]', thinking_part.lower())
        confidence = "High" if relevant_clauses else "Medium"
        
        # Clean up the list of clauses
        found_clauses = list(set(re.findall(r'\[\d+\]', thinking_part))) or ["Not specified"]

        return (
            final_answer,
            thinking_part, # The reasoning from the model
            found_clauses, # More precise clause extraction
            confidence
        )

    except Exception as e:
        return (
            f"❌ Error generating answer: {str(e)}",
            "Error occurred during processing",
            [],
            "Low"
        )