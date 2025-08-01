import re
from typing import List

def format_response(answer: str, reasoning: str, relevant_clauses: List[str], confidence: str = "Medium") -> str:
    """
    Extracts the complete 'Direct Answer' content from the verbose LLM response, without minimizing or trimming it.
    """
    match = re.search(r"\*\*Direct Answer\*\*[:\s]*([\s\S]*?)(?=\n\*\*|$)", answer, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback if Direct Answer not found
    return answer.strip()
