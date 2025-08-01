import re
import json
from typing import Dict, Any, List

def format_response(answer: str, reasoning: str, relevant_clauses: List[str], confidence: str = "Medium") -> str:
    """
    Format the response to provide clean, direct answers as expected by the API
    """
    # Clean up the answer - remove any JSON formatting
    clean_answer = answer.strip()
    
    # If the answer contains JSON-like structure, extract the actual answer
    if '"answer"' in clean_answer and clean_answer.startswith('{'):
        try:
            # Try to parse and extract just the answer field
            parsed = json.loads(clean_answer)
            if isinstance(parsed, dict) and 'answer' in parsed:
                clean_answer = parsed['answer']
        except:
            # If parsing fails, try to extract answer manually
            import re
            answer_match = re.search(r'"answer":\s*"([^"]+)"', clean_answer)
            if answer_match:
                clean_answer = answer_match.group(1)
    
    # Return just the clean answer as a string (not JSON)
    return clean_answer