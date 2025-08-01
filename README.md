# HackRx LLM-Powered Intelligent Query–Retrieval System

## Overview

An intelligent document processing system that can analyze large documents (PDFs, DOCX, emails) and provide contextual answers to complex queries in insurance, legal, HR, and compliance domains.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Docs    │───▶│  LLM Parser     │───▶│ Embedding Search│
│  (PDF/DOCX/EML) │    │                 │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  JSON Output    │◀───│ Logic Evaluation│◀───│ Clause Matching │
│ (Structured)    │    │                 │    │ (Semantic)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### ✅ Implemented Features

1. **Multi-Format Document Processing**
   - PDF documents (using PyPDF2)
   - DOCX files (using python-docx)
   - Email files (.eml format)

2. **Semantic Chunking**
   - Intelligent text segmentation based on sentences and paragraphs
   - Overlapping chunks for better context preservation
   - Configurable chunk sizes (default: 1000 chars with 200 char overlap)

3. **Advanced Embedding Search**
   - FAISS vector database for fast similarity search
   - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
   - Top-k retrieval (default: 3 most relevant chunks)

4. **LLM-Powered Analysis**
   - Groq integration with Llama3-70b model
   - Structured JSON responses with reasoning
   - Confidence scoring and clause traceability

5. **Explainable AI**
   - Step-by-step reasoning for each answer
   - Relevant clause identification
   - Confidence level assessment
   - Metadata tracking

6. **RESTful API**
   - FastAPI backend with automatic documentation
   - Bearer token authentication
   - Structured request/response format

## API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
```
Authorization: Bearer 3dea41115332ec6960807ebc546a0244e3bf91529888d3949b484cd22f0de72f
```

### Endpoints

#### POST `/api/v1/hackrx/run`
Process documents and answer questions.

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response Format:**
```json
{
    "answers": [
        {
            "answer": "Direct answer to the question",
            "reasoning": "Step-by-step reasoning process",
            "relevant_clauses": ["List of supporting clauses"],
            "confidence": "High/Medium/Low",
            "metadata": {
                "clause_count": 3,
                "has_reasoning": true,
                "response_type": "structured"
            }
        }
    ]
}
```

## Technical Specifications

### Tech Stack
- **Backend**: FastAPI
- **Vector DB**: FAISS
- **LLM**: Groq (Llama3-70b-8192)
- **Embeddings**: Sentence Transformers
- **Document Processing**: PyPDF2, python-docx

### Performance Optimizations
- Semantic chunking for better context preservation
- FAISS for fast similarity search
- Token-efficient prompts
- Structured responses for better parsing

### Evaluation Parameters

#### Accuracy
- Precision of query understanding through semantic search
- Clause matching accuracy using FAISS similarity

#### Token Efficiency
- Optimized prompts with clear instructions
- Structured output to reduce token usage
- Efficient chunking strategy

#### Latency
- Fast FAISS similarity search
- Optimized embedding model (all-MiniLM-L6-v2)
- Efficient document processing pipeline

#### Reusability
- Modular architecture with separate utility modules
- Configurable parameters for different use cases
- Extensible design for new document types

#### Explainability
- Structured reasoning for each answer
- Clause traceability with specific references
- Confidence scoring system
- Metadata tracking for transparency

## Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Hackrx
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file with your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

4. **Run the application**
```bash
uvicorn main:app --reload --port 8000
```

## Usage Examples

### Python Client
```python
import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 3dea41115332ec6960807ebc546a0244e3bf91529888d3949b484cd22f0de72f"
}

data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
}

response = requests.post("http://localhost:8000/api/v1/hackrx/run", json=data, headers=headers)
print(response.json())
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 3dea41115332ec6960807ebc546a0244e3bf91529888d3949b484cd22f0de72f" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## System Components

### 1. Document Parser (`utils/parser.py`)
- Multi-format document processing
- Semantic chunking algorithm
- Text extraction and preprocessing

### 2. Embedding Engine (`utils/embedder.py`)
- FAISS vector database integration
- Sentence Transformers for embeddings
- Similarity search functionality

### 3. LLM Integration (`utils/llm_gemini.py`)
- Groq API integration
- Structured prompt engineering
- JSON response parsing

### 4. Response Formatter (`utils/formatter.py`)
- Structured JSON output
- Metadata enrichment
- Explainability formatting

## Scoring System

The system is designed to work with the HackRx scoring methodology:

- **Known Documents**: Lower weightage (0.5)
- **Unknown Documents**: Higher weightage (2.0)
- **Question-Level Weightage**: Variable based on complexity
- **Score Calculation**: Question Weight × Document Weight × Accuracy

## Future Enhancements

1. **Pinecone Integration**: For cloud-based vector storage
2. **PostgreSQL**: For structured data storage
3. **Advanced Caching**: Redis for performance optimization
4. **Batch Processing**: Handle multiple documents simultaneously
5. **Custom Models**: Fine-tuned models for specific domains

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the HackRx competition and follows the competition guidelines.