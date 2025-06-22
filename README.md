1. Document Processing Pipeline

PDF (473 pages) 
    ↓
Text Extraction (PyPDF2)
    ↓
Text Cleaning (Fix PDF artifacts)
    ↓  
Smart Text Splitting (Multiple strategies)
    ↓
Semantic Chunking (Target: 1000 tokens)
    ↓
Code Detection (Identify technical content)


2. Embedding & Storage Pipeline

Text Chunks
    ↓
Generate Embeddings (sentence-transformers)
    ↓
Create Vector Points (with metadata)
    ↓
Store in Qdrant (with unique UUIDs)
    ↓
Index for Fast Retrieval (HNSW algorithm)


3. RAG Query Pipeline

User Question
    ↓
Generate Query Embedding
    ↓
Vector Similarity Search (Qdrant)
    ↓
Retrieve Top-K Chunks (with scores)
    ↓
Create Context Prompt
    ↓
LLM Generation (Groq)
    ↓
Formatted Response