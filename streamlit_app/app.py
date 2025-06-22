import streamlit as st
import os
import time
import uuid
from pathlib import Path
import PyPDF2
import re
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pandas RAG Assistant",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: transparent;
    }
    .user-message {
        border-left: 4px solid #2196f3;
        background-color: transparent;
    }
    .assistant-message {
        border-left: 4px solid #4caf50;
        background-color: transparent;
    }
    .relevance-score {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Working functions from our notebooks
@st.cache_resource
def initialize_tokenizer():
    """Initialize tokenizer with caching"""
    return tiktoken.get_encoding("cl100k_base")

@st.cache_resource
def initialize_embedding_model():
    """Initialize embedding model with caching"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def initialize_qdrant_client():
    """Initialize Qdrant client with caching"""
    try:
        client = QdrantClient("localhost", port=6333)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return None

def count_tokens(text, tokenizer):
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def clean_text(text):
    """Basic text cleaning"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def fixed_enhanced_cleaning(text):
    """Enhanced cleaning for pandas documentation"""
    text = clean_text(text)
    
    # Fix specific PDF artifacts
    text = re.sub(r'\bwher\s+e\b', 'where', text)
    text = re.sub(r'\btransfor\s+ms\b', 'transforms', text)
    text = re.sub(r'\bcomp\s+lex\b', 'complex', text)
    text = re.sub(r'\boper\s+ation\s+s\b', 'operations', text)
    text = re.sub(r'\bData\s+Frame\b', 'DataFrame', text)
    text = re.sub(r'\bgroup\s+by\b', 'groupby', text, flags=re.IGNORECASE)
    
    return text

def detect_code_blocks(text):
    """Detect if text contains code examples"""
    code_patterns = [
        r'import\s+\w+', r'pd\.\w+', r'df\.\w+', r'print\s*\(',
        r'=\s*pd\.', r'\.groupby\(', r'\.merge\(', r'\.iloc\[', r'\.loc\['
    ]
    code_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in code_patterns)
    return code_score > 2

def robust_text_splitting(text):
    """Split text using multiple strategies"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) < 3:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        paragraphs = []
        current_para = ""
        
        for line in lines:
            if line.endswith(('.', '!', '?', ':')) or len(current_para) > 300:
                current_para += " " + line if current_para else line
                if len(current_para.split()) > 20:
                    paragraphs.append(current_para)
                    current_para = ""
            else:
                current_para += " " + line if current_para else line
        
        if current_para:
            paragraphs.append(current_para)
    
    if len(paragraphs) < 2:
        sentences = re.split(r'[.!?]+\s+', text)
        paragraphs = []
        current_para = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if count_tokens(current_para + " " + sentence, st.session_state.tokenizer) > 200:
                if current_para:
                    paragraphs.append(current_para)
                current_para = sentence
            else:
                current_para += " " + sentence if current_para else sentence
        
        if current_para:
            paragraphs.append(current_para)
    
    return paragraphs

def working_chunking_strategy(text, target_size=1000, min_size=400):
    """Final working chunking strategy"""
    cleaned_text = fixed_enhanced_cleaning(text)
    segments = robust_text_splitting(cleaned_text)
    
    chunks = []
    current_chunk = ""
    
    for segment in segments:
        current_tokens = count_tokens(current_chunk, st.session_state.tokenizer)
        segment_tokens = count_tokens(segment, st.session_state.tokenizer)
        
        if current_tokens + segment_tokens > target_size and current_tokens >= min_size:
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'has_code': detect_code_blocks(current_chunk)
            })
            current_chunk = segment
        else:
            current_chunk += "\n\n" + segment if current_chunk else segment
    
    if current_chunk and count_tokens(current_chunk, st.session_state.tokenizer) >= min_size:
        chunks.append({
            'text': current_chunk.strip(),
            'token_count': count_tokens(current_chunk, st.session_state.tokenizer),
            'has_code': detect_code_blocks(current_chunk)
        })
    
    return chunks

def extract_text_from_pdf(pdf_path, start_page=11, end_page=50):
    """Extract text from PDF pages"""
    all_text = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        end_page = min(end_page, len(pdf_reader.pages))
        
        for page_num in range(start_page, end_page):
            try:
                text = pdf_reader.pages[page_num].extract_text()
                if text.strip():
                    all_text.append({'page': page_num, 'raw_text': text})
            except Exception as e:
                st.error(f"Error extracting page {page_num}: {e}")
    
    return all_text

def create_rag_prompt(query, retrieved_chunks, context_limit=3):
    """Create a well-structured RAG prompt"""
    context_chunks = retrieved_chunks[:context_limit]
    
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n--- Context {i} (Relevance: {chunk.score:.3f}) ---\n"
        context_text += chunk.payload['text'][:1000]
        if len(chunk.payload['text']) > 1000:
            context_text += "...\n"
    
    prompt = f"""You are an expert pandas assistant. Use the provided context to answer the user's question about pandas data analysis.

CONTEXT:
{context_text}

INSTRUCTIONS:
- Answer based primarily on the provided context
- Include relevant code examples when available
- If context doesn't fully answer the question, acknowledge this but provide helpful guidance
- Be concise but comprehensive
- Format code examples clearly

USER QUESTION: {query}

HELPFUL ANSWER:"""

    return prompt

def process_query(question, collection_name="pandas_fundamentals", top_k=3):
    """Complete RAG pipeline"""
    
    if not st.session_state.get('qdrant_client'):
        return None, "Database not connected. Please check Qdrant connection."
    
    if not st.session_state.get('groq_client'):
        return None, "LLM not connected. Please check Groq API key."
    
    try:
        # Retrieve
        query_embedding = st.session_state.embedding_model.encode(question)
        
        search_results = st.session_state.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=top_k
        )
        
        retrieved_chunks = search_results.points
        
        if not retrieved_chunks:
            return None, "No relevant content found. Please try a different question."
        
        # Generate
        prompt = create_rag_prompt(question, retrieved_chunks, context_limit=top_k)
        
        response = st.session_state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=1500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "chunks": retrieved_chunks,
            "avg_relevance": sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
        }, None
        
    except Exception as e:
        return None, f"Error processing query: {str(e)}"

def initialize_system():
    """Initialize all system components"""
    
    with st.spinner("Initializing RAG system..."):
        
        # Initialize models
        if 'tokenizer' not in st.session_state:
            st.session_state.tokenizer = initialize_tokenizer()
        
        if 'embedding_model' not in st.session_state:
            st.session_state.embedding_model = initialize_embedding_model()
        
        if 'qdrant_client' not in st.session_state:
            st.session_state.qdrant_client = initialize_qdrant_client()
        
        # Initialize Groq
        if 'groq_client' not in st.session_state:
            groq_api_key = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")
            if groq_api_key:
                try:
                    st.session_state.groq_client = Groq(api_key=groq_api_key)
                except Exception as e:
                    st.error(f"Failed to initialize Groq: {e}")
                    st.session_state.groq_client = None
            else:
                st.session_state.groq_client = None
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🐼 Pandas RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("*Your intelligent assistant for pandas data analysis questions*")
    
    # Initialize system
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # System status checks
        tokenizer_status = "✅" if st.session_state.get('tokenizer') else "❌"
        embedding_status = "✅" if st.session_state.get('embedding_model') else "❌"
        qdrant_status = "✅" if st.session_state.get('qdrant_client') else "❌"
        groq_status = "✅" if st.session_state.get('groq_client') else "❌"
        
        st.write(f"**Tokenizer:** {tokenizer_status}")
        st.write(f"**Embeddings:** {embedding_status}")
        st.write(f"**Vector DB:** {qdrant_status}")
        st.write(f"**LLM:** {groq_status}")
        
        system_ready = all([
            st.session_state.get('tokenizer'),
            st.session_state.get('embedding_model'),
            st.session_state.get('qdrant_client'),
            st.session_state.get('groq_client')
        ])
        
        if system_ready:
            st.success("🚀 System Ready!")
        else:
            st.error("⚠️ System Not Ready")
            if not st.session_state.get('groq_client'):
                groq_key = st.text_input("Enter Groq API Key:", type="password")
                if groq_key:
                    try:
                        st.session_state.groq_client = Groq(api_key=groq_key)
                        st.success("Groq client initialized!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Invalid API key: {e}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        top_k = st.slider("Results to retrieve:", 1, 5, 3)
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "What is pandas?",
            "How to create a DataFrame?",
            "How to read CSV files?",
            "What is groupby in pandas?",
            "How to handle missing data?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.current_question = q
        
        st.divider()
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Pandas Expert")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {chat['answer']}
            <div class="relevance-score">
                Relevance Score: {chat.get('relevance', 0):.3f} | 
                Sources: {len(chat.get('chunks', []))} chunks
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Query input
    question = st.text_input(
        "Ask a pandas question:",
        value=st.session_state.get('current_question', ''),
        placeholder="e.g., How do I merge two DataFrames?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    
    if ask_button and question and system_ready:
        
        with st.spinner("Processing your question..."):
            result, error = process_query(question, top_k=top_k)
        
        if error:
            st.markdown(f"""
            <div class="error-message">
                <strong>Error:</strong> {error}
            </div>
            """, unsafe_allow_html=True)
        
        elif result:
            # Add to chat history
            chat_entry = {
                'question': question,
                'answer': result['answer'],
                'relevance': result['avg_relevance'],
                'chunks': result['chunks']
            }
            st.session_state.chat_history.append(chat_entry)
            
            # Clear current question
            if 'current_question' in st.session_state:
                del st.session_state.current_question
            
            st.rerun()
    
    elif ask_button and not system_ready:
        st.error("System not ready. Please check the sidebar for missing components.")
    
    elif ask_button and not question:
        st.warning("Please enter a question.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Powered by Pandas Documentation • Qdrant Vector Database • Groq LLM
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()