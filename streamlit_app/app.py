import streamlit as st
import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv
import re
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pandas RAG Assistant",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #262730;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        font-weight: 400;
    }
    
    /* Status indicators */
    .status-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .status-success {
        background-color: #238636;
        color: #ffffff;
    }
    
    .status-error {
        background-color: #da3633;
        color: #ffffff;
    }
    
    .status-warning {
        background-color: #bf8700;
        color: #ffffff;
    }
    
    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-title {
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Quick action buttons */
    .quick-button {
        width: 100%;
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        color: #fafafa;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .quick-button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
    }
    
    /* Input styling */
    .stTextInput input {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #fafafa;
    }
    
    .stTextInput input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #238636;
        border: 1px solid #238636;
        border-radius: 6px;
        color: #ffffff;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #2ea043;
        border-color: #2ea043;
    }
    
    /* Chat styling */
    .chat-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    
    .user-message {
        background-color: #0d1117;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #161b22;
        margin-right: 2rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Override Streamlit's default colors */
    .stMarkdown, .stText, p, div, span {
        color: #fafafa;
    }
    
    /* Sidebar specific overrides */
    .css-1d391kg .stMarkdown {
        color: #fafafa;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stInfo, .stWarning {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Load system components
@st.cache_resource
def initialize_rag_system():
    """Initialize the optimized RAG system"""
    try:
        PROJECT_ROOT = Path.cwd()
        if 'streamlit_app' in str(PROJECT_ROOT):
            PROJECT_ROOT = PROJECT_ROOT.parent
        
        PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
        
        # Load evaluation data
        with open(PROCESSED_DIR / 'improved_evaluation.pkl', 'rb') as f:
            eval_data = pickle.load(f)
        
        # Initialize components
        embedding_model = SentenceTransformer(eval_data['configuration']['embedding_model'])
        qdrant_client = QdrantClient("localhost", port=6333)
        
        return {
            'embedding_model': embedding_model,
            'qdrant_client': qdrant_client,
            'collection_name': "pandas_docs_optimized",
            'eval_data': eval_data,
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def preprocess_query(query):
    """Enhanced query preprocessing"""
    normalizations = {
        'dataframe': 'DataFrame',
        'data frame': 'DataFrame',
        'series': 'Series',
        'groupby': 'group by aggregation',
        'concat': 'concatenate combine'
    }
    
    processed = query.lower()
    for wrong, correct in normalizations.items():
        processed = processed.replace(wrong, correct)
    
    if 'pandas' not in processed and any(term in processed for term in ['DataFrame', 'Series', 'csv']):
        processed = f"pandas {processed}"
    
    return processed

def process_question(question, system, groq_client):
    """Complete RAG pipeline processing"""
    try:
        # Preprocess and embed
        processed_query = preprocess_query(question)
        query_embedding = system['embedding_model'].encode(processed_query)
        
        # Vector search
        results = system['qdrant_client'].query_points(
            collection_name=system['collection_name'],
            query=query_embedding.tolist(),
            limit=3
        )
        
        if not results.points:
            return "I couldn't find relevant information for your question.", 0.0, []
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(results.points, 1):
            context_parts.append(f"""
Context {i} (Relevance: {chunk.score:.3f}):
{chunk.payload['text'][:1200]}
""")
        
        context = "\n".join(context_parts)
        
        # Generate response
        prompt = f"""You are a pandas expert assistant. Answer the user's question using the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Base your answer on the provided context
- Include code examples when available
- Be clear and concise
- If context is limited, acknowledge but still provide helpful guidance

QUESTION: {question}

ANSWER:"""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=1200,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        relevance = np.mean([r.score for r in results.points])
        sources = [f"Pages {r.payload['source_pages']}" for r in results.points]
        
        return answer, relevance, sources
        
    except Exception as e:
        return f"Error processing question: {str(e)}", 0.0, []

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🐼 Pandas RAG Assistant</div>
        <div class="app-subtitle">Your intelligent companion for pandas data analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    system = initialize_rag_system()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = None
    
    # Sidebar
    with st.sidebar:
        # System status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🔧 System Status</div>', unsafe_allow_html=True)
        
        if system['status'] == 'success':
            st.markdown("""
            <div class="status-container">
                <div class="status-badge status-success">✅ RAG System Online</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance metrics
            eval_data = system['eval_data']
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{eval_data['metrics']['good_plus_rate']:.0%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{eval_data['metrics']['avg_score']:.2f}</div>
                    <div class="metric-label">Avg Score</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-container">
                <div class="status-badge status-error">❌ System Error</div>
                <small>{system.get('error', 'Unknown error')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Configuration
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🤖 API Setup</div>', unsafe_allow_html=True)
        
        if not st.session_state.groq_client:
            groq_key = st.text_input("Groq API Key:", type="password", key="groq_key")
            if groq_key:
                try:
                    st.session_state.groq_client = Groq(api_key=groq_key)
                    st.success("Connected to Groq!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        else:
            st.markdown("""
            <div class="status-container">
                <div class="status-badge status-success">✅ Groq Connected</div>
                <small>Model: llama-3.1-8b-instant</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick questions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚡ Quick Start</div>', unsafe_allow_html=True)
        
        quick_questions = [
            "What is a DataFrame?",
            "How to read CSV files?",
            "Using loc vs iloc?",
            "GroupBy operations?",
            "Handle missing data?",
            "Merge DataFrames?"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Controls
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚙️ Controls</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                st.caption(f"Relevance: {metadata['relevance']:.3f} | Sources: {', '.join(metadata['sources'])}")
    
    # Chat input
    system_ready = system['status'] == 'success' and st.session_state.groq_client is not None
    
    if prompt := st.chat_input("Ask me anything about pandas...", disabled=not system_ready):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if not system_ready:
                error_msg = "Please check system status and API connection in the sidebar."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                with st.spinner("Thinking..."):
                    answer, relevance, sources = process_question(prompt, system, st.session_state.groq_client)
                
                st.markdown(answer)
                st.caption(f"Relevance: {relevance:.3f} | Sources: {', '.join(sources)}")
                
                # Add assistant message with metadata
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": {"relevance": relevance, "sources": sources}
                })
    
    # System requirements notice
    if not system_ready:
        if system['status'] != 'success':
            st.error("⚠️ RAG system not loaded. Please check if all components are available.")
        elif not st.session_state.groq_client:
            st.info("ℹ️ Please enter your Groq API key in the sidebar to start chatting.")

if __name__ == "__main__":
    main()