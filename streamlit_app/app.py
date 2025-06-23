import streamlit as st
import qdrant_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import pickle
import json
import os
import gc
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pandas AI Tutor",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PROJECT ROOT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- ENHANCED DARK THEME & MESSAGING APP STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stDeployButton {
        display: none;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 120px !important;
        max-width: 1200px;
        background: #0d1117;
    }
    
    /* Sidebar Dark Theme */
    .stSidebar {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    .stSidebar .block-container {
        background: #161b22 !important;
        padding-top: 1rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Remove all white backgrounds */
    .stTabs [data-baseweb="tab-list"] {
        background: #21262d !important;
        border-radius: 8px;
        padding: 4px;
        border: 1px solid #30363d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px;
        color: #8b949e;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
    }
    
    /* MESSAGING APP STYLE CHAT - FIXED BOTTOM INPUT */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        margin: 0.25rem 0 !important;
    }
    
    /* Chat Container with fixed height and scroll */
    .main .block-container .element-container:has(.stChatMessage) {
        height: 60vh !important;
        overflow-y: auto !important;
        margin-bottom: 80px !important;
        padding-bottom: 1rem !important;
    }
    
    /* Fixed Chat Input at Bottom */
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: #161b22 !important;
        border-top: 2px solid #30363d !important;
        padding: 1rem !important;
        z-index: 999 !important;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatInput {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    .stChatInput > div > div > div > div {
        background: #0d1117 !important;
        border: 2px solid #30363d !important;
        border-radius: 25px !important;
        color: #e6edf3 !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
    }
    
    .stChatInput > div > div > div > div:focus-within {
        border-color: #2ea043 !important;
        box-shadow: 0 0 0 3px rgba(46, 160, 67, 0.2) !important;
    }
    
    /* User Messages (Right aligned, blue) */
    .stChatMessage[data-testid="chat-message-user"] {
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 0 !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] .stMarkdown {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: white !important;
        padding: 12px 16px !important;
        border-radius: 20px 20px 6px 20px !important;
        max-width: 70% !important;
        margin-left: auto !important;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3) !important;
        font-size: 15px !important;
    }
    
    /* Assistant Messages (Left aligned, dark) */
    .stChatMessage[data-testid="chat-message-assistant"] {
        display: flex;
        justify-content: flex-start;
        margin: 0.5rem 0 !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] .stMarkdown {
        background: #21262d !important;
        color: #e6edf3 !important;
        padding: 12px 16px !important;
        border-radius: 20px 20px 20px 6px !important;
        max-width: 85% !important;
        margin-right: auto !important;
        border: 1px solid #30363d !important;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        font-size: 15px !important;
    }
    
    /* Typing indicator style */
    .stSpinner {
        position: fixed !important;
        bottom: 90px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        background: #21262d !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        border: 1px solid #30363d !important;
        z-index: 998 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #56d364 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3) !important;
    }
    
    .stButton > button:disabled {
        background: #30363d !important;
        color: #8b949e !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stRadio > div > div {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2ea043 !important;
        box-shadow: 0 0 0 3px rgba(46, 160, 67, 0.1) !important;
    }
    
    /* Containers */
    .element-container div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    /* Metrics */
    .stMetric {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent !important;
    }
    
    /* Progress */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    
    .streamlit-expanderContent {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Alerts */
    .stSuccess {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #721c24 0%, #dc3545 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #055160 0%, #0dcaf0 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #664d03 0%, #ffc107 100%) !important;
        color: black !important;
        border: none !important;
    }
    
    /* Header Styling */
    .professional-header {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
        text-align: center;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
    }
    
    /* Remove any remaining white backgrounds */
    .st-emotion-cache-1r4qj8v, 
    .st-emotion-cache-1xw8zd6, 
    .st-emotion-cache-4oy321,
    .st-emotion-cache-16txtl3 {
        background: #161b22 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- RESOURCE LOADING & CACHING ---
@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY not found. Please add it to your .env file.")
        return None
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        return None

@st.cache_resource
def get_qdrant_client():
    """Initializes and returns the Qdrant client."""
    try:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        client = qdrant_client.QdrantClient(host=host, port=port)
        collections = client.get_collections()
        return client
    except Exception as e:
        st.error(f"🔌 Failed to connect to Qdrant at localhost:6333. Please ensure Qdrant is running.\n\n**Error:** {e}")
        return None

@st.cache_resource
def get_embedding_model(_config):
    """Loads and returns the SentenceTransformer model with memory optimization."""
    model_name = _config.get("vector_database_config", {}).get("embedding_model")
    if not model_name:
        st.error("🤖 Embedding model name not found in configuration.")
        return None
    try:
        gc.collect()
        model = SentenceTransformer(model_name)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load embedding model '{model_name}': {e}")
        st.info("💡 If you're getting memory errors, try restarting the application.")
        return None

# --- ALL DATA LOADING FUNCTIONS ---
@st.cache_data
def load_system_config():
    """Loads the main system configuration file."""
    config_path = PROJECT_ROOT / "data" / "processed" / "complete_rag_system_config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"📁 Configuration file not found at: `{config_path}`")
        return None

@st.cache_data
def load_enhanced_chunks():
    """Loads enhanced chunks with all metadata."""
    chunks_path = PROJECT_ROOT / "data" / "processed" / "enhanced_chunks_complete.pkl"
    try:
        with open(chunks_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"📦 Enhanced chunks not found at: `{chunks_path}`")
        return []

@st.cache_data
def load_quiz_questions():
    """Loads flat quiz questions list."""
    quiz_path = PROJECT_ROOT / "data" / "processed" / "generated_quiz_questions.pkl"
    try:
        with open(quiz_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"📝 Generated quiz questions not found. Trying quiz bank...")
        return []

@st.cache_data
def load_quiz_question_bank():
    """Loads hierarchical quiz question bank."""
    quiz_bank_path = PROJECT_ROOT / "data" / "processed" / "quiz_question_bank.pkl"
    try:
        with open(quiz_bank_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"📚 Quiz question bank not found at: `{quiz_bank_path}`")
        return {}

@st.cache_data
def load_specialized_collections():
    """Loads specialized chunk collections."""
    collections_path = PROJECT_ROOT / "data" / "processed" / "specialized_collections.pkl"
    try:
        with open(collections_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning(f"🎯 Specialized collections not found at: `{collections_path}`")
        return {}

@st.cache_data  
def load_system_summary():
    """Load comprehensive system summary."""
    summary_path = PROJECT_ROOT / "data" / "processed" / "comprehensive_system_summary.json"
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_quiz_generation_stats():
    """Load quiz generation statistics."""
    stats_path = PROJECT_ROOT / "data" / "processed" / "quiz_generation_statistics.json"
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_system_readiness_metrics():
    """Load system readiness metrics."""
    metrics_path = PROJECT_ROOT / "data" / "processed" / "system_readiness_metrics.json"
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# --- QUIZ PROCESSING FUNCTIONS ---
def flatten_quiz_bank(quiz_bank):
    """Convert hierarchical quiz bank to flat list with different question types."""
    flat_questions = []
    
    for quiz_type, categories in quiz_bank.items():
        if isinstance(categories, dict):
            for category, chunks in categories.items():
                for chunk in chunks:
                    # Create a proper quiz question from chunk
                    question = create_quiz_question_from_chunk(chunk, quiz_type, category)
                    if question:
                        flat_questions.append(question)
    
    return flat_questions

def create_quiz_question_from_chunk(chunk, quiz_type, category):
    """Create a quiz question from chunk content."""
    content = chunk.get('content', '')
    
    if quiz_type == 'multiple_choice':
        return create_multiple_choice_question(content, category)
    elif quiz_type == 'true_false':
        return create_true_false_question(content, category)
    elif quiz_type == 'fill_blank':
        return create_fill_blank_question(content, category)
    elif quiz_type == 'code_completion':
        return create_code_completion_question(content, category)
    elif quiz_type == 'scenario_based':
        return create_scenario_question(content, category)
    
    return None

def create_multiple_choice_question(content, category):
    """Create a multiple choice question from content."""
    # Extract key concepts from content
    if 'DataFrame' in content and 'merge' in content:
        return {
            'type': 'multiple_choice',
            'difficulty': category,
            'question': 'Which method is used to combine DataFrames based on common columns?',
            'options': ['concat()', 'merge()', 'join()', 'combine()'],
            'correct_answer': 1,
            'explanation': 'merge() is used to combine DataFrames based on common columns or indices.',
            'source_pages': 'Generated from content analysis'
        }
    elif 'groupby' in content:
        return {
            'type': 'multiple_choice',
            'difficulty': category,
            'question': 'What does the groupby() method return?',
            'options': ['DataFrame', 'Series', 'GroupBy object', 'List'],
            'correct_answer': 2,
            'explanation': 'groupby() returns a GroupBy object that can be used for aggregation operations.',
            'source_pages': 'Generated from content analysis'
        }
    return None

def create_true_false_question(content, category):
    """Create a true/false question from content."""
    if 'pandas' in content.lower():
        return {
            'type': 'true_false',
            'difficulty': category,
            'statement': 'pandas is built on top of NumPy arrays.',
            'correct_answer': True,
            'explanation': 'True. pandas uses NumPy arrays as the underlying data structure for better performance.',
            'source_pages': 'Generated from content analysis'
        }
    return None

def create_fill_blank_question(content, category):
    """Create a fill-in-the-blank question from content."""
    if 'DataFrame' in content:
        return {
            'type': 'fill_blank',
            'difficulty': category,
            'question': 'To create a new DataFrame, you use: pd._______(data)',
            'correct_answer': 'DataFrame',
            'explanation': 'pd.DataFrame() is the constructor for creating new DataFrame objects.',
            'source_pages': 'Generated from content analysis'
        }
    return None

def create_code_completion_question(content, category):
    """Create a code completion question from content."""
    if 'import' in content and 'pandas' in content:
        return {
            'type': 'code_completion',
            'difficulty': category,
            'question': 'Complete the code to import pandas:',
            'code_template': 'import _____ as pd',
            'correct_answer': 'pandas',
            'explanation': 'The standard way to import pandas is "import pandas as pd".',
            'source_pages': 'Generated from content analysis'
        }
    return None

def create_scenario_question(content, category):
    """Create a scenario-based question from content."""
    return {
        'type': 'scenario_based',
        'difficulty': category,
        'scenario': 'You have a large dataset and need to analyze it efficiently.',
        'question': 'What pandas method would you use to get a quick overview of your data?',
        'suggested_answer': 'df.describe() or df.info()',
        'explanation': 'describe() provides statistical summary, info() shows data types and memory usage.',
        'source_pages': 'Generated from content analysis'
    }

# --- RAG PIPELINE FUNCTIONS ---
def preprocess_pandas_query(query):
    """Enhanced query preprocessing for pandas-specific content."""
    normalizations = {
        'dataframe': 'DataFrame', 'data frame': 'DataFrame', 'series': 'Series',
        'groupby': 'group by aggregation', 'concat': 'concatenate combine DataFrames',
        'merge': 'join merge DataFrames'
    }
    processed_query = query.lower()
    for wrong, correct in normalizations.items():
        processed_query = processed_query.replace(wrong, correct)
    if 'pandas' not in processed_query and any(term in processed_query for term in ['DataFrame', 'Series', 'csv', 'data']):
        processed_query = f"pandas {processed_query}"
    return processed_query

def enhanced_retrieval(query, collection_name, q_client, embed_model, top_k=3):
    """Retrieves context from Qdrant using the preprocessed query."""
    if not q_client or not embed_model:
        return []
    
    processed_query = preprocess_pandas_query(query)
    query_embedding = embed_model.encode(processed_query)

    try:
        search_results = q_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        return search_results
    except Exception as e:
        st.error(f"🔍 Search error: {e}")
        return []

def generate_rag_response(query, context, llm_client, llm_model):
    """Generates a response from the LLM based on the query and context."""
    if not llm_client:
        return "❌ LLM Client is not available."

    system_prompt = """You are an expert pandas tutor with deep knowledge of data manipulation and analysis. 

Your responses should be:
- Accurate and practical with clear code examples
- Based strictly on the provided context
- Well-formatted with proper code blocks using ```python
- Include helpful tips and best practices
- Conversational and easy to understand

If the context doesn't contain the answer, honestly say you don't have enough information."""
    
    user_prompt = f"""Based on the following pandas documentation context, please answer the user's question comprehensively.

CONTEXT:
---
{context}
---

QUESTION: {query}

Please provide a detailed, practical answer with code examples if applicable."""

    try:
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=llm_model,
            temperature=0.1,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"🤖 LLM Error: {e}")
        return "Sorry, I encountered an error while generating a response."

# --- MAIN APP ---
def main():
    # --- HEADER ---
    st.markdown("""
    <div class="professional-header">
        <h1>🐼 Pandas AI Tutor</h1>
        <p style="font-size: 1.2rem; color: #8b949e; margin: 0;">Master pandas with AI-powered learning and interactive quizzes</p>
    </div>
    """, unsafe_allow_html=True)

    # --- LOAD ALL DATA AND MODELS ---
    config = load_system_config()
    enhanced_chunks = load_enhanced_chunks()
    quiz_questions = load_quiz_questions()
    quiz_bank = load_quiz_question_bank()
    specialized_collections = load_specialized_collections()
    system_summary = load_system_summary()
    quiz_stats = load_quiz_generation_stats()
    readiness_metrics = load_system_readiness_metrics()
    
    if not config:
        st.stop()

    # Generate quiz questions if needed
    if not quiz_questions and quiz_bank:
        st.info("🔄 Generating quiz questions from question bank...")
        quiz_questions = flatten_quiz_bank(quiz_bank)

    qdrant_client = get_qdrant_client()
    embedding_model = get_embedding_model(config)
    groq_client = get_groq_client()
    
    collection_name = "pandas_docs_enhanced_100pct"
    llm_model = config.get("llm_integration_config", {}).get("model_name", "llama-3.1-8b-instant")

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("### 🚀 Welcome!")
        st.markdown("Your AI-powered pandas assistant built on comprehensive content analysis.")
        
        st.markdown("---")
        
        st.markdown("### 🔧 System Status")
        
        status_items = [
            ("LLM Model", llm_model, "✅"),
            ("Embeddings", config.get('vector_database_config', {}).get('embedding_model', 'Unknown'), "✅"),
            ("Knowledge Base", collection_name, "✅"),
            ("Qdrant", "Connected" if qdrant_client else "Disconnected", "✅" if qdrant_client else "❌"),
            ("Groq API", "Connected" if groq_client else "Disconnected", "✅" if groq_client else "❌")
        ]
        
        for label, value, status in status_items:
            st.markdown(f"**{label}:** {status}")

        # Show comprehensive system stats
        if system_summary:
            st.markdown("---")
            st.markdown("### 📊 System Performance")
            
            quiz_caps = system_summary.get('quiz_capabilities', {})
            retrieval_caps = system_summary.get('retrieval_capabilities', {})
            overview = system_summary.get('system_overview', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", f"{overview.get('total_chunks', 0)}")
                st.metric("Quiz Ready", f"{quiz_caps.get('quiz_ready_chunks', 0)}")
            
            with col2:
                st.metric("High Quality", f"{retrieval_caps.get('high_quality_chunks', 0)}")
                st.metric("Code Examples", f"{retrieval_caps.get('code_example_chunks', 0)}")
            
            if overview.get('chunk_improvement_factor'):
                st.success(f"🚀 **{overview['chunk_improvement_factor']}x** improvement!")

        # Show loaded data stats
        st.markdown("---")
        st.markdown("### 📁 Loaded Data")
        st.markdown(f"**Enhanced Chunks:** {len(enhanced_chunks)}")
        st.markdown(f"**Quiz Questions:** {len(quiz_questions)}")
        st.markdown(f"**Quiz Bank Types:** {len(quiz_bank) if quiz_bank else 0}")
        st.markdown(f"**Collections:** {len(specialized_collections) if specialized_collections else 0}")

    # --- MAIN CONTENT TABS ---
    tab1, tab2 = st.tabs(["💬 **AI Chat Assistant**", "🧠 **Knowledge Quiz**"])

    # --- CHAT TAB ---
    with tab1:
        st.markdown("### 💬 Chat with your pandas AI tutor")
        st.markdown("Get instant, context-aware answers with code examples.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Hi! I'm your pandas AI tutor. Ask me anything about data manipulation, analysis, or pandas best practices!"}
            ]

        # Create chat container with fixed height and scrolling
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages in reverse order (newest at top)
            for message in reversed(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("📚 View Sources", expanded=False):
                            st.markdown(message["sources"])

        # Fixed chat input at bottom
        if prompt := st.chat_input("💭 Ask me about pandas... (e.g., How do I merge DataFrames?)"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate assistant response
            with st.spinner("🔍 Searching knowledge base..."):
                retrieved_results = enhanced_retrieval(prompt, collection_name, qdrant_client, embedding_model)
                
                if retrieved_results:
                    context = ""
                    source_info = "**📖 Retrieved Sources:**\n\n"
                    
                    for i, result in enumerate(retrieved_results):
                        context += f"**Source {i+1}**:\n"
                        context += result.payload.get('text', '') + "\n\n"
                        
                        score_color = "🟢" if result.score > 0.8 else "🟡" if result.score > 0.6 else "🟠"
                        source_info += f"{score_color} **Source {i+1}** - Relevance: `{result.score:.3f}`\n"
                    
                    response = generate_rag_response(prompt, context, groq_client, llm_model)
                    response_message = {"role": "assistant", "content": response, "sources": source_info}
                    st.session_state.messages.append(response_message)
                    
                else:
                    response = "🤔 I couldn't find relevant information in my knowledge base. Could you try rephrasing or asking about a different pandas topic?"
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show new messages
            st.rerun()

    # --- QUIZ TAB ---
    with tab2:
        st.markdown("### 🎯 Test Your Pandas Knowledge")
        
        if not quiz_questions:
            st.warning("📝 Quiz questions are not available. Please check if the quiz data was generated correctly.")
            if quiz_bank:
                st.info("💡 Quiz bank detected. Generating questions...")
                quiz_questions = flatten_quiz_bank(quiz_bank)
                if quiz_questions:
                    st.success(f"✅ Generated {len(quiz_questions)} quiz questions!")
            else:
                st.error("❌ No quiz data found. Please run the complete notebook pipeline.")
                return

        st.markdown("Challenge yourself with different types of questions based on comprehensive pandas documentation.")

        # Initialize quiz state
        if 'question_idx' not in st.session_state:
            st.session_state.question_idx = 0
            st.session_state.score = 0
            st.session_state.answer_submitted = False
            np.random.shuffle(quiz_questions)
        
        # Display score and progress
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", f"{st.session_state.score}/{len(quiz_questions)}", f"{(st.session_state.score/len(quiz_questions)*100):.1f}%")
        with col2:
            st.metric("Question", f"{st.session_state.question_idx + 1}", f"of {len(quiz_questions)}")
        with col3:
            progress = (st.session_state.question_idx + 1) / len(quiz_questions)
            st.metric("Progress", f"{progress:.1%}", "")
        
        st.progress(progress)

        # Check if quiz is over
        if st.session_state.question_idx >= len(quiz_questions):
            final_score_pct = (st.session_state.score / len(quiz_questions)) * 100
            
            if final_score_pct >= 80:
                st.balloons()
                st.success(f"🎉 **Excellent work!** You scored {st.session_state.score}/{len(quiz_questions)} ({final_score_pct:.1f}%)")
            elif final_score_pct >= 60:
                st.success(f"👏 **Good job!** You scored {st.session_state.score}/{len(quiz_questions)} ({final_score_pct:.1f}%)")
            else:
                st.info(f"📚 **Keep learning!** You scored {st.session_state.score}/{len(quiz_questions)} ({final_score_pct:.1f}%)")
            
            if st.button("🔄 Start New Quiz", type="primary", use_container_width=True):
                st.session_state.question_idx = 0
                st.session_state.score = 0
                st.session_state.answer_submitted = False
                np.random.shuffle(quiz_questions)
                st.rerun()
            st.stop()
        
        # Get current question
        question = quiz_questions[st.session_state.question_idx]
        
        # Question container
        with st.container(border=True):
            # Question header with type and difficulty
            question_type = question.get('type', 'unknown').replace('_', ' ').title()
            difficulty = question.get('difficulty', 'medium').title()
            
            difficulty_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴", "Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
            difficulty_color = difficulty_colors.get(difficulty, "⚪")
            
            st.markdown(f"### Question {st.session_state.question_idx + 1}")
            st.markdown(f"**Type:** {question_type} | **Difficulty:** {difficulty_color} {difficulty}")
            st.markdown("---")
            
            # Display question based on type
            if question.get('type') == 'multiple_choice':
                st.markdown(f"**{question['question']}**")
                user_answer = st.radio("Choose your answer:", options=question['options'], index=None, key=f"q_{st.session_state.question_idx}")
                correct_answer = question['options'][question['correct_answer']]
                
            elif question.get('type') == 'true_false':
                st.markdown(f"**{question['statement']}**")
                user_answer = st.radio("Select your answer:", options=[True, False], index=None, key=f"q_{st.session_state.question_idx}")
                correct_answer = question['correct_answer']
                
            elif question.get('type') in ['code_completion', 'fill_blank']:
                st.markdown(f"**{question['question']}**")
                if question.get('code_template'):
                    st.code(question['code_template'], language='python')
                user_answer = st.text_input("Your answer:", key=f"q_{st.session_state.question_idx}", placeholder="Enter your answer here...").strip()
                correct_answer = question['correct_answer']
                
            else:  # Scenario type
                if question.get('scenario'):
                    st.markdown(f"**Scenario:** {question['scenario']}")
                st.markdown(f"**Question:** {question.get('question', '')}")
                user_answer = st.text_area("Your answer/code:", key=f"q_{st.session_state.question_idx}", placeholder="Enter your solution here...").strip()
                correct_answer = question.get('suggested_answer', question.get('correct_answer', ''))

        # Answer buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            submit_disabled = st.session_state.answer_submitted or user_answer is None or user_answer == ""
            if st.button("✅ Submit Answer", disabled=submit_disabled, type="primary", use_container_width=True):
                st.session_state.answer_submitted = True
                
                # Check answer
                is_correct = False
                if isinstance(user_answer, str):
                    is_correct = user_answer.lower().strip() == str(correct_answer).lower().strip()
                else:
                    is_correct = user_answer == correct_answer
                
                if is_correct:
                    st.session_state.score += 1
                    st.success("🎉 **Correct!** Well done!")
                else:
                    st.error(f"❌ **Incorrect.** The correct answer was: `{correct_answer}`")
                
                # Show explanation
                if question.get('explanation'):
                    with st.expander("💡 **Explanation & Source**", expanded=True):
                        st.markdown(f"**Explanation:** {question['explanation']}")
                        if question.get('source_pages'):
                            st.info(f"📄 **Source:** {question['source_pages']}")
                
                st.rerun()

        with col2:
            next_disabled = not st.session_state.answer_submitted
            if st.button("➡️ Next Question", disabled=next_disabled, use_container_width=True):
                st.session_state.question_idx += 1
                st.session_state.answer_submitted = False
                st.rerun()

if __name__ == "__main__":
    main()