import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.rag_engine import RAGEngine
from common.utils import show_langgraph_visualization, slack_dialog


st.markdown("""
    <style>
    /* Hide sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stSidebarUserContent"] {
        padding-bottom: 0;
    }
    [data-testid="stSidebarHeader"] {
        height: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stSpinner {
        background-color: rgba(255, 255, 255, 1);  /* lighter overlay */
    }
    </style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 350px; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Custom CSS
st.markdown("""
<style>
.page-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    margin-top: -4rem;
    color: white;
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
</style>

""", unsafe_allow_html=True)

st.set_page_config(page_title="Chat to PDF (LangGraph + Streamlit)", page_icon="📄", layout="wide")

@st.cache_resource(show_spinner=False)
def get_engine(provider, model, temperature, api_key):
    return RAGEngine(provider=provider, model=model, temperature=temperature, api_key=api_key)


# ----------------------------
# Session init
# ----------------------------
def init_session():
    for key, default in {
        "OPENAI_API_KEY": "",
        "engine": None,
        "messages": [],
        "indexed": False,
        "specialization": None,
        "uploaded_id": None,
        "graph": None,
        "last_provider": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session()

# Header
st.markdown("""
<div class="page-header">
    <h1>📄 Smart IT Troubleshoot Assist</h1>
    <p>Upload documents and ask questions to get AI-powered answers</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    if st.button("← Back to Home"):
        st.switch_page("app.py")
    st.header("⚙️ Settings")

    provider = st.selectbox("LLM Provider", ["AzureOpenAI", "OpenAI"], index=0,key="provider")
    model = st.text_input("Model",
                          value="gpt-4o-mini" if provider == "OpenAI" else "gpt-4.1",
                           disabled=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, step=0.05)

    api_key = ""
    if provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state["OPENAI_API_KEY"] or "",
            type="password",
            help="For OpenAI, put the OpenAI API KEY here.",
        )
        if api_key != st.session_state["OPENAI_API_KEY"]:
            st.session_state["OPENAI_API_KEY"] = api_key


    should_initialize = (
        st.session_state.engine is None or st.session_state.last_provider != provider
    )
    if should_initialize:
        if provider == "OpenAI" and not st.session_state["OPENAI_API_KEY"]:
            st.warning("Please enter a valid OpenAI API key to proceed.")
        else:
            st.session_state.engine = get_engine(
                provider=provider,
                model=model,
                temperature=temperature,
                api_key=st.session_state["OPENAI_API_KEY"] if provider == "OpenAI" else None
            )
            st.session_state.last_provider = provider

    # Apply settings
    st.session_state.engine.provider = provider
    st.session_state.engine.model = model
    st.session_state.engine.temperature = temperature

    st.markdown("---")
    uploaded = st.file_uploader("Upload a Troubleshooting Guidelines (in PDF)", type=["pdf"])
    if uploaded is not None:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.uploaded_id != file_id:
            try:
                file_bytes = uploaded.read()
                st.session_state.messages = []
                
                with st.spinner("Indexing PDF..."):
                    graph = st.session_state.engine.index_pdf(file_bytes, uploaded.name)
                    st.session_state.graph = graph
            
                with st.spinner("Analyzing document specialization..."):
                    st.session_state.specialization = st.session_state.engine.specialize()

                st.session_state.indexed = True
                st.session_state.uploaded_id = file_id
                st.toast(f"✅ Uploaded: {uploaded.name}")
            except Exception as e:
                st.session_state.indexed = False
                st.session_state.specialization = None
                st.error(f"Failed to index PDF: {e}")

    # Generate LangGraph visualization
    with st.sidebar:
        show_langgraph_visualization()

# ----------------------------
# Main
# ----------------------------

# Specialization panel (visible after indexing)
if st.session_state.indexed and st.session_state.specialization:
    st.markdown("### 🔎 Domain Specialization")
    st.markdown(st.session_state.specialization)
    st.markdown("---")

if not st.session_state.indexed:
    st.info("➡️ Upload a PDF from the sidebar to start troubleshooting.")
else:
    st.success("✅ Your troubleshooting guide is indexed! Ask me anything about it." )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Chat input
prompt = st.chat_input("Ask something about the uploaded PDF…", disabled=not st.session_state.indexed, key="chat_prompt")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        print("messages list", st.session_state.messages)
    with st.spinner("Thinking..."):
        try:
            answer, context_docs = st.session_state.engine.ask(prompt)
            st.session_state["last_ai_text"] = answer
        except Exception as e:
            answer, context_docs = f"Error: {e}", []

    sources = st.session_state.engine.summarize_sources(context_docs)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

last_ai = st.session_state.get("last_ai_text")
if not last_ai:
    last = next((m for m in reversed(st.session_state.messages) if m.get("role") == "assistant"), None)
    last_ai = last.get("content") if last else None

if last_ai:
    if st.button("Send Response to Slack", type="primary", key="btn_send_slack"):
        slack_dialog(last_ai)


