import streamlit as st
# from pathlib import Path
# import base64
from common.themes import load_css
from common.utils import show_static_image

# Page configuration
st.set_page_config(
    page_title="Explore AI - Solution's Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_css()

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🚀 Navigation")
    
    # Create navigation links
    pages = [
        ("🏠 Home", "app.py", False),
        ("📄 IT Troubleshoot Assist", "pages/1_IT_troubleshooting_RAG.py", False),
        ("🔬 Complex RAG", "pages/1_IT_troubleshooting_RAG.py", False),
        ("💬 AI Chatbot", "pages/1_IT_troubleshooting_RAG.py", True),
        ("📊 Analytics", "pages/1_IT_troubleshooting_RAG.py", True),
    ]
    
    for page_name, page_file, disabled in pages:
            st.page_link(page_file, label=page_name, disabled=disabled)

    

# Main content
st.markdown("""
<div class="main-header">
    <h1>🚀 Explore AI - Solution Hub</h1>
    <p>Explore our collection of advanced AI tools and implementations</p>
</div>
""", unsafe_allow_html=True)

# Define tiles data
tiles = [
    {
        "icon": "📄",
        "title": "Smart IT Troubleshoot Assist",
        "description": "Basic Retrieval-Augmented Generation system for document Q&A with straightforward implementation.",
        "page": "pages/1_IT_troubleshooting_RAG.py",
        "banner": "smart_it_troubleshoot.jpeg"
    },
    {
        "icon": "🔬",
        "title": "Complex RAG (Coming soon..)",
        "description": "Advanced RAG with hybrid search, reranking, and multi-modal capabilities for enterprise use.",
        "page": "pages/1_IT_troubleshooting_RAG.py.py",
        "disabled": True,
        "banner": "coming_soon.png"
    },
    {
        "icon": "💬",
        "title": "AI Chatbot (Coming soon..)",
        "description": "Interactive conversational AI with context awareness and personality customization.",
        "page": "pages/1_IT_troubleshooting_RAG.py.py",
        "disabled": True,
        "banner": "coming_soon.png"
    },
    {
        "icon": "📊",
        "title": "Analytics Dashboard (Coming soon..)",
        "description": "Real-time data visualization and insights powered by machine learning algorithms.",
        "page": "pages/1_IT_troubleshooting_RAG.py.py",
        "disabled": True,
        "banner": "coming_soon.png"
    },
    # {
    #     "icon": "🔍",
    #     "title": "Semantic Search",
    #     "description": "Neural search engine with semantic understanding and cross-lingual capabilities.",
    #     "page": "pages/5_Search_Engine.py"
    # },
    # {
    #     "icon": "⚙️",
    #     "title": "Settings & Config",
    #     "description": "Configure API keys, model parameters, and system preferences for all tools.",
    #     "page": "pages/6_Settings.py"
    # }
]

# Create columns for tiles (2 per row)
col1, col2 = st.columns(2)

for i, tile in enumerate(tiles):
    col = col1 if i % 2 == 0 else col2
    
    with col:
        # Create a clickable container
        show_static_image(tile["banner"])
        if st.button(
            f"{tile['icon']} **{tile['title']}**\n\n{tile['description']}",
            key=f"tile_{i}",
            use_container_width=True,
            help=f"Click to navigate to {tile['title']}",
            disabled=tile.get("disabled", False)
        ):
            st.switch_page(tile["page"])
        if i < len(tiles) - 2:
            st.markdown("<br>", unsafe_allow_html=True)

# Footer section
st.markdown("---")
st.markdown("### About")
st.info("This is an AI Solutions Hub featuring various RAG implementations, MCP-driven workflows, AI tools and multi-agent systems.")


