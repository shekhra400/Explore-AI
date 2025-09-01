
import streamlit as st

# Custom CSS for beautiful theme
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --text-color: #1a202c;
        --bg-color: #f7fafc;
        --card-bg: #ffffff;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        text-align: center;
        font-size: 2.5rem !important;
        margin: 0 !important;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        text-align: center;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* Tile container */
    .tile-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
        padding: 1rem;
    }
    
    @media (max-width: 768px) {
        .tile-container {
            grid-template-columns: 1fr;
        }
    }
    
    /* Individual tile styling */
    .tile {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        cursor: pointer;
        text-decoration: none;
        display: block;
        height: 100%;
    }
    
    .tile:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102,126,234,0.2);
        border-color: #667eea;
    }
    
    .tile-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .tile-title {
        color: #2d3748 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .tile-description {
        color: #718096 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Navigation link styling */
    .nav-link {
        display: block;
        padding: 0.75rem 1rem;
        color: white !important;
        text-decoration: none;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.1);
    }
    
    .nav-link:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }    
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    </style>
    """, unsafe_allow_html=True)

