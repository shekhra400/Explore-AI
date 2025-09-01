from langchain_openai import AzureChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List, Tuple
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import os, sys
from PIL import Image
import io
import time
import tempfile

# Loaders & RAG stack
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS



load_dotenv()   
# Ensure the environment variables are loaded
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")



def embeddingModel():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
    embeddings_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    return AzureOpenAIEmbeddings(
        api_key=subscription_key,
        azure_endpoint=endpoint,
        model=model_name,  # Specify the model if needed
        openai_api_version=embeddings_api_version,
        azure_deployment=deployment  # Specify the deployment if needed
    )


def llmObj(temperature=1):
    return AzureChatOpenAI(
    api_key=subscription_key,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_version=api_version,
    model=model_name,
    temperature=temperature
)

def build_and_save_vectorstore(chunks: List[Document], save_path="faiss_index"):
    if not chunks:
        raise ValueError("The 'chunks' list is empty. Cannot build vector store with no documents.")
    embeddings = embeddingModel()
    vector_store = FAISS.from_documents(embedding=embeddings, documents=chunks)
    vector_store.save_local(save_path)


def load_vectorstore(save_path="faiss_index"):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Vector store path '{save_path}' does not exist.")
    if not os.path.exists(os.path.join(save_path, "index.faiss")):
        raise FileNotFoundError("Missing index.faiss file.")
    if not os.path.exists(os.path.join(save_path, "index.pkl")):
        raise FileNotFoundError("Missing index.pkl file.")

    embeddings = embeddingModel()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS (defaults)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_K: int = 6
DEFAULT_WEIGHTS: Tuple[float, float] = (0.7, 0.3)   # vector vs BM25
DEFAULT_MMR: bool = True
DEFAULT_LAMBDA: float = 0.5


def get_hybrid_retriever(
    vectorstore,
    corpus_docs: List[Document],  # Same docs used to build the vector store
    k: int = DEFAULT_K,
    weights: Tuple[float, float] = DEFAULT_WEIGHTS,
    mmr: bool = DEFAULT_MMR,
    lambda_mult: float = DEFAULT_LAMBDA
) -> BaseRetriever:
    """
    Creates a hybrid retriever that combines:
    - Vector-based retrieval (semantic similarity, optionally with MMR)
    - BM25 keyword-based retrieval
    Weighted combination using EnsembleRetriever.

    Use this when you need both semantic and keyword relevance.
    """

    # 1. Create vector retriever (MMR or similarity)
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr" if mmr else "similarity",
        search_kwargs={"k": k, "lambda_mult": lambda_mult} if mmr else {"k": k}
    )

    # 2. Create BM25 retriever (keyword-based)
    # NOTE: Build once and reuse for performance in production
    bm25_retriever = BM25Retriever.from_documents(corpus_docs)
    bm25_retriever.k = k

    # 3. Combine both retrievers into a hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights)  # e.g., [0.7, 0.3]
    )

    return hybrid_retriever



class HybridSearchInput(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(DEFAULT_K, description="Number of results to return")


def make_hybrid_tool(vectorstore, corpus_docs):
    retriever = get_hybrid_retriever(vectorstore, corpus_docs)

    def run(query: str, k: int = DEFAULT_K):
        docs = retriever.invoke(query)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    return StructuredTool.from_function(
        name="hybrid_search",
        description="Retrieve documents using hybrid search (vector + BM25).",
        func=run,
        args_schema=HybridSearchInput
    )


def save_langgraph_workflow_image(graph):
    image_bytes = graph.get_graph(xray=1).draw_mermaid_png()

    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Define the path to save the image in the codebase
    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    os.makedirs(resources_dir, exist_ok=True)
    image_path = os.path.join(resources_dir, "langgraph_workflow.png")

    # Save the image
    image.save(image_path)
    return image_path

def show_langgraph_visualization(image = "langgraph_workflow.png"):

    st.markdown('---')
    st.html(
        "<h3 style='font-family: Poppins;text-align: center'>Agentic Flow Visualization</h3>"
    )
    image_path = os.path.join("resources", "static", image)
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.html(
            """
            <style>
                [data-testid="stImage"] {
                    border-radius: 10px;
                    overflow: hidden;
                }
            </style>
            """
        )
        st.image(image, caption="LangGraph Workflow", use_container_width=True)
    else:
        st.warning("Image not found at the specified path.")


def show_static_image(image):
    image_path = os.path.join("resources", "static", image)
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.html(
            """
            <style>
                [data-testid="stImage"] {
                    border-radius: 10px;
                    overflow: hidden;
                }
            </style>
            """
        )
        st.image(image, use_container_width=True)
        


# ----------------------------
# Slack Dialog, Send Message
# ----------------------------

@st.dialog("Send to L1 Support Slack Channels", width="large", dismissible=True)
def slack_dialog(default_text:str):
    st.caption("Edit the message before sending to Slack.")
    if "slack_draft" not in st.session_state:
        st.session_state.slack_draft = default_text

    default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "l1-support-team")
    channel = st.text_input("Channel (ID or name)", value=st.session_state.get("slack_channel", default_channel), key="slack_channel", disabled=True)
    st.text_area("Message", key="slack_draft", height=250)
    col1, _, _, _, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        if st.button("Send", type="primary"):
            if channel.strip() and st.session_state.slack_draft.strip():
                with st.spinner("Posting to Slack..."):
                    refined_message = st.session_state.slack_draft.strip()
                    query = f"Send the following Markdown-formatted message to the Slack {channel} #l1-support-team:\n\n{refined_message}"
                    ok = st.session_state.engine.notify(query)
                    st.session_state["notification"] = ok
                    if ok:
                        st.toast("✅ Message successfully posted to Slack")
                        time.sleep(2)
                        # Close the dialog after success
                        st.rerun()
            else:
                st.warning("Please provide both channel and message.")
    with col5:
        if st.button("Cancel"):
            st.rerun()



# ----------------------------
# PDF → Chunks → Retriever (PyPDFLoader)
# ----------------------------
def load_pdf_with_pypdfloader(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Use PyPDFLoader to load the PDF into page-level Documents.
    We temporarily write bytes to disk because PyPDFLoader expects a file path.
    """
    # Create a temp file and write the bytes
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()  # returns List[Document], one per page

        # Normalize metadata: use the original uploaded filename as "source"
        for d in docs:
            d.metadata["source"] = filename
            # PyPDFLoader already provides 'page' (usually zero-based); keep it for [p.X] display
        return docs
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                # On some systems, a quick retry or ignore is fine
                pass




def build_retriever_from_pdf(file_bytes: bytes, filename: str, k: int = 4, weights: Tuple[float, float] = DEFAULT_WEIGHTS):
    """Build an in-memory FAISS retriever from a PDF file using PyPDFLoader."""
    base_docs = load_pdf_with_pypdfloader(file_bytes, filename)
    if not base_docs:
        raise RuntimeError("No extractable text found in the PDF (PyPDFLoader).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(base_docs)

    embeddings = embeddingModel()
    vs = FAISS.from_documents(chunks, embeddings)

    # 1. Create vector retriever (similarity)
    vector_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # 2. Create BM25 retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # 3. Combine both retrievers into a hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights)  # e.g., [0.7, 0.3]
    )
    return hybrid_retriever, chunks