import os
import sys
from typing import List, TypedDict, Optional, Tuple
from pydantic import BaseModel, Field
from typing import Annotated, Sequence, List, Literal 
from langgraph.types import Command
#from langchain.schema.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from dotenv import load_dotenv

# LangChain core types
from langchain_core.documents import Document

# Loaders & RAG stack
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI

from langchain_community.agent_toolkits import SlackToolkit
from langgraph.prebuilt import create_react_agent



#import requests
DEFAULT_WEIGHTS: Tuple[float, float] = (0.7, 0.3)   # vector vs BM25

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import build_retriever_from_pdf


load_dotenv()

# def llmModel(temperature: float=1):
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
#     deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
#     subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")

#     return AzureChatOpenAI(
#         api_key=subscription_key,
#         azure_endpoint=endpoint,
#         azure_deployment=deployment,
#         api_version=api_version,
#         model=model_name,
#         temperature=temperature
#     )

class Supervisor(BaseModel):
        next: Literal["specialize", "notify"] = Field(
            description=(
                        "Determines which specialist to activate next in the workflow sequence: "
                        "'specialize' when the IT Troubleshooting Assistant needs to extract specialized knowledge from the documents in context"
                        "'notify' when the generated LLM response is ready to be communicated to the Slack channel for implementation, escalation, or user awareness"
                    )
        )
        reason: str = Field(
            description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
        )

# ----------------------------
# LangGraph State & Engine
# ----------------------------
class RAGState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str
    specialization: str  # specialization analysis markdown
    notification: bool  # whether Slack notification was sent
    notification_message: str  # message to send in Slack notification


class RAGEngine:
    """
    Encapsulates the retriever + compiled LangGraph with a specialization node.
    Use:
        engine = RAGEngine(provider="OpenAI", model="gpt-4o-mini")
        engine.index_pdf(file_bytes, filename)
        spec_md = engine.specialize()
        answer, docs = engine.ask("Your question")
    """
    def __init__(self, provider: str = "OpenAI", model: str = "gpt-4o-mini", temperature: float = 0.2, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

        self.retriever = None
        self._graph = None

        self._all_chunks: List[Document] = []
        self.specialization: Optional[str] = None

        # Create a callable LLM function
        self.llmModel = self._create_llm_object()
        self.notification = None


    def _create_llm_object(self):
        if self.provider == "OpenAI" and self.api_key:
            api_key = self.api_key
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        elif self.provider == "AzureOpenAI":
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            subscription_key = os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION")

            return AzureChatOpenAI(
                api_key=subscription_key,
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_version=api_version,
                model=model_name,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        
    # ---- Nodes ----
    
    def _supervise_node(self, state: RAGState) -> Command[Literal["specialize", "notify"]]:
        system_prompt = ('''
                    
            You are a Workflow Supervisor responsible for managing a two-step IT troubleshooting workflow. Your role is to decide the next action in the process based on the current state of the task and the user’s needs. You must provide a clear, concise rationale for every decision to ensure transparency.

                **Workflow Context**:
                - This system assists with IT troubleshooting and communication.
                - The workflow involves two possible next steps:
                    1. **specialize**: Activate the IT Troubleshooting Assistant to extract specialized knowledge from contextual documents, logs, or diagnostic data. This is chosen when additional technical details or deeper analysis are required to resolve the issue.
                    2. **notify**: Send the generated LLM response to the designated Slack channel for implementation, escalation, or user awareness. This is chosen when the solution or update is finalized and ready for communication.

                **Your Responsibilities**:
                1. Analyze the current state of the task and determine whether more technical context is needed or if the response is ready for delivery.
                2. Select the most appropriate next step:  
                - Choose **specialize** when the response lacks sufficient technical depth or requires further troubleshooting insights.  
                - Choose **notify** when the response is complete, actionable, and ready to share with stakeholders.
                3. Provide a detailed justification for your decision, explaining why this step is necessary and how it moves the workflow toward resolution.
                4. Avoid unnecessary loops or redundant steps to maintain efficiency.

                **Output Format**:
                - `next`: Either `"specialize"` or `"notify"`.
                - `reason`: A clear explanation of why this decision was made.

                Your goal is to ensure accurate, efficient troubleshooting and timely communication.
    
        ''')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.get("question", "")}
        ]

        response = self.llmModel.with_structured_output(Supervisor).invoke(messages)

        goto = response.next
        #reason = response.reason

        print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")
        
        return Command(
            goto=goto
        )
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Upload/index a PDF first.")
        docs = self.retriever.invoke(state["question"])
        return {"context": docs}

    def _generate_node(self, state: RAGState) -> RAGState:
        docs = state.get("context", []) or []
        if not docs:
            return {"answer": "I couldn't find anything relevant in the document to answer that."}

        def fmt_doc(d: Document) -> str:
            page = d.metadata.get("page")
            page_str = f"[p.{page + 1}]" if isinstance(page, int) else ""
            return f"{page_str} {d.page_content}"

        context_text = "\n\n".join(fmt_doc(d) for d in docs[:4])

        sys = (
            "You are a helpful RAG assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say you don't know. "
        )
        user = f"Question:\n{state['question']}\n\nContext:\n{context_text}"
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
        try:
            ans = self.llmModel.invoke(messages).content
        except Exception as e:
            ans = f"LLM error: {e}"
        return {"answer": ans}

    def _specialize_node(self, state: RAGState) -> RAGState:
        """
        Produce a concise specialization summary of the document in markdown.
        Cached in self.specialization to avoid recomputation.
        """
        if self.specialization:
            return {"specialization": self.specialization}

        if not self._all_chunks:
            self.specialization = "*(No document indexed yet — specialization unavailable.)*"
            return {"specialization": self.specialization}

        total = len(self._all_chunks)
        sample_idxs = sorted(set([0, total // 3, (2 * total) // 3, max(0, total - 1)]))
        samples = [self._all_chunks[i] for i in sample_idxs if 0 <= i < total]

        buf = []
        pages = []
        for d in samples:
            page = d.metadata.get("page")
            if isinstance(page, int):
                pages.append(page + 1)
            buf.append(d.page_content.strip())
        context = "\n\n---\n\n".join(buf)[:2000]

        page_min = min(pages) if pages else "?"
        page_max = max(pages) if pages else "?"

        sys = (
            "You are a document classifier and topic analyst. "
            "Given sampled passages from a single PDF, infer the document's specialization."
        )
        user = f"""
            Analyze the following excerpts (pages approx. {page_min}–{page_max}) and produce a concise markdown report.
            Your task is to analyze technical documents and identify the most specific IT specialization or domain they focus on
            **Output format (markdown, max ~120 words):**
            - **Domain**:: <specific Domain with platform, max ~5 words>
                Examples:
                - Domain: Cloud security and certificate management (AWS)
                - Domain: Infrastructure automation and CI/CD (Azure)
                - Domain: Network configuration and routing (On-Prem)
            - **Subdomains**: comma-separated
            - **Key topics**: 5–10 comma-separated keywords/phrases
            - **Summary**: 2–3 short sentences capturing what the document is about and who it's for

            Excerpts:
            {context}
        """.strip()

        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
        try:
            spec_md = self.llmModel.invoke(messages).content
        except Exception as e:
            spec_md = f"*Specialization analysis failed: {e}*"

        self.specialization = spec_md
        return {"specialization": spec_md}


    def _slack_node(self, state: RAGState):
        toolkit = SlackToolkit()
        query = state.get("question", "Send a friendly greeting to channel #l1-support-team")
        agent_executor = create_react_agent(self.llmModel, toolkit.get_tools())

        events = agent_executor.stream(
            {"messages": [("user", query)]},
            stream_mode="values",
        )

        output_messages = []
        for event in events:
            message = event["messages"][-1]
            if message.type != "tool":
                output_messages.append(message.content)

        return {
            "notification": True
        }
    
    def _route_after_specialize(self, state: RAGState) -> str:
        q = (state.get("question") or "").strip()
        return "has_question" if q else "no_question"

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("supervisor", self._supervise_node)
        g.add_node("specialize", self._specialize_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate_node)
        g.add_node("notify", self._slack_node)

        g.add_edge(START, "supervisor")
        # g.add_edge(START, "specialize")
        g.add_conditional_edges(
            "specialize",
            self._route_after_specialize,
            {"has_question": "retrieve", "no_question": END}
        )
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        g.add_edge("notify", END)
        checkpointer = MemorySaver()
        return g.compile(checkpointer=checkpointer)

    # ---- Public API ----
    def index_pdf(self, file_bytes: bytes, filename: str, k: int = 4):
        """Build a retriever from the given PDF and compile the graph."""
        hybrid_retriever, chunks = build_retriever_from_pdf(file_bytes, filename, k=k)
        self.retriever = hybrid_retriever
        self._all_chunks = chunks
        self.specialization = None  # reset specialization cache for new doc
        self._graph = self._build_graph()
        return self._graph

    def specialize(self) -> str:
        """Run specialization-only path (no question)."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Index a PDF first.")
        result: RAGState = self._graph.invoke({}, config={"configurable": {"thread_id": "1"}})
        return result.get("specialization", "*No specialization available.*")

    def notify(self, question: str) -> str:
        """Run notification path (no question)."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Index a PDF first.")
        result: RAGState = self._graph.invoke({"question": question}, config={"configurable": {"thread_id": "1"}})
        return result.get("notification", "*No notification available.*")

    def get_graph(self) -> str:
        """Run specialization-only path (no question)."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Index a PDF first.")
        result: RAGState = self._graph
        return self._graph

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Run the graph for a single QA turn."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized. Index a PDF first.")
        result: RAGState = self._graph.invoke({"question": question}, config={"configurable": {"thread_id": "1"}})
        return result.get("answer", "No answer."), result.get("context", [])

    @staticmethod
    def summarize_sources(docs: List[Document], limit: int = 4) -> List[str]:
        sources: List[str] = []
        for d in docs[:limit]:
            src = d.metadata.get("source", "uploaded.pdf")
            page = d.metadata.get("page")
            page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
            sources.append(f"{src} — {page_str}")
        return sources
