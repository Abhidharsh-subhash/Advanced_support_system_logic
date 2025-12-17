import os
import re
import json
import operator
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Optional, Literal

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ========================================
# VECTOR STORE
# ========================================


def load_vector_store(index_path: str = "faiss_index"):
    if not os.path.exists(index_path):
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )


VECTOR_STORE = load_vector_store()


# ========================================
# STATE DEFINITION
# ========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict
    clarification_needed: bool
    clarification_type: str
    follow_up_questions: List[str]
    pending_clarification: bool
    original_query: str
    clarification_attempts: int
    user_intent: str
    satisfaction_asked: bool
    topic_history: List[str]
    search_confidence: float
    has_searched: bool
    search_results: str


# ========================================
# QUERY ANALYZER
# ========================================


class QueryAnalyzer:
    """Analyzes user queries for clarity."""

    TRULY_VAGUE_PATTERNS = [
        r"^(it|this|that|these|those)[\?\.]?$",
        r"^(what|how|why|where|when)[\?\.]?$",
        r"^(help|more|info|details)[\?\.]?$",
        r"^tell me[\?\.]?$",
        r"^explain[\?\.]?$",
        r"^show me[\?\.]?$",
    ]

    @classmethod
    def analyze(cls, query: str, context: dict = None) -> dict:
        query_lower = query.lower().strip()

        analysis = {
            "is_clear": True,
            "issues": [],
            "clarification_type": None,
            "follow_up_questions": [],
            "confidence": 0.8,
        }

        if len(query_lower) < 2:
            analysis["is_clear"] = False
            analysis["issues"].append("empty_query")
            analysis["clarification_type"] = "incomplete"
            analysis["follow_up_questions"].append(
                "I didn't catch that. What would you like to know?"
            )
            analysis["confidence"] = 0.0
            return analysis

        for pattern in cls.TRULY_VAGUE_PATTERNS:
            if re.match(pattern, query_lower):
                has_context = context and context.get("last_topic")
                if not has_context:
                    analysis["is_clear"] = False
                    analysis["issues"].append("vague_reference")
                    analysis["clarification_type"] = "vague"
                    analysis["follow_up_questions"].append(
                        "Could you please be more specific about what you'd like to know?"
                    )
                    analysis["confidence"] = 0.2
                    return analysis

        return analysis

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "greetings",
        ]
        return query.lower().strip() in greetings

    @classmethod
    def analyze_search_results(cls, results: list, query: str) -> dict:
        if not results:
            return {
                "has_results": False,
                "confidence": 0.0,
                "best_score": 0.0,
                "needs_clarification": True,
            }

        # ✅ Convert numpy float32 to Python float
        scores = [float(score) for _, score in results]
        best_score = min(scores)

        if best_score < 0.5:
            confidence = 0.9
        elif best_score < 0.8:
            confidence = 0.7
        elif best_score < 1.2:
            confidence = 0.5
        else:
            confidence = 0.3

        return {
            "has_results": True,
            "confidence": float(confidence),
            "best_score": float(best_score),
            "needs_clarification": confidence < 0.3,
        }


# ========================================
# TOOLS
# ========================================


@tool
def search_documents(query: str, num_results: int = 5) -> str:
    """
    Search the document database for relevant information.

    Args:
        query: The search query
        num_results: Number of results to return (default: 5)

    Returns:
        Relevant document chunks with source information
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {"error": "No documents indexed", "found_answer": False, "documents": []}
        )

    try:
        results = VECTOR_STORE.similarity_search_with_score(query, k=num_results)
    except Exception as e:
        return json.dumps({"error": str(e), "found_answer": False, "documents": []})

    if not results:
        return json.dumps({"found_answer": False, "documents": []})

    analysis = QueryAnalyzer.analyze_search_results(results, query)

    documents = []
    for doc, score in results:
        documents.append(
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "score": float(score),  # ✅ Convert to Python float
            }
        )

    return json.dumps(
        {
            "found_answer": bool(analysis["confidence"] > 0.3),
            "confidence": float(analysis["confidence"]),
            "documents": documents,
            "count": int(len(documents)),
            "best_match_score": float(analysis.get("best_score", 0)),
        }
    )


@tool
def list_available_documents() -> str:
    """List all available documents in the knowledge base."""
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed", "documents": []})

    try:
        docs = VECTOR_STORE.similarity_search("", k=10000)
        files = list(set(doc.metadata.get("filename", "Unknown") for doc in docs))
        return json.dumps({"documents": sorted(files), "count": int(len(files))})
    except Exception as e:
        return json.dumps({"error": str(e), "documents": []})


@tool
def search_specific_document(query: str, filename: str) -> str:
    """Search within a specific document only."""
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    try:
        results = VECTOR_STORE.similarity_search_with_score(query, k=20)

        filtered = []
        for doc, score in results:
            if doc.metadata.get("filename") == filename:
                filtered.append(
                    {
                        "content": doc.page_content,
                        "score": float(score),
                        "filename": filename,
                    }
                )

        filtered = filtered[:5]

        if not filtered:
            return json.dumps(
                {
                    "found_answer": False,
                    "message": f"No relevant information found in {filename}",
                    "documents": [],
                }
            )

        return json.dumps(
            {"found_answer": True, "documents": filtered, "filename": filename}
        )
    except Exception as e:
        return json.dumps({"error": str(e), "found_answer": False})


# ========================================
# SETUP
# ========================================

tools = [
    search_documents,
    list_available_documents,
    search_specific_document,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """You are a document-based support assistant. You MUST follow these rules:

## CRITICAL RULES:

### RULE 1: ALWAYS SEARCH FIRST
- Call `search_documents` for EVERY user question
- NEVER answer from your general knowledge
- NEVER make up information

### RULE 2: USE ONLY DOCUMENT CONTENT
- Answers MUST come from search results only
- Quote or paraphrase the document content
- **NEVER mention source filenames, document names, or file references in your response**
- **NEVER say things like "according to [filename]" or "from [document]"**

### RULE 3: WHEN NOTHING FOUND
If search returns no relevant results, say:
"I couldn't find information about that topic in the available documentation."

### RULE 4: BE HONEST
- Don't guess or use external knowledge
- If documents don't have the answer, say so

### RULE 5: RESPONSE FORMAT
- Provide clear, direct answers
- Do NOT reference where the information came from
- Do NOT mention file names, document names, or sources
- Present information as if you inherently know it from the documentation

## TOOLS:
- `search_documents`: Search all documents (USE THIS FOR EVERY QUESTION)
- `list_available_documents`: Show available documents
- `search_specific_document`: Search in a specific document"""

# ========================================
# GRAPH NODES
# ========================================


def should_clarify(state: AgentState) -> dict:
    messages = state["messages"]
    context = state.get("context", {})

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"clarification_needed": False}

    if state.get("pending_clarification", False):
        return {
            "clarification_needed": False,
            "pending_clarification": False,
        }

    if QueryAnalyzer.is_greeting(user_message):
        return {"clarification_needed": False}

    analysis = QueryAnalyzer.analyze(user_message, context)
    needs_clarification = not analysis["is_clear"] and analysis["confidence"] < 0.3

    return {
        "clarification_needed": needs_clarification,
        "clarification_type": (
            analysis["clarification_type"] if needs_clarification else None
        ),
        "follow_up_questions": (
            analysis["follow_up_questions"] if needs_clarification else []
        ),
        "pending_clarification": needs_clarification,
        "original_query": user_message if needs_clarification else "",
        "search_confidence": analysis["confidence"],
    }


def ask_clarification(state: AgentState) -> dict:
    follow_up_questions = state.get("follow_up_questions", [])
    attempts = state.get("clarification_attempts", 0)

    if attempts >= 1:
        return {
            "messages": [AIMessage(content="Let me search for what I can find...")],
            "clarification_needed": False,
            "pending_clarification": False,
        }

    message = (
        follow_up_questions[0]
        if follow_up_questions
        else "Could you provide more details?"
    )

    return {
        "messages": [AIMessage(content=message)],
        "pending_clarification": True,
        "clarification_attempts": attempts + 1,
    }


def agent(state: AgentState) -> dict:
    messages = state["messages"]
    context = state.get("context", {})
    topic_history = state.get("topic_history", [])

    context_info = ""
    if topic_history:
        context_info = f"\n\nRecent topics: {', '.join(topic_history[-3:])}"

    if state.get("original_query"):
        context_info += f"\nOriginal question: {state['original_query']}"

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def route_after_clarity(state: AgentState) -> Literal["ask_clarification", "agent"]:
    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 1:
            return "ask_clarification"
    return "agent"


# ========================================
# BUILD GRAPH
# ========================================


def create_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("check_clarity", should_clarify)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "check_clarity")

    workflow.add_conditional_edges(
        "check_clarity",
        route_after_clarity,
        {"ask_clarification": "ask_clarification", "agent": "agent"},
    )

    workflow.add_edge("ask_clarification", END)

    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========================================
# CHATBOT CLASS
# ========================================


class AgenticChatbot:
    def __init__(self):
        self.agent = create_agent()
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "session_start": datetime.now().isoformat(),
        }
        self.topic_history = []
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0

    def _extract_topic(self, message: str) -> Optional[str]:
        keywords = [
            "user",
            "password",
            "login",
            "admin",
            "settings",
            "permission",
            "create",
            "delete",
            "update",
            "error",
            "configure",
            "setup",
            "install",
            "access",
            "account",
            "profile",
            "report",
            "data",
            "tmc",
            "visa",
            "booking",
            "invoice",
            "staff",
            "portal",
        ]
        message_lower = message.lower()
        for keyword in keywords:
            if keyword in message_lower:
                return keyword
        return None

    def chat(self, message: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}

        self.context["questions_asked"] += 1
        topic = self._extract_topic(message)
        if topic:
            self.context["last_topic"] = topic
            self.topic_history.append(topic)

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_type": "",
            "follow_up_questions": [],
            "pending_clarification": self.pending_clarification,
            "original_query": self.original_query or "",
            "clarification_attempts": self.clarification_attempts,
            "user_intent": "",
            "satisfaction_asked": False,
            "topic_history": self.topic_history,
            "search_confidence": 1.0,
            "has_searched": False,
            "search_results": "",
        }

        try:
            result = self.agent.invoke(initial_state, config=config)

            self.pending_clarification = result.get("pending_clarification", False)
            if self.pending_clarification:
                self.original_query = result.get("original_query", message)
                self.clarification_attempts = result.get("clarification_attempts", 0)
            else:
                self.original_query = None
                self.clarification_attempts = 0

            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content

        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an error. Could you please rephrase your question?"

        return "I couldn't generate a response. Please try again."

    def run(self):
        print("\n" + "=" * 60)
        print("  📚 DOCUMENT ASSISTANT")
        print("=" * 60)
        print("\nI answer questions based on your indexed documents.")
        print("Type 'quit' to exit, 'docs' to list documents.\n")

        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
            print(f"📁 {len(files)} document(s) available.\n")
        else:
            print("⚠️ No documents indexed.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! 👋")
                    break

                if user_input.lower() == "docs":
                    if VECTOR_STORE:
                        docs = VECTOR_STORE.similarity_search("", k=10000)
                        files = set(
                            doc.metadata.get("filename", "Unknown") for doc in docs
                        )
                        print("\nAvailable documents:")
                        for f in sorted(files):
                            print(f"  • {f}")
                        print()
                    continue

                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = AgenticChatbot()
    chatbot.run()
