import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from dotenv import load_dotenv
from datetime import datetime
import json

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
        return json.dumps({"error": "No documents indexed"})

    results = VECTOR_STORE.similarity_search_with_score(query, k=num_results)

    documents = []
    for doc, score in results:
        documents.append(
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "score": float(score),
            }
        )

    return json.dumps({"documents": documents, "count": len(documents)})


@tool
def list_documents() -> str:
    """
    List all available documents in the database.

    Returns:
        List of document names
    """
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    docs = VECTOR_STORE.similarity_search("", k=10000)
    files = list(set(doc.metadata.get("filename", "Unknown") for doc in docs))

    return json.dumps({"documents": sorted(files), "count": len(files)})


@tool
def search_in_specific_document(query: str, filename: str) -> str:
    """
    Search within a specific document only.

    Args:
        query: The search query
        filename: The name of the document to search in

    Returns:
        Relevant chunks from the specified document
    """
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    results = VECTOR_STORE.similarity_search_with_score(query, k=20)

    filtered = [
        {"content": doc.page_content, "score": float(score)}
        for doc, score in results
        if doc.metadata.get("filename") == filename
    ][:5]

    if not filtered:
        return json.dumps({"error": f"No results found in {filename}"})

    return json.dumps({"documents": filtered, "filename": filename})


@tool
def compare_documents(topic: str, doc1: str, doc2: str) -> str:
    """
    Compare information about a topic across two documents.

    Args:
        topic: The topic to compare
        doc1: First document name
        doc2: Second document name

    Returns:
        Relevant information from both documents for comparison
    """
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    results = VECTOR_STORE.similarity_search_with_score(topic, k=20)

    doc1_results = [
        {"content": doc.page_content, "score": float(score)}
        for doc, score in results
        if doc.metadata.get("filename") == doc1
    ][:3]

    doc2_results = [
        {"content": doc.page_content, "score": float(score)}
        for doc, score in results
        if doc.metadata.get("filename") == doc2
    ][:3]

    return json.dumps({"topic": topic, doc1: doc1_results, doc2: doc2_results})


# ========================================
# STATE
# ========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ========================================
# AGENT SETUP
# ========================================

tools = [
    search_documents,
    list_documents,
    search_in_specific_document,
    compare_documents,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an intelligent document assistant with access to a document database.

Your capabilities:
1. **search_documents**: Search across all documents for information
2. **list_documents**: List all available documents
3. **search_in_specific_document**: Search within a specific document
4. **compare_documents**: Compare information across two documents

Guidelines:
- Always search documents before answering questions about their content
- Cite sources when providing information
- If information isn't found, say so clearly
- Be helpful and conversational
- Use the appropriate tool based on the user's request
- For comparisons, use the compare_documents tool

When presenting information:
- Be clear and organized
- Use bullet points for lists
- Mention the source document
- Summarize key points"""


def agent(state: AgentState):
    """Main agent node."""
    messages = state["messages"]
    system = SystemMessage(content=SYSTEM_PROMPT)
    response = llm_with_tools.invoke([system] + list(messages))
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Check if we should continue to tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========================================
# BUILD GRAPH
# ========================================


def create_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
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

    def chat(self, message: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )

        # Get last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "I couldn't generate a response."

    def new_session(self):
        self.thread_id = f"session_{datetime.now().timestamp()}"
        print("🔄 Started new session")

    def run(self):
        print("\n" + "=" * 60)
        print("🤖 AGENTIC DOCUMENT CHATBOT")
        print("=" * 60)
        print("Commands: 'quit', 'new' (new session)")
        print("=" * 60)

        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break

            if user_input.lower() == "new":
                self.new_session()
                continue

            if user_input:
                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = AgenticChatbot()
    chatbot.run()
