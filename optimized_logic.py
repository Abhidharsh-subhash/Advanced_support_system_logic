import os
import json
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ========================================
# VECTOR STORE
# ========================================

FAISS_INDEX_PATH = "/home/abhidharsh-fgil/FGIL Projects/Convergent/vector_store/faiss"


def load_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return None


VECTOR_STORE = load_vector_store()


# ========================================
# TOOLS
# ========================================


@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for relevant information about the user's question."""
    if VECTOR_STORE is None:
        return "Knowledge base not available."

    try:
        results = VECTOR_STORE.similarity_search_with_score(query, k=3)
        if not results:
            return "No relevant information found."

        # Filter and format results
        relevant = []
        for doc, score in results:
            if score < 1.0:
                relevant.append(f"[Score: {score:.2f}]\n{doc.page_content[:1000]}")

        if not relevant:
            return "No sufficiently relevant information found."

        return "\n\n---\n\n".join(relevant)

    except Exception as e:
        return f"Search error: {str(e)}"


# ========================================
# PREBUILT REACT AGENT (MINIMAL CODE!)
# ========================================

SYSTEM_PROMPT = """You are a helpful support assistant.

For greetings, respond warmly. For farewells, say goodbye politely.
For questions, use the search_docs tool to find information, then answer based on results.
If no information is found, say so politely and offer to help with something else.
Be concise and helpful. Never make up information."""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
tools = [search_docs]
memory = MemorySaver()

# This creates a complete ReAct agent in ONE LINE!
agent = create_react_agent(
    llm, tools, checkpointer=memory, state_modifier=SYSTEM_PROMPT  # System prompt
)


# ========================================
# CHATBOT
# ========================================


class MinimalChatbot:
    def __init__(self):
        self.thread_id = f"session_{datetime.now().timestamp()}"

    def chat(self, message: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}

        result = agent.invoke(
            {"messages": [HumanMessage(content=message)]}, config=config
        )

        # Get last AI message
        for msg in reversed(result["messages"]):
            if (
                hasattr(msg, "content")
                and msg.content
                and not hasattr(msg, "tool_calls")
            ):
                return msg.content

        return "I couldn't process that."

    def run(self):
        print("\n🚀 Minimal ReAct Agent")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            if user_input:
                response = self.chat(user_input)
                print(f"Bot: {response}\n")


if __name__ == "__main__":
    chatbot = MinimalChatbot()
    chatbot.run()
