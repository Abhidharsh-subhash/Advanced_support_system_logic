import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, List, Optional
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
# ENHANCED STATE WITH CONTEXT
# ========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict  # Store conversation context
    clarification_needed: bool
    follow_up_questions: List[str]
    user_intent: str
    satisfaction_asked: bool
    topic_history: List[str]


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


@tool
def get_related_topics(current_topic: str) -> str:
    """
    Find related topics that the user might be interested in.

    Args:
        current_topic: The current topic being discussed

    Returns:
        List of related topics found in the documents
    """
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    results = VECTOR_STORE.similarity_search(current_topic, k=10)

    # Extract unique topics/sections from results
    related = set()
    for doc in results:
        content = doc.page_content.lower()
        # Extract potential topic keywords
        if "how to" in content:
            related.add("How-to guides")
        if "troubleshoot" in content or "error" in content:
            related.add("Troubleshooting")
        if "setting" in content or "config" in content:
            related.add("Configuration settings")
        if "permission" in content or "access" in content:
            related.add("Permissions and access")
        if "user" in content:
            related.add("User management")

    return json.dumps(
        {"related_topics": list(related)[:5], "current_topic": current_topic}
    )


@tool
def check_query_clarity(query: str) -> str:
    """
    Analyze if the user's query is clear enough to provide a good answer.

    Args:
        query: The user's query

    Returns:
        Analysis of query clarity and suggested clarifying questions
    """
    analysis = {"is_clear": True, "missing_info": [], "clarifying_questions": []}

    query_lower = query.lower()

    # Check for vague queries
    vague_indicators = ["it", "this", "that", "thing", "stuff", "something"]
    if (
        any(word in query_lower.split() for word in vague_indicators)
        and len(query.split()) < 5
    ):
        analysis["is_clear"] = False
        analysis["missing_info"].append("specific subject")
        analysis["clarifying_questions"].append(
            "Could you please specify what you're referring to?"
        )

    # Check for action without context
    if (
        any(word in query_lower for word in ["how", "what", "where"])
        and len(query.split()) < 4
    ):
        analysis["is_clear"] = False
        analysis["missing_info"].append("more context")
        analysis["clarifying_questions"].append(
            "Could you provide more details about what you're trying to accomplish?"
        )

    # Check for multiple topics
    if " and " in query_lower and len(query.split()) > 10:
        analysis["is_clear"] = False
        analysis["missing_info"].append("focus on single topic")
        analysis["clarifying_questions"].append(
            "You mentioned multiple things. Which one would you like me to address first?"
        )

    return json.dumps(analysis)


# ========================================
# AGENT SETUP
# ========================================

tools = [
    search_documents,
    list_documents,
    search_in_specific_document,
    compare_documents,
    get_related_topics,
    check_query_clarity,
]

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.3
)  # Slight temperature for more natural responses
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an intelligent, empathetic, and proactive document support assistant. Your goal is to provide exceptional support by being helpful, thorough, and engaging.

## Your Capabilities:
1. **search_documents**: Search across all documents for information
2. **list_documents**: List all available documents
3. **search_in_specific_document**: Search within a specific document
4. **compare_documents**: Compare information across two documents
5. **get_related_topics**: Find related topics the user might find helpful
6. **check_query_clarity**: Analyze if you need more information from the user

## Interactive Support Guidelines:

### 1. CLARIFICATION & FOLLOW-UP
- If the user's question is vague or ambiguous, ask clarifying questions BEFORE searching
- Use the check_query_clarity tool when needed
- Examples of clarifying questions:
  - "Just to make sure I understand correctly, are you asking about X or Y?"
  - "Could you tell me more about what you're trying to accomplish?"
  - "Which specific feature/area are you referring to?"

### 2. PROACTIVE ASSISTANCE
- After answering, suggest related topics they might find helpful
- Offer to explain concepts in more detail if the topic is complex
- Ask if they need help with anything else related to the topic
- Example: "I've answered your question about user creation. Would you also like to know about setting permissions for new users?"

### 3. STRUCTURED RESPONSES
- Use clear formatting with headers, bullet points, and numbered steps
- For complex answers, break them into digestible sections
- Provide step-by-step instructions when applicable

### 4. EMPATHY & ENGAGEMENT
- Acknowledge the user's situation
- Use friendly, conversational language
- If the user seems frustrated, acknowledge it: "I understand this can be frustrating. Let me help you..."

### 5. FEEDBACK LOOP
- After providing detailed answers, ask: "Did this answer your question?" or "Is there anything else you'd like me to clarify?"
- If the answer wasn't helpful, offer alternatives: "Let me try a different approach..."

### 6. SMART SUGGESTIONS
- If you notice patterns in questions, proactively offer guidance
- Suggest related documentation they might find useful
- Offer to compare different approaches if applicable

### 7. HANDLING EDGE CASES
- If information isn't found: "I couldn't find specific information about X in the documents. However, I found related information about Y that might help. Would you like me to share that?"
- If multiple interpretations exist: "Your question could relate to A or B. Which would you like me to focus on?"

### Response Format:
1. **Acknowledge** the question/concern
2. **Clarify** if needed (ask follow-up questions)
3. **Answer** thoroughly with sources
4. **Suggest** related topics or next steps
5. **Check** if the user needs more help

Remember: You're not just answering questions - you're providing a complete support experience!"""


def agent(state: AgentState):
    """Main agent node with enhanced context awareness."""
    messages = state["messages"]
    context = state.get("context", {})
    topic_history = state.get("topic_history", [])

    # Build enhanced system message with context
    context_info = ""
    if topic_history:
        context_info = f"\n\nPrevious topics discussed in this session: {', '.join(topic_history[-5:])}"

    if context.get("last_topic"):
        context_info += f"\nLast topic discussed: {context['last_topic']}"

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
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
# ENHANCED CHATBOT CLASS
# ========================================


class AgenticChatbot:
    def __init__(self):
        self.agent = create_agent()
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "clarifications_made": 0,
            "session_start": datetime.now().isoformat(),
            "user_sentiment": "neutral",
            "pending_followup": None,
        }
        self.topic_history = []
        self.feedback_pending = False

    def _analyze_sentiment(self, message: str) -> str:
        """Simple sentiment analysis for user messages."""
        message_lower = message.lower()

        frustrated_indicators = [
            "not working",
            "doesn't work",
            "frustrated",
            "annoying",
            "confused",
            "don't understand",
            "help me",
            "stuck",
            "error",
            "problem",
            "issue",
            "wrong",
            "broken",
        ]

        positive_indicators = [
            "thanks",
            "thank you",
            "great",
            "perfect",
            "helpful",
            "awesome",
            "excellent",
            "good",
            "works",
            "solved",
        ]

        if any(indicator in message_lower for indicator in frustrated_indicators):
            return "frustrated"
        elif any(indicator in message_lower for indicator in positive_indicators):
            return "positive"
        return "neutral"

    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract the main topic from a message."""
        # Simple keyword extraction
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
        ]

        message_lower = message.lower()
        for keyword in keywords:
            if keyword in message_lower:
                return keyword
        return None

    def _get_greeting(self) -> str:
        """Get appropriate greeting based on time of day."""
        hour = datetime.now().hour
        if hour < 12:
            return "Good morning"
        elif hour < 17:
            return "Good afternoon"
        else:
            return "Good evening"

    def chat(self, message: str) -> str:
        """Send a message and get a response with enhanced context."""
        config = {"configurable": {"thread_id": self.thread_id}}

        # Update context
        self.context["questions_asked"] += 1
        self.context["user_sentiment"] = self._analyze_sentiment(message)

        topic = self._extract_topic(message)
        if topic:
            self.context["last_topic"] = topic
            self.context["topics_discussed"].append(topic)
            self.topic_history.append(topic)

        # Prepare state with context
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "follow_up_questions": [],
            "user_intent": "",
            "satisfaction_asked": False,
            "topic_history": self.topic_history,
        }

        result = self.agent.invoke(initial_state, config=config)

        # Get last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return (
            "I couldn't generate a response. Could you please rephrase your question?"
        )

    def new_session(self):
        """Start a new conversation session."""
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {
            "last_topic": None,
            "topics_discussed": [],
            "questions_asked": 0,
            "clarifications_made": 0,
            "session_start": datetime.now().isoformat(),
            "user_sentiment": "neutral",
            "pending_followup": None,
        }
        self.topic_history = []
        print("🔄 Started new session. How can I help you today?")

    def show_help(self):
        """Display help information."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    📚 CHATBOT HELP                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  COMMANDS:                                                    ║
║  ─────────                                                    ║
║  • quit/exit  - Exit the chatbot                             ║
║  • new        - Start a new conversation                     ║
║  • help       - Show this help message                       ║
║  • docs       - List available documents                     ║
║  • history    - Show topics discussed in this session        ║
║  • status     - Show session information                     ║
║                                                               ║
║  TIPS FOR BETTER ANSWERS:                                    ║
║  ────────────────────────                                    ║
║  • Be specific in your questions                             ║
║  • Mention the document name if you know it                  ║
║  • Ask follow-up questions for more details                  ║
║  • Use "compare" to compare information across documents     ║
║                                                               ║
║  EXAMPLE QUESTIONS:                                          ║
║  ──────────────────                                          ║
║  • "How do I create a new user?"                             ║
║  • "What are the admin permissions?"                         ║
║  • "Search for password reset in AdminUserManual.docx"       ║
║  • "Compare user roles between doc1 and doc2"                ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)

    def show_history(self):
        """Show topics discussed in this session."""
        if not self.topic_history:
            print("\n📝 No topics discussed yet in this session.")
        else:
            print("\n📝 Topics discussed in this session:")
            for i, topic in enumerate(self.topic_history, 1):
                print(f"   {i}. {topic}")

    def show_status(self):
        """Show current session status."""
        status = f"""
╔══════════════════════════════════════════════════════════════╗
║                    📊 SESSION STATUS                         ║
╠══════════════════════════════════════════════════════════════╣
║  Session Started: {self.context['session_start'][:19]}       ║
║  Questions Asked: {self.context['questions_asked']}          ║
║  Topics Discussed: {len(self.context['topics_discussed'])}   ║
║  Last Topic: {self.context['last_topic'] or 'None':<20}      ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(status)

    def list_docs(self):
        """List available documents."""
        if VECTOR_STORE is None:
            print("\n❌ No documents indexed. Please add documents first.")
            return

        docs = VECTOR_STORE.similarity_search("", k=10000)
        files = set(doc.metadata.get("filename", "Unknown") for doc in docs)

        print("\n📂 Available Documents:")
        print("─" * 40)
        for f in sorted(files):
            print(f"   • {f}")
        print("─" * 40)
        print(f"   Total: {len(files)} document(s)")

    def run(self):
        """Run the interactive chatbot."""
        greeting = self._get_greeting()

        print("\n" + "═" * 60)
        print("║" + " " * 15 + "🤖 INTELLIGENT DOCUMENT ASSISTANT" + " " * 8 + "║")
        print("═" * 60)
        print(
            f"""
{greeting}! I'm your document support assistant. 

I can help you:
  📚 Search through documents for information
  🔍 Answer questions about your documents  
  📊 Compare information across documents
  💡 Suggest related topics you might find helpful

Type 'help' for more commands or just ask me anything!
"""
        )
        print("═" * 60)

        # Welcome interaction
        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
            print(
                f"\n📁 I have access to {len(files)} document(s). What would you like to know?"
            )
        else:
            print("\n⚠️ No documents are currently indexed.")
            print("Please add documents using the document storage script first.")

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    self._farewell()
                    break

                if user_input.lower() == "new":
                    self.new_session()
                    continue

                if user_input.lower() == "help":
                    self.show_help()
                    continue

                if user_input.lower() == "docs":
                    self.list_docs()
                    continue

                if user_input.lower() == "history":
                    self.show_history()
                    continue

                if user_input.lower() == "status":
                    self.show_status()
                    continue

                # Process the message
                print("\n🤔 Let me look into that for you...")

                # Check sentiment and add empathetic response if needed
                sentiment = self._analyze_sentiment(user_input)
                if sentiment == "frustrated":
                    print(
                        "💭 I understand this might be frustrating. Let me help you resolve this."
                    )

                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n")
                self._farewell()
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Let me try again. Could you rephrase your question?")

    def _farewell(self):
        """Display farewell message."""
        questions = self.context["questions_asked"]
        topics = len(set(self.context["topics_discussed"]))

        print("\n" + "═" * 60)
        print(
            f"""
👋 Thank you for using the Document Assistant!

📊 Session Summary:
   • Questions answered: {questions}
   • Topics covered: {topics}
   
We hope you found the information helpful!
Feel free to come back anytime you have questions.

Goodbye! 👋
"""
        )
        print("═" * 60)


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = AgenticChatbot()
    chatbot.run()
