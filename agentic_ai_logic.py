import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
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
# ENHANCED STATE WITH CLARIFICATION TRACKING
# ========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: dict
    # ✅ NEW: Clarification tracking
    clarification_needed: bool
    clarification_type: str  # 'vague', 'ambiguous', 'incomplete', 'low_confidence'
    follow_up_questions: List[str]
    pending_clarification: bool  # Waiting for user clarification
    original_query: str  # Store original query for context
    clarification_attempts: int  # Track how many times we've asked
    # Existing fields
    user_intent: str
    satisfaction_asked: bool
    topic_history: List[str]
    search_confidence: float  # Confidence in search results


# ========================================
# QUERY ANALYZER CLASS
# ========================================


class QueryAnalyzer:
    """Analyzes user queries for clarity and completeness."""

    VAGUE_WORDS = [
        "it",
        "this",
        "that",
        "thing",
        "stuff",
        "something",
        "anything",
        "everything",
        "those",
        "these",
    ]

    INCOMPLETE_PATTERNS = [
        ("how", 3),  # "how" with less than 3 words
        ("what", 3),
        ("where", 3),
        ("why", 3),
        ("when", 3),
    ]

    AMBIGUOUS_TOPICS = {
        "user": ["create user", "delete user", "user permissions", "user settings"],
        "settings": ["system settings", "user settings", "admin settings"],
        "error": ["error type", "when error occurs", "error message"],
        "access": ["grant access", "remove access", "access levels"],
        "permission": ["add permission", "remove permission", "view permissions"],
    }

    @classmethod
    def analyze(cls, query: str, context: dict = None) -> dict:
        """
        Analyze query for clarity issues.

        Returns:
            dict with keys:
                - is_clear: bool
                - issues: List[str]
                - clarification_type: str
                - follow_up_questions: List[str]
                - confidence: float (0-1)
        """
        query_lower = query.lower().strip()
        words = query_lower.split()

        analysis = {
            "is_clear": True,
            "issues": [],
            "clarification_type": None,
            "follow_up_questions": [],
            "confidence": 1.0,
        }

        # Check 1: Too short
        if len(words) < 2:
            analysis["is_clear"] = False
            analysis["issues"].append("query_too_short")
            analysis["clarification_type"] = "incomplete"
            analysis["follow_up_questions"].append(
                "Could you please provide more details about what you're looking for?"
            )
            analysis["confidence"] = 0.2

        # Check 2: Vague references without context
        elif any(word in words[:3] for word in cls.VAGUE_WORDS) and len(words) < 6:
            # Check if we have context from previous conversation
            has_context = context and context.get("last_topic")
            if not has_context:
                analysis["is_clear"] = False
                analysis["issues"].append("vague_reference")
                analysis["clarification_type"] = "vague"
                analysis["follow_up_questions"].append(
                    "I'm not sure what you're referring to. Could you please be more specific?"
                )
                analysis["confidence"] = 0.3

        # Check 3: Incomplete question patterns
        for pattern, min_words in cls.INCOMPLETE_PATTERNS:
            if query_lower.startswith(pattern) and len(words) < min_words:
                analysis["is_clear"] = False
                analysis["issues"].append("incomplete_question")
                analysis["clarification_type"] = "incomplete"
                analysis["follow_up_questions"].append(
                    f"Your question seems incomplete. Could you tell me more about what you want to know?"
                )
                analysis["confidence"] = 0.4
                break

        # Check 4: Ambiguous topics
        for topic, subtopics in cls.AMBIGUOUS_TOPICS.items():
            if topic in query_lower and not any(
                sub in query_lower for sub in subtopics
            ):
                # Check if query is too generic
                if len(words) < 5:
                    analysis["is_clear"] = False
                    analysis["issues"].append("ambiguous_topic")
                    analysis["clarification_type"] = "ambiguous"
                    options = ", ".join(subtopics[:3])
                    analysis["follow_up_questions"].append(
                        f"When you mention '{topic}', could you specify which aspect? For example: {options}?"
                    )
                    analysis["confidence"] = 0.5
                    break

        # Check 5: Multiple questions in one
        question_words = ["how", "what", "where", "why", "when", "can", "does", "is"]
        question_count = sum(
            1 for word in question_words if f" {word} " in f" {query_lower} "
        )
        if question_count > 1 and " and " in query_lower:
            analysis["is_clear"] = False
            analysis["issues"].append("multiple_questions")
            analysis["clarification_type"] = "ambiguous"
            analysis["follow_up_questions"].append(
                "You've asked multiple questions. Which one would you like me to address first?"
            )
            analysis["confidence"] = 0.6

        return analysis

    @classmethod
    def analyze_search_results(cls, results: list, query: str) -> dict:
        """
        Analyze search results for confidence level.

        Args:
            results: List of (document, score) tuples from similarity search
            query: Original user query

        Returns:
            dict with confidence analysis
        """
        if not results:
            return {
                "has_results": False,
                "confidence": 0.0,
                "needs_clarification": True,
                "follow_up_questions": [
                    "I couldn't find relevant information. Could you rephrase your question or provide more context?"
                ],
            }

        # Analyze scores (lower is better for FAISS L2 distance)
        scores = [score for _, score in results]
        avg_score = sum(scores) / len(scores)
        best_score = min(scores)

        # Determine confidence based on scores
        # These thresholds may need tuning based on your data
        if best_score < 0.3:
            confidence = 0.95
        elif best_score < 0.5:
            confidence = 0.8
        elif best_score < 0.7:
            confidence = 0.6
        elif best_score < 1.0:
            confidence = 0.4
        else:
            confidence = 0.2

        needs_clarification = confidence < 0.5

        follow_up = []
        if needs_clarification:
            follow_up.append(
                "I found some information, but I'm not very confident it matches what you're looking for. "
                "Could you provide more specific details about your question?"
            )

        return {
            "has_results": True,
            "confidence": confidence,
            "best_score": best_score,
            "avg_score": avg_score,
            "needs_clarification": needs_clarification,
            "follow_up_questions": follow_up,
        }


# ========================================
# ENHANCED TOOLS
# ========================================


@tool
def search_documents_with_confidence(query: str, num_results: int = 5) -> str:
    """
    Search the document database with confidence scoring.

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        JSON with documents, confidence score, and potential clarification needs
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {"error": "No documents indexed", "needs_clarification": False}
        )

    results = VECTOR_STORE.similarity_search_with_score(query, k=num_results)

    # Analyze results for confidence
    result_analysis = QueryAnalyzer.analyze_search_results(results, query)

    documents = []
    for doc, score in results:
        documents.append(
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "score": float(score),
            }
        )

    return json.dumps(
        {
            "documents": documents,
            "count": len(documents),
            "confidence": result_analysis["confidence"],
            "needs_clarification": result_analysis["needs_clarification"],
            "clarification_reason": (
                "low_confidence" if result_analysis["needs_clarification"] else None
            ),
        }
    )


@tool
def analyze_query_clarity(query: str, conversation_context: str = "") -> str:
    """
    Analyze if the user's query is clear enough to provide a good answer.

    Args:
        query: The user's query
        conversation_context: Previous conversation context as JSON string

    Returns:
        Analysis of query clarity with follow-up questions if needed
    """
    context = {}
    if conversation_context:
        try:
            context = json.loads(conversation_context)
        except:
            pass

    analysis = QueryAnalyzer.analyze(query, context)
    return json.dumps(analysis)


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
    """List all available documents in the database."""
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    docs = VECTOR_STORE.similarity_search("", k=10000)
    files = list(set(doc.metadata.get("filename", "Unknown") for doc in docs))

    return json.dumps({"documents": sorted(files), "count": len(files)})


@tool
def search_in_specific_document(query: str, filename: str) -> str:
    """Search within a specific document only."""
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
def get_related_topics(current_topic: str) -> str:
    """Find related topics that the user might be interested in."""
    if VECTOR_STORE is None:
        return json.dumps({"error": "No documents indexed"})

    results = VECTOR_STORE.similarity_search(current_topic, k=10)

    related = set()
    for doc in results:
        content = doc.page_content.lower()
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


# ========================================
# AGENT SETUP
# ========================================

tools = [
    search_documents_with_confidence,
    analyze_query_clarity,
    search_documents,
    list_documents,
    search_in_specific_document,
    get_related_topics,
]

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

# Separate LLM for clarity analysis (faster, cheaper)
clarity_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


SYSTEM_PROMPT = """You are an intelligent, empathetic, and proactive document support assistant.

## CRITICAL: CLARIFICATION-FIRST APPROACH

Before searching for information, you MUST ensure you understand the user's question clearly.

### STEP 1: ANALYZE QUERY CLARITY
Always start by evaluating if the query is clear enough:

**Ask for clarification if:**
- The query is too vague (e.g., "How does it work?", "Tell me about that")
- The query is ambiguous (could mean multiple things)
- The query is incomplete (missing key details)
- You're not confident about what the user is asking

**Proceed with search if:**
- The query is specific and clear
- You have enough context from the conversation
- The user has already provided clarification

### STEP 2: ASK SMART FOLLOW-UP QUESTIONS
When clarification is needed, ask specific questions:
- "When you mention [X], are you referring to [option A] or [option B]?"
- "Could you tell me more about what you're trying to accomplish?"
- "Which specific aspect of [topic] would you like to know about?"

### STEP 3: SEARCH AND RESPOND
Only after the query is clear:
1. Use search_documents_with_confidence to find relevant information
2. If confidence is low, inform the user and ask if they want related information
3. Provide structured, helpful answers

### STEP 4: VERIFY SATISFACTION
After answering:
- Ask if the answer was helpful
- Offer to provide more details
- Suggest related topics

## Response Guidelines:
- Be empathetic and patient
- Never guess when unsure - ask for clarification
- Acknowledge when you're asking follow-up questions
- Keep clarification questions focused and specific
- Limit to 1-2 clarifying questions at a time

## Tools Available:
1. **analyze_query_clarity**: Check if query needs clarification
2. **search_documents_with_confidence**: Search with confidence scoring
3. **search_documents**: Standard document search
4. **list_documents**: List all documents
5. **search_in_specific_document**: Search specific document
6. **get_related_topics**: Find related topics"""


CLARIFICATION_PROMPT = """You are a query clarity analyzer. Analyze the user's query and determine if it's clear enough to answer.

User Query: {query}

Previous Topics Discussed: {topics}
Last Topic: {last_topic}

Analyze and respond with a JSON object:
{{
    "is_clear": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues if any"],
    "clarification_needed": true/false,
    "follow_up_question": "A single, specific follow-up question if needed",
    "reasoning": "Brief explanation of your analysis"
}}

Consider:
1. Is the query specific enough?
2. Are there ambiguous terms that could mean multiple things?
3. Is there missing context that's essential?
4. Can the query be answered with document search?

Respond ONLY with the JSON object, no other text."""


# ========================================
# GRAPH NODES
# ========================================


def analyze_clarity(state: AgentState) -> dict:
    """
    Node to analyze query clarity before processing.
    """
    messages = state["messages"]
    context = state.get("context", {})

    # Get the latest user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"clarification_needed": False, "pending_clarification": False}

    # Check if this is a response to a clarification request
    if state.get("pending_clarification", False):
        # User is responding to our clarification question
        # Combine with original query for better context
        original = state.get("original_query", "")
        if original:
            # Update context with the clarification
            return {
                "clarification_needed": False,
                "pending_clarification": False,
                "clarification_attempts": state.get("clarification_attempts", 0),
            }

    # Analyze query using QueryAnalyzer
    analysis = QueryAnalyzer.analyze(user_message, context)

    # Also use LLM for more nuanced analysis
    topics = state.get("topic_history", [])
    last_topic = context.get("last_topic", "None")

    try:
        llm_prompt = CLARIFICATION_PROMPT.format(
            query=user_message,
            topics=", ".join(topics[-5:]) if topics else "None",
            last_topic=last_topic,
        )

        llm_response = clarity_llm.invoke([HumanMessage(content=llm_prompt)])
        llm_analysis = json.loads(llm_response.content)

        # Combine both analyses
        needs_clarification = (
            not analysis["is_clear"]
            or not llm_analysis.get("is_clear", True)
            or llm_analysis.get("confidence", 1.0) < 0.5
        )

        follow_up_questions = analysis["follow_up_questions"]
        if llm_analysis.get("follow_up_question"):
            follow_up_questions.insert(0, llm_analysis["follow_up_question"])

        return {
            "clarification_needed": needs_clarification,
            "clarification_type": (
                analysis["clarification_type"]
                or llm_analysis.get("issues", ["unclear"])[0]
                if needs_clarification
                else None
            ),
            "follow_up_questions": follow_up_questions[:2],  # Limit to 2 questions
            "pending_clarification": needs_clarification,
            "original_query": (
                user_message if needs_clarification else state.get("original_query", "")
            ),
            "search_confidence": llm_analysis.get("confidence", analysis["confidence"]),
        }

    except Exception as e:
        print(f"⚠️ Clarity analysis error: {e}")
        # Fall back to basic analysis
        return {
            "clarification_needed": not analysis["is_clear"],
            "clarification_type": analysis["clarification_type"],
            "follow_up_questions": analysis["follow_up_questions"],
            "pending_clarification": not analysis["is_clear"],
            "original_query": user_message if not analysis["is_clear"] else "",
            "search_confidence": analysis["confidence"],
        }


def ask_clarification(state: AgentState) -> dict:
    """
    Node to ask clarifying questions.
    """
    follow_up_questions = state.get("follow_up_questions", [])
    clarification_type = state.get("clarification_type", "unclear")
    attempts = state.get("clarification_attempts", 0)

    # Build clarification message
    if attempts >= 2:
        # After 2 attempts, try to answer anyway
        message = (
            "I've asked a couple of clarifying questions already. "
            "Let me try to help with what I understand so far. "
            "I'll search for relevant information and you can let me know if I'm on the right track."
        )
        return {
            "messages": [AIMessage(content=message)],
            "clarification_needed": False,
            "pending_clarification": False,
            "clarification_attempts": attempts + 1,
        }

    # Prepare empathetic clarification message
    prefixes = {
        "vague": "I want to make sure I understand your question correctly.",
        "ambiguous": "Your question could relate to a few different things.",
        "incomplete": "I'd like to get a bit more information to help you better.",
        "low_confidence": "I want to ensure I find the most relevant information for you.",
        "unclear": "I'd like to clarify something before I search.",
    }

    prefix = prefixes.get(clarification_type, prefixes["unclear"])

    if follow_up_questions:
        question = follow_up_questions[0]
        message = f"{prefix}\n\n🤔 {question}"
    else:
        message = f"{prefix}\n\n🤔 Could you please provide more details about what you're looking for?"

    return {
        "messages": [AIMessage(content=message)],
        "pending_clarification": True,
        "clarification_attempts": attempts + 1,
    }


def agent(state: AgentState) -> dict:
    """Main agent node with enhanced context awareness."""
    messages = state["messages"]
    context = state.get("context", {})
    topic_history = state.get("topic_history", [])

    # Build enhanced system message with context
    context_info = ""
    if topic_history:
        context_info = f"\n\nPrevious topics discussed: {', '.join(topic_history[-5:])}"

    if context.get("last_topic"):
        context_info += f"\nLast topic: {context['last_topic']}"

    # Add clarification context if returning from clarification
    if state.get("original_query") and not state.get("pending_clarification"):
        context_info += f"\n\nOriginal question was: {state['original_query']}"
        context_info += "\nThe user has now provided clarification. Use both the original question and the clarification to search."

    # Add search confidence guidance
    search_confidence = state.get("search_confidence", 1.0)
    if search_confidence < 0.7:
        context_info += f"\n\nNote: Initial confidence in understanding the query is {search_confidence:.0%}. Consider verifying with the user if the answer seems off."

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response]}


def check_search_results(state: AgentState) -> dict:
    """
    Post-search node to check result confidence and potentially ask for clarification.
    """
    messages = state["messages"]

    # Look for the last tool response
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                if "confidence" in msg.content and "documents" in msg.content:
                    result = json.loads(msg.content)

                    if (
                        result.get("needs_clarification")
                        and result.get("confidence", 1.0) < 0.4
                    ):
                        # Very low confidence - ask for clarification
                        return {
                            "clarification_needed": True,
                            "clarification_type": "low_confidence",
                            "follow_up_questions": [
                                "The search results don't seem to match your question well. "
                                "Could you rephrase or provide more specific details about what you're looking for?"
                            ],
                        }
            except (json.JSONDecodeError, TypeError):
                continue

    return {"clarification_needed": False}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if we should continue to tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def route_after_clarity(state: AgentState) -> Literal["ask_clarification", "agent"]:
    """Route based on clarity analysis results."""
    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 2:  # Limit clarification attempts
            return "ask_clarification"
    return "agent"


def route_after_tools(state: AgentState) -> Literal["check_results", "agent"]:
    """Route after tools to optionally check results."""
    # Check if the last tool was a search
    messages = state["messages"]
    for msg in reversed(messages):
        if hasattr(msg, "content"):
            try:
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if "confidence" in content:
                    return "check_results"
            except:
                pass
        break
    return "agent"


def route_after_result_check(
    state: AgentState,
) -> Literal["ask_clarification", "agent"]:
    """Route after checking search results."""
    if state.get("clarification_needed", False):
        return "ask_clarification"
    return "agent"


# ========================================
# BUILD ENHANCED GRAPH
# ========================================


def create_agent():
    """Create the enhanced agent workflow with clarification handling."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_clarity", analyze_clarity)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("check_results", check_search_results)

    # Define edges
    workflow.add_edge(START, "analyze_clarity")

    # After clarity analysis, either ask for clarification or proceed to agent
    workflow.add_conditional_edges(
        "analyze_clarity",
        route_after_clarity,
        {"ask_clarification": "ask_clarification", "agent": "agent"},
    )

    # After asking clarification, end (wait for user response)
    workflow.add_edge("ask_clarification", END)

    # Agent decides whether to use tools or end
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    # After tools, check results or go back to agent
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
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0

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

        if any(ind in message_lower for ind in frustrated_indicators):
            return "frustrated"
        elif any(ind in message_lower for ind in positive_indicators):
            return "positive"
        return "neutral"

    def _extract_topic(self, message: str) -> Optional[str]:
        """Extract the main topic from a message."""
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
        return "Good evening"

    def chat(self, message: str) -> str:
        """Send a message and get a response with clarification handling."""
        config = {"configurable": {"thread_id": self.thread_id}}

        # Update context
        self.context["questions_asked"] += 1
        self.context["user_sentiment"] = self._analyze_sentiment(message)

        topic = self._extract_topic(message)
        if topic:
            self.context["last_topic"] = topic
            self.context["topics_discussed"].append(topic)
            self.topic_history.append(topic)

        # Prepare state with clarification tracking
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
        }

        try:
            result = self.agent.invoke(initial_state, config=config)

            # Update clarification state
            self.pending_clarification = result.get("pending_clarification", False)
            if self.pending_clarification:
                self.original_query = result.get("original_query", message)
                self.clarification_attempts = result.get("clarification_attempts", 0)
                self.context["clarifications_made"] += 1
            else:
                # Reset clarification state after successful response
                self.original_query = None
                self.clarification_attempts = 0

            # Get last AI message
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content

        except Exception as e:
            print(f"Error in chat: {e}")
            return "I encountered an error processing your request. Could you please try rephrasing your question?"

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
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0
        print("🔄 Started new session. How can I help you today?")

    def show_help(self):
        """Display help information."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    📚 CHATBOT HELP                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  COMMANDS:                                                   ║
║  • quit/exit  - Exit the chatbot                            ║
║  • new        - Start a new conversation                    ║
║  • help       - Show this help message                      ║
║  • docs       - List available documents                    ║
║  • history    - Show topics discussed                       ║
║  • status     - Show session information                    ║
║                                                              ║
║  FEATURES:                                                   ║
║  ✨ Smart clarification - I'll ask follow-up questions      ║
║     if your query needs more details                        ║
║  📊 Confidence scoring - I'll let you know if I'm           ║
║     unsure about the answer                                 ║
║  💡 Related topics - I'll suggest related information       ║
║                                                              ║
║  TIPS FOR BETTER ANSWERS:                                   ║
║  • Be specific in your questions                            ║
║  • Mention the document name if you know it                 ║
║  • Answer my clarifying questions for better results        ║
║                                                              ║
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
║  Session Started: {self.context['session_start'][:19]:<25}   ║
║  Questions Asked: {self.context['questions_asked']:<26}      ║
║  Clarifications: {self.context['clarifications_made']:<27}   ║
║  Topics Discussed: {len(self.context['topics_discussed']):<25}║
║  Pending Clarification: {str(self.pending_clarification):<20}║
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
        print("║" + " " * 10 + "🤖 INTELLIGENT DOCUMENT ASSISTANT" + " " * 13 + "║")
        print("═" * 60)
        print(
            f"""
{greeting}! I'm your document support assistant.

I can help you:
  📚 Search through documents for information
  🔍 Answer questions about your documents
  📊 Compare information across documents
  💡 Suggest related topics

✨ NEW: I'll ask clarifying questions if I need more details
   to give you the best possible answer!

Type 'help' for more commands or just ask me anything!
"""
        )
        print("═" * 60)

        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
            print(f"\n📁 I have access to {len(files)} document(s).")
        else:
            print("\n⚠️ No documents are currently indexed.")

        while True:
            try:
                # Indicate if waiting for clarification
                if self.pending_clarification:
                    prompt = "\n👤 Your clarification: "
                else:
                    prompt = "\n👤 You: "

                user_input = input(prompt).strip()

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
                if self.pending_clarification:
                    print("\n💭 Thanks for clarifying! Let me search for that...")
                else:
                    print("\n🤔 Let me analyze your question...")

                sentiment = self._analyze_sentiment(user_input)
                if sentiment == "frustrated":
                    print("💭 I understand this might be frustrating. Let me help you.")

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
        clarifications = self.context["clarifications_made"]

        print("\n" + "═" * 60)
        print(
            f"""
👋 Thank you for using the Document Assistant!

📊 Session Summary:
   • Questions answered: {questions}
   • Topics covered: {topics}
   • Clarifications made: {clarifications}

We hope you found the information helpful!
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
