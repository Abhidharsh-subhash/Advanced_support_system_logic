import os
import re
import json
import operator
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Literal
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ========================================
# CONFIGURATION
# ========================================


class Config:
    """Configuration constants."""

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4
    LOW_CONFIDENCE_THRESHOLD = 0.2

    # Score thresholds (lower is better for FAISS L2 distance)
    EXCELLENT_SCORE = 0.5
    GOOD_SCORE = 0.7
    ACCEPTABLE_SCORE = 1.0

    # If best score is above this, consider it "not found"
    NOT_FOUND_SCORE_THRESHOLD = 1.2

    # Minimum number of relevant results needed
    MIN_RELEVANT_RESULTS = 1


# ========================================
# ENUMS
# ========================================


class SearchQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    LOW = "low"
    NOT_FOUND = "not_found"


class InteractionMode(str, Enum):
    GREETING = "greeting"
    QUERY = "query"
    CLARIFICATION = "clarification"
    NOT_FOUND = "not_found"
    CLOSING = "closing"


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

    # Clarification handling
    clarification_needed: bool
    clarification_reason: str
    follow_up_questions: List[str]
    pending_clarification: bool
    original_query: str
    clarification_attempts: int

    # Intent & Understanding
    user_intent: str
    detected_topics: List[str]
    sentiment: str

    # Interaction tracking
    interaction_mode: str
    conversation_history: List[dict]
    topic_history: List[str]

    # Search state - CRITICAL FOR VALIDATION
    search_confidence: float
    search_quality: str
    has_searched: bool
    search_results: str
    found_relevant_info: bool
    best_match_score: float

    # Response control
    should_respond_not_found: bool
    not_found_message: str


# ========================================
# SEARCH RESULT ANALYZER
# ========================================


class SearchResultAnalyzer:
    """Analyzes search results to determine if we have valid information."""

    @classmethod
    def analyze(cls, results: list, query: str) -> dict:
        """
        Analyze search results and determine quality.

        Returns dict with:
        - found_relevant_info: bool
        - confidence: float (0-1)
        - quality: SearchQuality
        - best_score: float
        - should_respond: bool
        - reason: str
        """
        if not results:
            return {
                "found_relevant_info": False,
                "confidence": 0.0,
                "quality": SearchQuality.NOT_FOUND.value,
                "best_score": float("inf"),
                "should_respond": False,
                "reason": "No search results returned",
                "relevant_count": 0,
            }

        # Extract scores (lower is better for FAISS L2)
        scores = [float(score) for _, score in results]
        best_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Count relevant results (score below acceptable threshold)
        relevant_results = [s for s in scores if s < Config.ACCEPTABLE_SCORE]
        relevant_count = len(relevant_results)

        # Determine quality and confidence
        if best_score < Config.EXCELLENT_SCORE:
            quality = SearchQuality.EXCELLENT.value
            confidence = 0.95
        elif best_score < Config.GOOD_SCORE:
            quality = SearchQuality.GOOD.value
            confidence = 0.8
        elif best_score < Config.ACCEPTABLE_SCORE:
            quality = SearchQuality.MODERATE.value
            confidence = 0.5
        elif best_score < Config.NOT_FOUND_SCORE_THRESHOLD:
            quality = SearchQuality.LOW.value
            confidence = 0.25
        else:
            quality = SearchQuality.NOT_FOUND.value
            confidence = 0.0

        # Check keyword overlap for additional validation
        query_keywords = set(query.lower().split())
        keyword_matches = 0

        for doc, _ in results:
            content_lower = doc.page_content.lower()
            matches = sum(
                1 for kw in query_keywords if kw in content_lower and len(kw) > 3
            )
            keyword_matches = max(keyword_matches, matches)

        keyword_relevance = keyword_matches / max(len(query_keywords), 1)

        # Final determination: should we respond with this info?
        should_respond = (
            quality != SearchQuality.NOT_FOUND.value
            and relevant_count >= Config.MIN_RELEVANT_RESULTS
            and (
                confidence > Config.LOW_CONFIDENCE_THRESHOLD or keyword_relevance > 0.3
            )
        )

        # Reason for the decision
        if not should_respond:
            if quality == SearchQuality.NOT_FOUND.value:
                reason = "No relevant information found in knowledge base"
            elif relevant_count < Config.MIN_RELEVANT_RESULTS:
                reason = "Insufficient relevant results"
            else:
                reason = "Low confidence in search results"
        else:
            reason = "Relevant information found"

        return {
            "found_relevant_info": should_respond,
            "confidence": confidence,
            "quality": quality,
            "best_score": best_score,
            "avg_score": avg_score,
            "should_respond": should_respond,
            "reason": reason,
            "relevant_count": relevant_count,
            "keyword_relevance": keyword_relevance,
        }

    @classmethod
    def validate_response_content(cls, response: str, search_results: list) -> bool:
        """
        Validate that the response is actually based on search results.
        Returns True if response seems to be from documents.
        """
        if not search_results:
            return False

        # Extract key phrases from search results
        result_content = " ".join(doc.page_content.lower() for doc, _ in search_results)
        response_lower = response.lower()

        # Check if response contains key terms from results
        result_words = set(result_content.split())
        response_words = set(response_lower.split())

        # Filter to meaningful words (length > 4)
        meaningful_result_words = {w for w in result_words if len(w) > 4}
        meaningful_response_words = {w for w in response_words if len(w) > 4}

        if not meaningful_result_words:
            return False

        overlap = meaningful_response_words.intersection(meaningful_result_words)
        overlap_ratio = (
            len(overlap) / len(meaningful_response_words)
            if meaningful_response_words
            else 0
        )

        # At least 20% of response words should be from documents
        return overlap_ratio > 0.2


# ========================================
# NOT FOUND RESPONSE GENERATOR
# ========================================


class NotFoundResponseGenerator:
    """Generates appropriate responses when information is not found."""

    RESPONSES = {
        "general": [
            "I don't have information about that in my knowledge base. Could you try asking about a different topic, or rephrase your question?",
            "I couldn't find any relevant information about this topic. Is there something else I can help you with?",
            "That topic doesn't appear to be covered in the documentation I have access to. Would you like to ask about something else?",
        ],
        "partial": [
            "I found some related information, but nothing that directly answers your question. Would you like me to share what I found, or would you prefer to ask about something else?",
            "I have some information that might be tangentially related, but I'm not confident it answers your question. Shall I share it, or would you like to try a different question?",
        ],
        "suggest_rephrase": [
            "I couldn't find a match for your query. Could you try rephrasing it or being more specific about what you're looking for?",
            "No results found for that query. Perhaps try using different keywords or asking in a different way?",
        ],
    }

    @classmethod
    def generate(
        cls, query: str, search_analysis: dict, available_topics: List[str] = None
    ) -> str:
        """Generate an appropriate not-found response."""
        import random

        quality = search_analysis.get("quality", SearchQuality.NOT_FOUND.value)
        confidence = search_analysis.get("confidence", 0)

        # Choose response type based on situation
        if quality == SearchQuality.LOW.value and confidence > 0.1:
            response = random.choice(cls.RESPONSES["partial"])
        elif confidence == 0:
            response = random.choice(cls.RESPONSES["general"])
        else:
            response = random.choice(cls.RESPONSES["suggest_rephrase"])

        # Add topic suggestions if available
        if available_topics and len(available_topics) > 0:
            topic_list = ", ".join(available_topics[:5])
            response += f"\n\nI can help you with topics like: {topic_list}."

        return response


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
                "I'm here to help! What would you like to know about?"
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
            "hi there",
            "hello there",
        ]
        return query.lower().strip().rstrip("!.,") in greetings

    @classmethod
    def is_closing(cls, query: str) -> bool:
        closings = [
            "bye",
            "goodbye",
            "see you",
            "thanks",
            "thank you",
            "that's all",
            "done",
            "exit",
            "quit",
            "thx",
        ]
        query_clean = query.lower().strip().rstrip("!.,")
        return any(c in query_clean for c in closings)


# ========================================
# TOOLS WITH STRICT VALIDATION
# ========================================


@tool
def search_documents(query: str, num_results: int = 5) -> str:
    """
    Search the document database for relevant information.
    Returns search results with quality analysis.
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No knowledge base available",
                "should_respond": False,
            }
        )

    try:
        results = VECTOR_STORE.similarity_search_with_score(query, k=num_results)
    except Exception as e:
        return json.dumps(
            {
                "found_answer": False,
                "error": str(e),
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "should_respond": False,
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No results found for this query",
                "should_respond": False,
            }
        )

    # Analyze results quality
    analysis = SearchResultAnalyzer.analyze(results, query)

    # Only include documents if we should respond
    documents = []
    if analysis["should_respond"]:
        for doc, score in results:
            if float(score) < Config.ACCEPTABLE_SCORE:
                documents.append(
                    {
                        "content": doc.page_content,
                        "relevance": (
                            "high" if float(score) < Config.GOOD_SCORE else "medium"
                        ),
                    }
                )

    return json.dumps(
        {
            "found_answer": analysis["found_relevant_info"],
            "should_respond": analysis["should_respond"],
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "best_score": float(analysis["best_score"]),
            "documents": documents,
            "count": len(documents),
            "reason": analysis["reason"],
        }
    )


@tool
def get_available_topics() -> str:
    """Get list of topics available in the knowledge base."""
    if VECTOR_STORE is None:
        return json.dumps({"topics": [], "message": "No knowledge base available"})

    try:
        # Sample documents to extract topics
        docs = VECTOR_STORE.similarity_search("", k=100)

        # Extract potential topics from content
        all_text = " ".join(doc.page_content.lower() for doc in docs)

        topic_keywords = {
            "User Management": ["user", "account", "profile", "permission", "role"],
            "Authentication": ["login", "password", "sso", "authentication", "sign in"],
            "Settings & Configuration": [
                "settings",
                "configure",
                "setup",
                "preferences",
            ],
            "Billing & Payments": ["invoice", "payment", "billing", "subscription"],
            "Bookings & Reservations": ["booking", "reservation", "travel", "trip"],
            "Reports & Analytics": ["report", "analytics", "dashboard", "export"],
            "Integration": ["api", "integration", "sync", "webhook"],
            "Troubleshooting": ["error", "issue", "problem", "fix", "troubleshoot"],
        }

        available_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text for kw in keywords):
                available_topics.append(topic)

        return json.dumps({"topics": available_topics, "count": len(available_topics)})
    except Exception as e:
        return json.dumps({"topics": [], "error": str(e)})


# ========================================
# SETUP
# ========================================

tools = [search_documents, get_available_topics]

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


# STRICT SYSTEM PROMPT - Emphasizes document-only responses
SYSTEM_PROMPT = """You are a document-based support assistant. You can ONLY provide information that exists in the search results.

## ABSOLUTE RULES - YOU MUST FOLLOW THESE:

### RULE 1: SEARCH FIRST, ALWAYS
- Call `search_documents` for EVERY user question
- Wait for search results before responding

### RULE 2: ONLY USE SEARCH RESULTS
- You can ONLY use information from the `documents` array in search results
- NEVER use your general knowledge
- NEVER make up information
- NEVER infer or guess beyond what's in the documents

### RULE 3: CHECK `should_respond` FLAG
- If search returns `"should_respond": false` → DO NOT answer the question
- Instead, say you don't have information about this topic
- Suggest the user ask about available topics

### RULE 4: WHEN `should_respond` IS FALSE, SAY:
"I don't have information about [topic] in my knowledge base. I can only answer questions about topics covered in the documentation. Would you like to ask about something else?"

### RULE 5: NEVER DO THESE:
- Never say "Based on my knowledge..."
- Never say "Generally speaking..."
- Never say "In most cases..."
- Never provide information that's not in the search results
- Never mention file names or sources

### RULE 6: RESPONSE FORMAT
When you DO have information:
- Provide clear, direct answers
- Use only content from the documents
- Be helpful and conversational

When you DON'T have information:
- Clearly state you don't have this information
- Suggest the user try a different question
- Offer to show available topics

## TOOLS:
- `search_documents`: Search for information (USE THIS FIRST)
- `get_available_topics`: Show what topics are available"""


# ========================================
# GRAPH NODES
# ========================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input for intent and clarity."""
    messages = state["messages"]
    context = state.get("context", {})

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {
            "clarification_needed": False,
            "interaction_mode": InteractionMode.QUERY.value,
        }

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {
            "interaction_mode": InteractionMode.GREETING.value,
            "clarification_needed": False,
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "clarification_needed": False,
        }

    # Handle pending clarification
    if state.get("pending_clarification", False):
        return {
            "clarification_needed": False,
            "pending_clarification": False,
            "interaction_mode": InteractionMode.QUERY.value,
        }

    # Analyze query clarity
    analysis = QueryAnalyzer.analyze(user_message, context)
    needs_clarification = not analysis["is_clear"] and analysis["confidence"] < 0.3

    return {
        "clarification_needed": needs_clarification,
        "clarification_reason": analysis.get("clarification_type", ""),
        "follow_up_questions": analysis.get("follow_up_questions", []),
        "pending_clarification": needs_clarification,
        "original_query": user_message if needs_clarification else "",
        "search_confidence": analysis["confidence"],
        "interaction_mode": (
            InteractionMode.CLARIFICATION.value
            if needs_clarification
            else InteractionMode.QUERY.value
        ),
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    import random

    greetings = [
        "Hello! I'm your support assistant. I can answer questions based on the available documentation. How can I help you today?",
        "Hi there! I'm here to help you find information from our knowledge base. What would you like to know?",
        "Hey! I can help you with questions about topics in our documentation. What are you looking for?",
    ]
    return {
        "messages": [AIMessage(content=random.choice(greetings))],
    }


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    import random

    closings = [
        "You're welcome! Feel free to come back if you have more questions. Have a great day! 👋",
        "Happy to help! Don't hesitate to ask if anything else comes up. Take care!",
        "Glad I could assist! I'm here whenever you need help. Goodbye!",
    ]
    return {
        "messages": [AIMessage(content=random.choice(closings))],
    }


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification when needed."""
    follow_up_questions = state.get("follow_up_questions", [])
    attempts = state.get("clarification_attempts", 0)

    if attempts >= 2:
        return {
            "messages": [AIMessage(content="Let me search with what I have...")],
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
    """Main agent node - processes queries with strict document-only responses."""
    messages = state["messages"]
    topic_history = state.get("topic_history", [])

    context_info = ""
    if topic_history:
        context_info = f"\n\nRecent topics: {', '.join(topic_history[-3:])}"

    if state.get("original_query"):
        context_info += f"\nOriginal question: {state['original_query']}"

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_search_results(state: AgentState) -> dict:
    """
    CRITICAL NODE: Validate search results and determine if we should respond.
    This prevents the LLM from using general knowledge.
    """
    messages = state["messages"]

    # Find the last tool message (search results)
    last_tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                last_tool_result = json.loads(msg.content)
                break
            except (json.JSONDecodeError, AttributeError):
                continue

    if last_tool_result is None:
        return {"should_respond_not_found": False}

    # Check if search found relevant information
    should_respond = last_tool_result.get("should_respond", False)
    quality = last_tool_result.get("quality", SearchQuality.NOT_FOUND.value)
    confidence = last_tool_result.get("confidence", 0)
    found_answer = last_tool_result.get("found_answer", False)

    if not should_respond or quality == SearchQuality.NOT_FOUND.value:
        # Generate not-found response
        not_found_msg = NotFoundResponseGenerator.generate(
            query=state.get("original_query", ""),
            search_analysis=last_tool_result,
            available_topics=None,  # Could fetch from get_available_topics
        )

        return {
            "should_respond_not_found": True,
            "not_found_message": not_found_msg,
            "search_quality": quality,
            "search_confidence": confidence,
            "found_relevant_info": False,
        }

    return {
        "should_respond_not_found": False,
        "search_quality": quality,
        "search_confidence": confidence,
        "found_relevant_info": True,
    }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    not_found_message = state.get("not_found_message")

    # If no custom message, use a safe default
    if not not_found_message:
        not_found_message = (
            "I don't have information about that topic in my knowledge base. "
            "I can only answer questions about topics covered in the documentation. "
            "Would you like to ask about something else?"
        )

    # NOT_FOUND messages should NOT be sanitized - they're pre-written
    return {
        "messages": [AIMessage(content=not_found_message)],
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def route_after_analysis(
    state: AgentState,
) -> Literal["handle_greeting", "handle_closing", "ask_clarification", "agent"]:
    """Route based on analysis results."""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"

    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"

    if state.get("clarification_needed", False):
        attempts = state.get("clarification_attempts", 0)
        if attempts < 2:
            return "ask_clarification"

    return "agent"


def route_after_validation(state: AgentState) -> Literal["handle_not_found", "agent"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"
    return "agent"


# ========================================
# RESPONSE SANITIZER
# ========================================


class ResponseSanitizer:
    """Sanitize responses to remove file references without breaking normal text."""

    # Patterns that indicate general knowledge usage
    GENERAL_KNOWLEDGE_PATTERNS = [
        r"(?i)\bbased on my (general )?knowledge\b",
        r"(?i)\bgenerally speaking\b",
        r"(?i)\bin most cases\b",
        r"(?i)\bfrom what I know\b",
        r"(?i)\bI believe that\b",
        r"(?i)\bit'?s? commonly known\b",
    ]

    # File reference patterns - MORE SPECIFIC to avoid false positives
    FILE_PATTERNS = [
        # Match actual file names with extensions
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml|html?|md)\b",
        # Match "from/in [filename.ext]" - MUST have file extension
        r'(?:from|in|according to)\s+["\']?[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?)["\']?',
        # Match explicit source references
        r"\(source:\s*[^)]+\)",
        r"\[source:\s*[^\]]+\]",
        r"source:\s*[\w\-\.]+\.(pdf|docx?|txt)",
        r"file:\s*[\w\-\.]+\.(pdf|docx?|txt)",
        # Match "According to [Document Name]:" pattern
        r"(?i)according to the [\w\s]+ document[,:]?\s*",
        # Match "Based on [filename]" with extension
        r"(?i)based on [\w\-]+\.(pdf|docx?|txt)[,:]?\s*",
        # Match "From the [X] file:" pattern
        r"(?i)from the [\w\s]+ file[,:]?\s*",
    ]

    @classmethod
    def contains_general_knowledge(cls, response: str) -> bool:
        """Check if response seems to use general knowledge."""
        for pattern in cls.GENERAL_KNOWLEDGE_PATTERNS:
            if re.search(pattern, response):
                return True
        return False

    @classmethod
    def sanitize(cls, response: str) -> str:
        """Remove file references without breaking normal text."""
        if not response:
            return response

        sanitized = response

        # Apply each pattern carefully
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Clean up any double spaces or leading/trailing spaces
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
        sanitized = sanitized.strip()

        return sanitized


# ========================================
# BUILD GRAPH
# ========================================


def create_agent():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_closing", handle_closing)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("validate_search", validate_search_results)
    workflow.add_node("handle_not_found", handle_not_found)

    # Entry point
    workflow.add_edge(START, "analyze_input")

    # Route after input analysis
    workflow.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        {
            "handle_greeting": "handle_greeting",
            "handle_closing": "handle_closing",
            "ask_clarification": "ask_clarification",
            "agent": "agent",
        },
    )

    # Terminal nodes
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("handle_not_found", END)

    # Agent -> Tools or End
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    # Tools -> Validate Search Results
    workflow.add_edge("tools", "validate_search")

    # Validate -> Handle Not Found or Continue to Agent
    workflow.add_conditional_edges(
        "validate_search",
        route_after_validation,
        {"handle_not_found": "handle_not_found", "agent": "agent"},
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========================================
# CHATBOT CLASS
# ========================================


class StrictDocumentChatbot:
    """Chatbot that ONLY responds from document content."""

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
        self.conversation_history = []
        self.pending_clarification = False
        self.original_query = None
        self.clarification_attempts = 0

    def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message."""
        topic_keywords = {
            "authentication": ["login", "password", "auth", "sign in"],
            "user_management": ["user", "account", "profile", "permission"],
            "configuration": ["setting", "config", "setup"],
            "billing": ["invoice", "payment", "billing"],
            "booking": ["booking", "reservation", "travel"],
        }

        message_lower = message.lower()
        detected = []

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(topic)

        return detected

    def chat(self, message: str) -> str:
        """Process a chat message."""
        config = {"configurable": {"thread_id": self.thread_id}}

        self.context["questions_asked"] += 1
        topics = self._extract_topics(message)
        if topics:
            self.context["last_topic"] = topics[0]
            self.topic_history.extend(topics)

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": self.pending_clarification,
            "original_query": self.original_query or message,
            "clarification_attempts": self.clarification_attempts,
            "user_intent": "",
            "detected_topics": topics,
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
            "conversation_history": self.conversation_history,
            "topic_history": self.topic_history,
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": "",
            "found_relevant_info": False,
            "best_match_score": float("inf"),
            "should_respond_not_found": False,
            "not_found_message": "",
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

            # Get response
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    response = msg.content

                    # Only sanitize if it's NOT a pre-written not-found response
                    is_not_found_response = result.get(
                        "should_respond_not_found", False
                    )

                    if not is_not_found_response:
                        # Check for general knowledge usage
                        if ResponseSanitizer.contains_general_knowledge(response):
                            response = (
                                "I don't have specific information about that in my knowledge base. "
                                "Could you try asking about a different topic?"
                            )
                        else:
                            # Only sanitize LLM-generated responses for file references
                            response = ResponseSanitizer.sanitize(response)

                    # Track conversation
                    self.conversation_history.append(
                        {"role": "user", "content": message}
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": response}
                    )

                    return response

        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an issue processing your request. Could you please try again?"

        return "I couldn't generate a response. Please try again."

    def run(self):
        """Run interactive chat."""
        print("\n" + "=" * 60)
        print("  📚 DOCUMENT-BASED SUPPORT ASSISTANT")
        print("=" * 60)
        print("\nI can ONLY answer questions from the indexed documents.")
        print("Type 'quit' to exit, 'topics' to see available topics.\n")

        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
            print(f"📁 {len(files)} document(s) loaded.\n")
        else:
            print("⚠️ No documents indexed. I won't be able to answer questions.\n")

        print("-" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye! Have a great day!\n")
                    break

                if user_input.lower() == "topics":
                    result = get_available_topics.invoke({})
                    try:
                        data = json.loads(result)
                        topics = data.get("topics", [])
                        if topics:
                            print("\n📋 Available topics:")
                            for topic in topics:
                                print(f"   • {topic}")
                        else:
                            print(
                                "\n📋 No specific topics detected in the knowledge base."
                            )
                    except:
                        print("\n📋 Could not retrieve topics.")
                    continue

                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = StrictDocumentChatbot()
    chatbot.run()
