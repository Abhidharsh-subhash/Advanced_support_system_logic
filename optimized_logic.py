import os
import re
import json
import operator
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Literal, Optional
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

    # Scenario detection settings
    MAX_DISAMBIGUATION_DEPTH = 3  # Kept, though LLM handles depth now
    MIN_SCENARIOS_FOR_DISAMBIGUATION = 2  # Kept for LLM prompt guidance
    MAX_SCENARIOS_TO_SHOW = 5  # Kept for LLM prompt guidance

    # Minimum confidence to trigger scenario disambiguation
    SCENARIO_CONFIDENCE_THRESHOLD = (
        0.8  # Kept, for potential future use or prompt guidance
    )


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
    DISAMBIGUATION = "disambiguation"


class ScenarioStatus(str, Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    NONE = "none"
    RESOLVED = "resolved"


# ========================================
# VECTOR STORE
# ========================================

FAISS_INDEX_PATH = "/home/abhidharsh-fgil/FGIL Projects/Convergent/vector_store/faiss"


def load_vector_store(index_path: str = FAISS_INDEX_PATH):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
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

    # Search state
    search_confidence: float
    search_quality: str
    has_searched: bool
    search_results: str  # Now stores raw search tool output
    found_relevant_info: bool
    best_match_score: float

    # Response control
    should_respond_not_found: bool
    not_found_message: str

    # Scenario/disambiguation state
    has_multiple_scenarios: bool
    detected_scenarios: List[dict]  # Scenarios identified by LLM
    scenario_status: str
    disambiguation_question: str
    selected_scenario: Optional[str]
    disambiguation_depth: int
    scenario_context: List[dict]
    awaiting_scenario_selection: bool
    filtered_search_results: str
    current_scenario_options: List[str]


# ========================================
# SEARCH RESULT ANALYZER (NON-LLM)
# ========================================


class SearchResultAnalyzer:
    """Analyzes search results and determines quality, without using an LLM."""

    @classmethod
    def analyze(cls, results: list, query: str) -> dict:
        """Analyze search results and determine quality."""
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

        scores = [float(score) for _, score in results]
        best_score = min(scores)
        # avg_score = sum(scores) / len(scores) # Not strictly needed for routing

        relevant_results_count = sum(1 for s in scores if s < Config.ACCEPTABLE_SCORE)

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

        should_respond_via_info = (
            quality != SearchQuality.NOT_FOUND.value
            and relevant_results_count >= Config.MIN_RELEVANT_RESULTS
        )

        if not should_respond_via_info:
            if quality == SearchQuality.NOT_FOUND.value:
                reason = "No relevant information found in knowledge base."
            elif relevant_results_count < Config.MIN_RELEVANT_RESULTS:
                reason = (
                    "Insufficient relevant results found to form a confident answer."
                )
            else:
                reason = "Low confidence in search results."
        else:
            reason = "Relevant information found."

        return {
            "found_relevant_info": should_respond_via_info,
            "confidence": confidence,
            "quality": quality,
            "best_score": best_score,
            # "avg_score": avg_score,
            "should_respond": should_respond_via_info,  # This indicates if *any* relevant info was found
            "reason": reason,
            "relevant_count": relevant_results_count,
        }


# ========================================
# NOT FOUND RESPONSE GENERATOR - Can be simplified, LLM will handle more now
# ========================================


class NotFoundResponseGenerator:
    """Generates appropriate responses when information is not found."""

    RESPONSES = {
        "general": [
            "I don't have information about that in my knowledge base. Could you try asking about a different topic?",
            "I couldn't find any relevant information about this topic. Is there something else I can help you with?",
            "Sorry, I don't have data about that. Would you like to ask about something else?",
        ],
        "partial": [
            "I found some related information, but nothing that directly answers your question. Would you like me to share what I found?",
        ],
        "suggest_rephrase": [
            "I couldn't find a match for your query. Could you try rephrasing or being more specific?",
        ],
    }

    @classmethod
    def generate(cls, search_analysis: dict, available_topics: List[str] = None) -> str:
        import random

        quality = search_analysis.get("quality", SearchQuality.NOT_FOUND.value)
        confidence = search_analysis.get("confidence", 0)

        if quality == SearchQuality.LOW.value and confidence > 0.1:
            response = random.choice(cls.RESPONSES["partial"])
        elif confidence == 0 or quality == SearchQuality.NOT_FOUND.value:
            response = random.choice(cls.RESPONSES["general"])
        else:  # Default for low but not not_found, or other cases
            response = random.choice(cls.RESPONSES["suggest_rephrase"])

        if available_topics and len(available_topics) > 0:
            topic_list = ", ".join(available_topics[:5])
            response += f"\n\nI can help you with topics like: {topic_list}."

        return response


# ========================================
# QUERY ANALYZER (SIMPLIFIED)
# ========================================


class QueryAnalyzer:
    """Analyzes user queries - simplified to avoid premature clarification."""

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
            "what's up",
            "yo",
        ]
        query_clean = query.lower().strip().rstrip("!.,")
        return any(g == query_clean for g in greetings)

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
            "finished",
            "im done",
        ]
        query_clean = query.lower().strip().rstrip("!.,")
        return any(c in query_clean for c in closings)

    @classmethod
    def is_too_short(cls, query: str) -> bool:
        """Check if query is too short to be meaningful."""
        return len(query.strip()) < 3

    @classmethod
    def is_scenario_selection(
        cls, query: str, available_options: List[str]
    ) -> Optional[str]:
        """Check if user's response is selecting a scenario from available options."""
        if not available_options:
            return None

        query_lower = query.lower().strip()

        # Check for numeric selection (1, 2, 3, etc.)
        if query_lower.isdigit():
            idx = int(query_lower) - 1
            if 0 <= idx < len(available_options):
                return available_options[idx]

        # Check for letter selection (a, b, c, etc.)
        if len(query_lower) == 1 and query_lower.isalpha():
            idx = ord(query_lower) - ord("a")
            if 0 <= idx < len(available_options):
                return available_options[idx]

        # Check for keyword match with options
        for option in available_options:
            option_lower = option.lower()
            if option_lower in query_lower or query_lower in option_lower:
                return option

            # Check for significant word overlap
            option_words = set(option_lower.split())
            query_words = set(query_lower.split())
            overlap = option_words.intersection(query_words)
            # Require at least one significant word, or a good proportion of words
            if len(overlap) >= 1 and len(overlap) / len(option_words) >= 0.5:
                return option

        return None


# ========================================
# TOOLS - MODIFIED TO REMOVE NESTED LLM CALLS
# ========================================


@tool
def search_documents(query: str) -> str:
    """
    Search the document database for relevant information.
    This tool performs the FAISS search and initial relevance analysis,
    but does NOT detect scenarios. Scenario detection is handled by the main agent LLM.

    Returns:
        JSON with search results, quality, confidence, and a list of relevant documents.
        Each document will contain 'content', 'relevance_score', and 'metadata'.
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No knowledge base available",
            }
        )

    try:
        results = VECTOR_STORE.similarity_search_with_score(query, k=5)
    except Exception as e:
        return json.dumps(
            {
                "found_answer": False,
                "error": str(e),
                "message": f"Error during search: {str(e)}",
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "message": "No information found for this query in the knowledge base.",
            }
        )

    analysis = SearchResultAnalyzer.analyze(results, query)

    documents_to_return = []
    # Only return documents that are considered at least moderately relevant for the LLM to process
    for doc, score in results:
        if float(score) < Config.ACCEPTABLE_SCORE:  # Filter out very irrelevant docs
            documents_to_return.append(
                {
                    "content": doc.page_content,
                    "relevance_score": float(score),
                    "metadata": doc.metadata,  # useful for context, but keep it concise
                }
            )

    # Sort documents by score (lower is better) to present the best ones first to the LLM
    documents_to_return.sort(key=lambda x: x["relevance_score"])

    # Ensure 'found_answer' reflects if any *relevant* documents were actually kept
    final_found_answer = analysis["should_respond"] and len(documents_to_return) > 0

    return json.dumps(
        {
            "found_answer": final_found_answer,
            "should_respond": final_found_answer,
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "documents": documents_to_return[
                : Config.MAX_SCENARIOS_TO_SHOW + 2
            ],  # Provide enough docs for LLM to reason, maybe a few more than scenarios
            "count": len(documents_to_return),
            "message": (
                analysis["reason"]
                if not final_found_answer
                else "Relevant documents found."
            ),
        }
    )


@tool
def get_scenario_answer(selected_scenario_context: str, original_query: str) -> str:
    """
    Refines search based on a user-selected scenario and provides a focused answer.
    This is called AFTER the user has selected a scenario from a disambiguation.
    It performs a new, more targeted search using the selected scenario.

    Args:
        selected_scenario_context: The user's selected scenario or the key phrase identified.
        original_query: The user's initial query.

    Returns:
        JSON string containing refined documents and the suggested message.
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {"found_answer": False, "message": "No knowledge base available"}
        )

    # Use the selected scenario and original query to form a more precise search
    refined_query = f"{original_query} {selected_scenario_context}"

    try:
        results = VECTOR_STORE.similarity_search_with_score(refined_query, k=5)
    except Exception as e:
        return json.dumps({"found_answer": False, "error": str(e)})

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "message": f"No specific information found for '{selected_scenario_context}' after refining search.",
                "documents": [],
            }
        )

    filtered_docs = []
    # Filter based on relevance and potentially keyword match with the scenario
    scenario_keywords = set(selected_scenario_context.lower().split())
    for doc, score in results:
        content_lower = doc.page_content.lower()
        keyword_matches = sum(
            1 for kw in scenario_keywords if kw in content_lower and len(kw) > 2
        )  # require longer keywords

        # Only include documents that are highly relevant OR strongly match the selected scenario
        if float(score) < Config.GOOD_SCORE or keyword_matches > 0:
            filtered_docs.append(
                {
                    "content": doc.page_content,
                    "relevance_score": float(score),
                    "metadata": doc.metadata,
                }
            )

    filtered_docs.sort(key=lambda x: x["relevance_score"])

    if not filtered_docs:
        return json.dumps(
            {
                "found_answer": False,
                "message": f"Could not find specific relevant details for the selected scenario: {selected_scenario_context}.",
                "documents": [],
            }
        )

    return json.dumps(
        {
            "found_answer": True,
            "selected_scenario": selected_scenario_context,
            "documents": filtered_docs[:3],  # Provide top 3 most relevant filtered docs
            "should_respond": True,
            "message": "Refined search results for selected scenario.",
        }
    )


@tool
def get_available_topics() -> str:
    """Get list of topics available in the knowledge base."""
    if VECTOR_STORE is None:
        return json.dumps({"topics": [], "message": "No knowledge base available"})

    try:
        docs = VECTOR_STORE.similarity_search(
            "", k=100
        )  # Search for general terms to get diverse docs
        all_text = " ".join(doc.page_content.lower() for doc in docs)

        topic_keywords = {
            "User Management": ["user", "account", "profile", "permission", "role"],
            "Authentication": [
                "login",
                "password",
                "sso",
                "authentication",
                "sign in",
                "security",
            ],
            "Settings": [
                "settings",
                "configure",
                "setup",
                "preferences",
                "customization",
            ],
            "Billing": [
                "invoice",
                "payment",
                "billing",
                "subscription",
                "plan",
                "charge",
            ],
            "Bookings": ["booking", "reservation", "travel", "trip", "flight", "hotel"],
            "Reports": [
                "report",
                "analytics",
                "dashboard",
                "export",
                "data",
                "metrics",
            ],
            "Integration": [
                "api",
                "integration",
                "sync",
                "webhook",
                "connect",
                "third-party",
            ],
            "Troubleshooting": ["error", "issue", "troubleshoot", "fix", "problem"],
        }

        available_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text for kw in keywords):
                available_topics.append(topic)

        if not available_topics:  # Fallback if no specific topics detected
            available_topics = ["general topics related to business operations"]

        return json.dumps({"topics": available_topics, "count": len(available_topics)})
    except Exception as e:
        return json.dumps({"topics": [], "error": str(e)})


# ========================================
# SETUP
# ========================================

tools = [
    search_documents,
    get_scenario_answer,
    get_available_topics,
]  # Updated tool name

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


# UPDATED SYSTEM PROMPT - LLM DOES SCENARIO DETECTION
SYSTEM_PROMPT = """You are a document-based support assistant. You ONLY provide information from the provided context and search results.

## CRITICAL WORKFLOW:

### STEP 1: INITIAL USER QUERY HANDLING
For ANY user message, analyze its nature:
- If it's a greeting (e.g., "hi", "hello"), respond appropriately (e.g., "Hello! How can I assist you?").
- If it's a closing (e.g., "bye", "thank you"), respond appropriately (e.g., "You're welcome! Goodbye!").
- If it's too short or ambiguous (e.g., "what?"), ask for clarification.
- Otherwise, for any substantive query, you MUST proceed to search.

### STEP 2: SEARCHING FOR INFORMATION
- For a substantive user query, immediately call the `search_documents` tool with the user's query.
- DO NOT generate a response before searching.

### STEP 3: INTERPRET SEARCH RESULTS (LLM's Core Task)
After calling `search_documents` and receiving the `ToolMessage` containing `documents` (or a message indicating no results):

**A. If `search_documents` returned `found_answer: false` or no relevant `documents`:**
- Generate a "not found" response. Use the `message` from the tool output if it provides a specific reason.
- Example: "I don't have information about that in my knowledge base. Is there something else I can help you with?"

**B. If `search_documents` returned `found_answer: true` and relevant `documents` (provided as 'documents' in the ToolMessage content):**
    You must now analyze the `content` of these provided `documents` to determine the best response.

    **SUB-STEP B1: SCENARIO DETECTION**
    - Carefully read the `content` of the provided `documents`.
    - Determine if the documents describe MULTIPLE DISTINCT procedures, options, or cases (scenarios) that require the user to choose between them.
    - Multiple scenarios exist ONLY if the content EXPLICITLY mentions different cases, e.g., "For users with X, do this... For users with Y, do that...", "Option A: ... Option B: ...", "If you are an admin... If you are a regular user...", or clearly separate steps for different roles/conditions.
    - **CRITICAL:** Do NOT guess or infer scenarios. Only identify them if they are explicitly presented as distinct choices in the document content. If the content describes ONE process/procedure, there is NO disambiguation needed.

    **SUB-STEP B2: FORMULATE RESPONSE**

    **IF you detect {MIN_SCENARIOS_FOR_DISAMBIGUATION} or more distinct scenarios (from B1):**
    - Your response MUST be a JSON object with the following structure:
      ```json
      {{
          "response_type": "disambiguation_needed",
          "question": "I found information about a few different scenarios. Which one applies to your situation?",
          "scenarios": [
              {{"id": "[unique_key]", "title": "Scenario A: [brief title from document]", "description": "[brief description from document]"}},
              {{"id": "[unique_key]", "title": "Scenario B: [brief title from document]", "description": "[brief description from document]"}}
          ]
      }}
      ```
    - The `id` should be a unique identifier (e.g., "scenario_A", "admin_flow") for internal tracking, `title` a user-friendly name, and `description` a short summary, ALL derived directly from the document content.
    - Limit to {MAX_SCENARIOS_TO_SHOW} scenarios.

    **IF you DO NOT detect multiple distinct scenarios (from B1), or if only one is clearly applicable:**
    - Generate a direct, comprehensive answer to the user's original query using ONLY the information from the provided `documents`.
    - Be conversational, helpful, and concise.

### STEP 4: AFTER USER SELECTS SCENARIO (If applicable)
- If the user's previous input was a scenario selection (after you presented options), you will see their selected scenario in the state.
- IMMEDIATELY call the `get_scenario_answer` tool, passing the `selected_scenario_context` and the `original_query`.
- Then, interpret the results from `get_scenario_answer` to provide a focused answer based on the refined information. This will typically be a direct answer.

## ABSOLUTE RULES:

1.  **NEVER ask for clarification BEFORE searching** unless the query is extremely short/ambiguous (e.g., one word).
2.  **NEVER speculate about scenarios that might exist.** Only use those EXPLICITLY found in the `documents`.
3.  **If a direct answer can be formed from the documents, provide it.** Do not force disambiguation if not clearly necessary.
4.  **If content is NOT found, say so clearly.**
5.  **Be conversational and helpful.**
6.  **Do not mention "documents" or "search results"** in your final user-facing responses. Present information naturally.
7.  **Always prefer tool calls when appropriate** (e.g., `search_documents` for a query, `get_scenario_answer` after scenario selection). If you have enough information after a tool call, provide a direct answer.

## CONTEXTUAL CLUES FROM STATE:
- `awaiting_scenario_selection`: {awaiting_scenario_selection_status}
- `selected_scenario`: {selected_scenario_value}
"""


# ========================================
# GRAPH NODES - MODIFIED
# ========================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input - for initial routing (greeting, closing, too short)."""
    messages = state["messages"]

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"interaction_mode": InteractionMode.QUERY.value}

    # If we are awaiting scenario selection, the LLM (`agent` node) will handle parsing user input.
    # We only set the mode if it's a clear greeting/closing here.
    if state.get("awaiting_scenario_selection", False):
        # We need to detect if user's response IS a scenario selection to update `selected_scenario`
        # for the next `agent` call. `QueryAnalyzer.is_scenario_selection` is key here.
        current_options = state.get("current_scenario_options", [])
        selected = QueryAnalyzer.is_scenario_selection(user_message, current_options)

        if selected:
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": selected,
                # Keep awaiting_scenario_selection=True. The 'agent' LLM will process this selection
                # and then set awaiting_scenario_selection=False after calling the tool.
            }
        else:
            # If not a clear selection, let the agent LLM decide (re-prompt or new query)
            # We don't change mode here, let agent handle
            return {}

    # Check for greetings/closings for direct routing
    if QueryAnalyzer.is_greeting(user_message):
        return {"interaction_mode": InteractionMode.GREETING.value}

    if QueryAnalyzer.is_closing(user_message):
        return {"interaction_mode": InteractionMode.CLOSING.value}

    # Check for too short
    if QueryAnalyzer.is_too_short(user_message):
        return {
            "interaction_mode": InteractionMode.CLARIFICATION.value,
            "clarification_needed": True,
            "follow_up_questions": [
                "Could you please tell me more about what you're looking for?"
            ],
        }

    # Default: proceed to main agent logic (which will likely search)
    return {
        "interaction_mode": InteractionMode.QUERY.value,
        "original_query": user_message,
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    import random

    greetings = [
        "Hello! I'm here to help you find information. What would you like to know?",
        "Hi there! How can I assist you today?",
        "Hey! I can answer questions based on available documentation. What do you need?",
    ]
    return {"messages": [AIMessage(content=random.choice(greetings))]}


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    import random

    closings = [
        "Goodbye! Feel free to come back if you have more questions.",
        "Happy to help! Take care!",
        "Glad I could assist! Have a great day!",
    ]
    return {"messages": [AIMessage(content=random.choice(closings))]}


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification for very short/unclear queries."""
    follow_up = state.get("follow_up_questions", ["What would you like to know?"])
    return {
        "messages": [AIMessage(content=follow_up[0])],
        "pending_clarification": True,
    }


def agent(state: AgentState) -> dict:
    """
    Main agent node. This LLM call now orchestrates tool calls and interprets results.
    It takes on the role of scenario detection previously done by a separate LLM call.
    """
    messages = state["messages"]

    # Retrieve relevant state for prompt context
    awaiting_scenario = state.get("awaiting_scenario_selection", False)
    selected_scenario_val = state.get("selected_scenario", "None")

    # This is critical for the LLM to understand what to do next
    current_human_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            current_human_message = msg.content
            break
        elif isinstance(
            msg, ToolMessage
        ):  # If it's a tool output, the LLM needs to process it
            pass  # Keep it in messages for the LLM to see

    # Inject dynamic state into the system prompt
    final_system_prompt = SYSTEM_PROMPT.format(
        MIN_SCENARIOS_FOR_DISAMBIGUATION=Config.MIN_SCENARIOS_FOR_DISAMBIGUATION,
        MAX_SCENARIOS_TO_SHOW=Config.MAX_SCENARIOS_TO_SHOW,
        awaiting_scenario_selection_status=str(awaiting_scenario),
        selected_scenario_value=selected_scenario_val,
    )

    # Make the LLM call
    # The LLM will either call a tool, or generate a structured JSON for disambiguation, or a direct answer.
    response = llm_with_tools.invoke(
        [SystemMessage(content=final_system_prompt)] + list(messages)
    )

    # Store search results from previous tool call if present, so get_scenario_answer can use them
    # This assumes search_results is added to state by validate_llm_output after search_documents completes
    new_state = {"messages": [response]}
    if selected_scenario_val != "None":
        new_state["selected_scenario"] = (
            selected_scenario_val  # Preserve it for potential next step
        )
        # Reset awaiting_scenario_selection here, as agent will now *act* on the selection
        new_state["awaiting_scenario_selection"] = False

    return new_state


def validate_llm_output(state: AgentState) -> dict:
    """
    Validates the LLM's response (either tool call or direct answer/disambiguation JSON)
    and updates the state for routing.
    This replaces the old `validate_and_route`.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Case 1: The LLM decided to call a tool (e.g., search_documents, get_scenario_answer)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # If the tool call was `search_documents`, store its raw output for potential `get_scenario_answer` later
        if last_message.tool_calls[0].function == "search_documents":
            # This node should not store the *output* of the tool, but rather the *call itself*.
            # The ToolNode will execute it and put its output in the messages list.
            # We'll rely on the agent to see the ToolMessage.
            pass  # No state change needed here for tool call; graph handles it
        return (
            {}
        )  # No specific state update to change routing logic at this point, just proceed to tools

    # Case 2: The LLM generated a direct response, potentially a structured JSON for disambiguation
    content = last_message.content
    try:
        parsed_content = json.loads(content)
        if parsed_content.get("response_type") == "disambiguation_needed":
            scenarios_data = parsed_content.get("scenarios", [])
            options = [
                s.get("title", f"Scenario {i+1}") for i, s in enumerate(scenarios_data)
            ]
            question = parsed_content.get("question", "Which scenario applies?")

            return {
                "has_multiple_scenarios": True,
                "detected_scenarios": scenarios_data,
                "disambiguation_question": question,
                "awaiting_scenario_selection": True,  # Now we need user input
                "current_scenario_options": options,
                "should_respond_not_found": False,
                # The 'search_results' from the last tool call should already be in state if agent processed it.
            }
        else:
            # If the LLM returned JSON but not for disambiguation (e.g., internal debug JSON),
            # treat it as a final text response.
            return {
                "found_relevant_info": True
            }  # LLM produced an answer, so info was found
    except json.JSONDecodeError:
        # The LLM generated a direct text response (not JSON for disambiguation)
        # This could be a direct answer, a "not found" message, a greeting, or a closing.
        # We need to explicitly check for "not found" state.
        # Other cases (greeting, closing, direct answer) will route to END.
        # The `agent`'s SYSTEM_PROMPT should guide it to set interaction_mode if it's a "not found" case.

        # If LLM *directly* generated a "not found" message as a final answer
        if state.get("interaction_mode") == InteractionMode.NOT_FOUND.value:
            return {"should_respond_not_found": True, "not_found_message": content}

        # For other direct text responses (greeting, closing, actual answer),
        # no special routing flags are needed here as they proceed to END via should_continue.
        return {"found_relevant_info": True}


def handle_not_found(state: AgentState) -> dict:
    """Handle case when LLM generated a "not found" response."""
    # The message is now directly from the LLM or generated by NotFoundResponseGenerator if initial search failed.
    message = state.get("not_found_message")
    if not message:  # Fallback
        # Attempt to use last LLM message if not specifically set in state
        last_llm_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
                last_llm_msg = msg.content
                break
        message = last_llm_msg or NotFoundResponseGenerator.generate(
            {"quality": SearchQuality.NOT_FOUND.value, "confidence": 0},
            state.get("detected_topics", []),
        )
    return {
        "messages": [AIMessage(content=message)],
        "interaction_mode": InteractionMode.NOT_FOUND.value,
    }


def present_scenarios(state: AgentState) -> dict:
    """Present scenario options when disambiguation is needed, using LLM-generated JSON."""
    question = state.get(
        "disambiguation_question",
        "I found multiple related scenarios. Could you specify which one you're asking about?",
    )
    scenarios_list = state.get("detected_scenarios", [])
    if scenarios_list:
        formatted_options = []
        for i, scenario in enumerate(scenarios_list):
            title = scenario.get("title", f"Option {i+1}")
            desc = scenario.get("description", "")
            formatted_options.append(f"{i+1}. **{title}**\n   {desc}")
        question += "\n\n" + "\n".join(formatted_options)
    else:
        question += (
            "\n(No specific options could be detailed, please clarify your need.)"
        )

    return {
        "messages": [AIMessage(content=question)],
        "awaiting_scenario_selection": True,  # Ensure we are waiting
    }


# ========================================
# ROUTING FUNCTIONS - MODIFIED
# ========================================


def should_continue(
    state: AgentState,
) -> Literal["tools", "disambiguation", "not_found", "end"]:
    """
    Determine if we should continue to tools, disambiguation, not_found, or end.
    This function now routes based on `validate_llm_output`'s state updates or tool calls.
    """
    last_message = state["messages"][-1]

    # If the LLM returned a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If `validate_llm_output` detected disambiguation is needed
    if state.get("has_multiple_scenarios", False) and state.get(
        "awaiting_scenario_selection", False
    ):
        return "disambiguation"

    # If `validate_llm_output` (or the LLM's direct message) indicates not found
    if state.get("should_respond_not_found", False):
        return "not_found"

    # Otherwise, the LLM has generated a final answer, greeting, or closing
    return "end"


def route_after_analysis(
    state: AgentState,
) -> Literal["handle_greeting", "handle_closing", "ask_clarification", "agent"]:
    """Route based on initial input analysis (greeting, closing, too short)"""
    mode = state.get("interaction_mode", InteractionMode.QUERY.value)

    if mode == InteractionMode.GREETING.value:
        return "handle_greeting"
    if mode == InteractionMode.CLOSING.value:
        return "handle_closing"
    if mode == InteractionMode.CLARIFICATION.value:
        return "ask_clarification"

    # For `DISAMBIGUATION` mode (user selected scenario or needs re-prompt), or `QUERY` mode:
    # Always send to the `agent` node. The `agent`'s prompt is sophisticated enough
    # to handle whether it needs to search, or call `get_scenario_answer` based on state.
    return "agent"


# ========================================
# RESPONSE SANITIZER
# ========================================


class ResponseSanitizer:
    """Sanitize responses to remove file references and general knowledge markers."""

    FILE_PATTERNS = [
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml|html?|md)\b",
        r"\(source:\s*[^)]+\)",
        r"source:\s*[\w\-\.]+",
        r"(?i)according to the [\w\s]+ document",
        r"(?i)based on the provided document\(s\)",
        r"(?i)information from the knowledge base",
    ]

    @classmethod
    def sanitize(cls, response: str) -> str:
        if not response:
            return response
        sanitized = response
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"\s{2,}", " ", sanitized
        )  # Replace multiple spaces with single
        sanitized = re.sub(
            r"\s+([.,!?])", r"\1", sanitized
        )  # Remove space before punctuation
        return sanitized.strip()


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
    workflow.add_node("agent", agent)  # Main LLM decision/reasoning node
    workflow.add_node("tools", ToolNode(tools))  # Executes actual tools
    workflow.add_node(
        "validate_llm_output", validate_llm_output
    )  # Parses LLM's response for routing
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("present_scenarios", present_scenarios)

    # Entry point
    workflow.add_edge(START, "analyze_input")

    # Route after initial input analysis (direct responses or to agent)
    workflow.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        {
            "handle_greeting": "handle_greeting",
            "handle_closing": "handle_closing",
            "ask_clarification": "ask_clarification",
            "agent": "agent",  # All substantive queries go to the main agent LLM
        },
    )

    # Terminal nodes (or nodes that output a message and wait for user)
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_closing", END)
    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("handle_not_found", END)
    workflow.add_edge(
        "present_scenarios", END
    )  # Presents options, waits for user input

    # Main agent logic flow:
    # Agent node (LLM) makes its decision. Its output is then validated.
    workflow.add_edge("agent", "validate_llm_output")

    # After LLM output validation, decide next step (tool call, disambiguation, not found, or end)
    workflow.add_conditional_edges(
        "validate_llm_output",
        should_continue,
        {
            "tools": "tools",  # LLM decided to call a tool
            "disambiguation": "present_scenarios",  # LLM detected scenarios and returned JSON
            "not_found": "handle_not_found",  # LLM directly indicated no info or `should_respond_not_found` was set
            "end": END,  # LLM generated a final answer
        },
    )

    # After a tool is executed, control always returns to the `agent` node.
    # The `agent` (LLM) will then process the ToolMessage output and decide the next action.
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========================================
# CHATBOT CLASS
# ========================================


class SmartChatbot:
    """Chatbot that searches first, then disambiguates only when necessary, with optimized LLM calls."""

    def __init__(self):
        self.agent = create_agent()
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {}  # General context, not used much in current implementation
        self.topic_history = []
        self.conversation_history = []

        # Scenario tracking is now more reliant on state in LangGraph
        # These are for UI/input parsing convenience.
        self.awaiting_scenario_selection = False
        self.current_scenario_options = []
        self.original_query = None
        self.search_results = (
            None  # This will store the raw output from search_documents for re-use
        )

    def chat(self, message: str) -> str:
        """Process a chat message."""
        config = {"configurable": {"thread_id": self.thread_id}}

        # Restore state for the current turn based on last run's output
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,  # Reset per turn, or manage in graph
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": False,
            "original_query": self.original_query
            or message,  # Preserve original query across turns
            "clarification_attempts": 0,
            "user_intent": "",
            "detected_topics": [],
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,  # Default, graph will update
            "conversation_history": self.conversation_history,
            "topic_history": self.topic_history,
            "search_confidence": 0.0,
            "search_quality": "",
            "has_searched": False,
            "search_results": self.search_results or "",
            "found_relevant_info": False,
            "best_match_score": float("inf"),
            "should_respond_not_found": False,
            "not_found_message": "",
            "has_multiple_scenarios": False,
            "detected_scenarios": [],
            "scenario_status": ScenarioStatus.SINGLE.value,
            "disambiguation_question": "",
            "selected_scenario": None,  # Will be set by analyze_input if user selects
            "disambiguation_depth": 0,
            "scenario_context": [],
            "awaiting_scenario_selection": self.awaiting_scenario_selection,  # Restore for current run
            "filtered_search_results": "",
            "current_scenario_options": self.current_scenario_options,  # Restore for current run
        }

        try:
            # Invoke the graph with the current state
            result = self.agent.invoke(initial_state, config=config)

            # Update chatbot's internal state based on the graph's final state
            self.awaiting_scenario_selection = result.get(
                "awaiting_scenario_selection", False
            )
            self.current_scenario_options = result.get("current_scenario_options", [])
            self.original_query = result.get("original_query", message)  # Keep updated
            self.search_results = result.get(
                "search_results", self.search_results
            )  # Persist search results

            # Get the final AI message content
            final_response = "I couldn't generate a response. Please try again."
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    final_response = ResponseSanitizer.sanitize(msg.content)
                    break
                elif isinstance(msg, ToolMessage) and msg.name == "search_documents":
                    # If the last message is a search tool output, it means the agent is about to process it.
                    # We might want to just return a pending message or internal state for debugging.
                    # For user-facing, we need the *next* LLM message.
                    # This case means the graph hasn't completed a full cycle to a user-facing message yet.
                    pass

            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append(
                {"role": "assistant", "content": final_response}
            )

            return final_response

        except Exception as e:
            print(f"Error during chat: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an issue. Please try again."

    def run(self):
        """Run interactive chat."""
        print("\n" + "=" * 60)
        print("  🤖 SMART SUPPORT ASSISTANT (Optimized)")
        print("=" * 60)
        print("\nAsk me anything! I'll search my knowledge base first.")
        print("Type 'quit' to exit, 'topics' for available topics.\n")

        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(
                doc.metadata.get("filename", "Unknown") for doc in docs if doc.metadata
            )
            print(f"📁 {len(files)} unique document(s) loaded from FAISS index.\n")
        else:
            print("⚠️ No documents indexed.\n")

        print("-" * 60)

        while True:
            try:
                prompt = "\n👤 You: "
                if self.awaiting_scenario_selection:
                    # If waiting for selection, previous message should have detailed options
                    prompt = "\n👤 You (select an option or new query): "

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!\n")
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
                            print("\n📋 No specific topics detected.")
                    except:
                        print("\n📋 Could not retrieve topics.")
                    continue

                response = self.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}\n")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = SmartChatbot()
    chatbot.run()
