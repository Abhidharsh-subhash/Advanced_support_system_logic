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
    MAX_DISAMBIGUATION_DEPTH = 3
    MIN_SCENARIOS_FOR_DISAMBIGUATION = 2
    MAX_SCENARIOS_TO_SHOW = 5

    # Minimum confidence to trigger scenario disambiguation
    SCENARIO_CONFIDENCE_THRESHOLD = 0.8


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
    search_results: str
    found_relevant_info: bool
    best_match_score: float

    # Response control
    should_respond_not_found: bool
    not_found_message: str

    # Scenario/disambiguation state
    has_multiple_scenarios: bool
    detected_scenarios: List[dict]
    scenario_status: str
    disambiguation_question: str
    selected_scenario: Optional[str]
    disambiguation_depth: int
    scenario_context: List[dict]
    awaiting_scenario_selection: bool
    filtered_search_results: str
    current_scenario_options: List[str]


# ========================================
# SEARCH RESULT ANALYZER
# ========================================


class SearchResultAnalyzer:
    """Analyzes search results to determine if we have valid information."""

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
        avg_score = sum(scores) / len(scores)

        relevant_results = [s for s in scores if s < Config.ACCEPTABLE_SCORE]
        relevant_count = len(relevant_results)

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

        query_keywords = set(query.lower().split())
        keyword_matches = 0

        for doc, _ in results:
            content_lower = doc.page_content.lower()
            matches = sum(
                1 for kw in query_keywords if kw in content_lower and len(kw) > 3
            )
            keyword_matches = max(keyword_matches, matches)

        keyword_relevance = keyword_matches / max(len(query_keywords), 1)

        should_respond = (
            quality != SearchQuality.NOT_FOUND.value
            and relevant_count >= Config.MIN_RELEVANT_RESULTS
            and (
                confidence > Config.LOW_CONFIDENCE_THRESHOLD or keyword_relevance > 0.3
            )
        )

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


# ========================================
# NOT FOUND RESPONSE GENERATOR
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
    def generate(
        cls, query: str, search_analysis: dict, available_topics: List[str] = None
    ) -> str:
        import random

        quality = search_analysis.get("quality", SearchQuality.NOT_FOUND.value)
        confidence = search_analysis.get("confidence", 0)

        if quality == SearchQuality.LOW.value and confidence > 0.1:
            response = random.choice(cls.RESPONSES["partial"])
        elif confidence == 0:
            response = random.choice(cls.RESPONSES["general"])
        else:
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
            if len(overlap) >= min(2, len(option_words) // 2 + 1):
                return option

        return None


# ========================================
# SCENARIO DETECTOR (STRICT - CONTENT-BASED ONLY)
# ========================================


class ScenarioDetector:
    """
    Detects multiple scenarios ONLY from actual content in search results.
    Does NOT speculate about scenarios that might exist.
    """

    DETECTION_PROMPT = """Analyze the following search results and determine if they contain MULTIPLE DISTINCT scenarios that the user needs to choose between.

IMPORTANT RULES:
1. ONLY identify scenarios that are EXPLICITLY mentioned in the search results
2. DO NOT guess or infer scenarios that are not in the content
3. If the content describes ONE process/procedure, there is NO disambiguation needed
4. Multiple scenarios exist ONLY if the content explicitly mentions different cases like:
   - "For users with X, do this... For users with Y, do that..."
   - "Option A: ... Option B: ..."
   - "If you are an admin... If you are a regular user..."

USER QUERY: {query}

SEARCH RESULTS CONTENT:
{search_results}

RESPOND IN JSON FORMAT:
{{
    "has_multiple_scenarios": true/false,
    "scenarios": [
        {{
            "id": "scenario_1",
            "title": "<brief title from content>",
            "description": "<description from content>",
            "exact_quote": "<quote from content that describes this scenario>"
        }}
    ],
    "disambiguation_needed": true/false,
    "suggested_question": "<question ONLY if disambiguation_needed is true>",
    "reason": "<explanation>"
}}

CRITICAL: Set has_multiple_scenarios to FALSE unless the content EXPLICITLY contains multiple distinct procedures/options that require the user to choose.
"""

    @classmethod
    def detect(cls, query: str, search_results: List[dict], llm: ChatOpenAI) -> dict:
        """Detect if search results contain multiple scenarios."""
        formatted_results = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        prompt = cls.DETECTION_PROMPT.format(
            query=query, search_results=formatted_results
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a strict scenario detector. Only identify scenarios EXPLICITLY present in the content. Never speculate."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Additional validation - require exact quotes for scenarios
            scenarios = result.get("scenarios", [])
            valid_scenarios = [s for s in scenarios if s.get("exact_quote")]

            if len(valid_scenarios) < Config.MIN_SCENARIOS_FOR_DISAMBIGUATION:
                result["has_multiple_scenarios"] = False
                result["disambiguation_needed"] = False

            result["scenarios"] = valid_scenarios
            return result

        except (json.JSONDecodeError, Exception) as e:
            return {
                "has_multiple_scenarios": False,
                "scenarios": [],
                "disambiguation_needed": False,
                "suggested_question": "",
                "reason": f"Error: {str(e)}",
            }


# ========================================
# TOOLS
# ========================================


@tool
def search_and_analyze(query: str) -> str:
    """
    Search the document database and analyze if multiple scenarios exist.
    This is the primary tool - always call this first.

    Returns:
        JSON with search results and scenario analysis
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
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
                "has_multiple_scenarios": False,
            }
        )

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No information found for this query in the knowledge base.",
            }
        )

    # Analyze search quality
    analysis = SearchResultAnalyzer.analyze(results, query)

    if not analysis["should_respond"]:
        return json.dumps(
            {
                "found_answer": False,
                "quality": analysis["quality"],
                "confidence": float(analysis["confidence"]),
                "documents": [],
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No relevant information found for this query.",
                "reason": analysis["reason"],
            }
        )

    # Prepare documents
    documents = []
    for doc, score in results:
        if float(score) < Config.ACCEPTABLE_SCORE:
            documents.append(
                {
                    "content": doc.page_content,
                    "relevance": (
                        "high" if float(score) < Config.GOOD_SCORE else "medium"
                    ),
                    "score": float(score),
                }
            )

    if not documents:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "has_multiple_scenarios": False,
                "disambiguation_needed": False,
                "message": "No relevant information found.",
            }
        )

    # Now check for multiple scenarios in the ACTUAL content
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    scenario_result = ScenarioDetector.detect(query, documents, llm)

    has_multiple = scenario_result.get("has_multiple_scenarios", False)
    disambiguation_needed = scenario_result.get("disambiguation_needed", False)

    return json.dumps(
        {
            "found_answer": True,
            "should_respond": True,
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "documents": documents,
            "count": len(documents),
            # Scenario info
            "has_multiple_scenarios": has_multiple,
            "disambiguation_needed": disambiguation_needed,
            "scenarios": scenario_result.get("scenarios", []),
            "disambiguation_question": (
                scenario_result.get("suggested_question", "")
                if disambiguation_needed
                else ""
            ),
        }
    )


@tool
def get_scenario_answer(
    search_results_json: str, selected_scenario: str, original_query: str
) -> str:
    """
    Get answer for a specific scenario after user selection.
    Call this after user selects a scenario from disambiguation.

    Args:
        search_results_json: Previous search results
        selected_scenario: User's selected scenario
        original_query: Original question
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {"found_answer": False, "message": "No knowledge base available"}
        )

    try:
        original_data = json.loads(search_results_json)
    except json.JSONDecodeError:
        original_data = {"documents": []}

    # Enhanced search with scenario context
    enhanced_query = f"{original_query} {selected_scenario}"

    try:
        results = VECTOR_STORE.similarity_search_with_score(enhanced_query, k=5)
    except Exception as e:
        return json.dumps({"found_answer": False, "error": str(e)})

    if not results:
        return json.dumps(
            {
                "found_answer": False,
                "message": f"No specific information found for '{selected_scenario}'.",
            }
        )

    # Filter for scenario relevance
    scenario_keywords = set(selected_scenario.lower().split())
    filtered_docs = []

    for doc, score in results:
        content_lower = doc.page_content.lower()
        keyword_matches = sum(
            1 for kw in scenario_keywords if kw in content_lower and len(kw) > 3
        )

        if keyword_matches > 0 or float(score) < Config.GOOD_SCORE:
            filtered_docs.append(
                {
                    "content": doc.page_content,
                    "relevance": "high" if keyword_matches >= 2 else "medium",
                    "score": float(score),
                }
            )

    filtered_docs.sort(key=lambda x: x["score"])

    return json.dumps(
        {
            "found_answer": len(filtered_docs) > 0,
            "selected_scenario": selected_scenario,
            "documents": filtered_docs[:3],
            "should_respond": len(filtered_docs) > 0,
        }
    )


@tool
def get_available_topics() -> str:
    """Get list of topics available in the knowledge base."""
    if VECTOR_STORE is None:
        return json.dumps({"topics": [], "message": "No knowledge base available"})

    try:
        docs = VECTOR_STORE.similarity_search("", k=100)
        all_text = " ".join(doc.page_content.lower() for doc in docs)

        topic_keywords = {
            "User Management": ["user", "account", "profile", "permission"],
            "Authentication": ["login", "password", "sso", "authentication"],
            "Settings": ["settings", "configure", "setup", "preferences"],
            "Billing": ["invoice", "payment", "billing", "subscription"],
            "Bookings": ["booking", "reservation", "travel", "trip"],
            "Reports": ["report", "analytics", "dashboard", "export"],
            "Integration": ["api", "integration", "sync", "webhook"],
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

tools = [search_and_analyze, get_scenario_answer, get_available_topics]

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)


# UPDATED SYSTEM PROMPT - SEARCH FIRST, NO SPECULATION
SYSTEM_PROMPT = """You are a highly specialized document-based support assistant. Your SOLE purpose is to provide information by STRICTLY and ONLY extracting or directly quoting content from the search results provided.

## CRITICAL WORKFLOW:

### STEP 1: ALWAYS SEARCH FIRST
For ANY user question, IMMEDIATELY call `search_and_analyze` tool.
DO NOT ask clarifying questions BEFORE searching.
DO NOT speculate about what scenarios might exist.

### STEP 2: INTERPRET TOOL RESULTS
After getting results from `search_and_analyze` or `get_scenario_answer` (these results will contain document snippets in their `documents` array):

**If the tool result indicates `found_answer: false` (from either tool):**
- Respond: "I don't have information about that in my knowledge base."
- Avoid offering further assistance unless explicitly asked, to remain strictly focused.

**If the tool result indicates `found_answer: true` AND `disambiguation_needed: false` (from `search_and_analyze`), OR if results come from `get_scenario_answer`:**
- Your task is to generate an answer by PRECISELY extracting or directly rephrasing specific sentences or bullet points that are EXPLICITLY and DIRECTLY stated within the `content` field of the `documents` from the tool's output.
- **ABSOLUTELY DO NOT:**
    - Introduce any external knowledge or information not present in the provided document snippets.
    - Make inferences, deductions, or assumptions.
    - Elaborate, expand, or add examples beyond what is explicitly written in the `documents` content.
    - Provide any information that is not directly traceable to the provided `documents` content.
- If the provided `documents` content does not contain a direct answer to a specific part of the user's question, you must explicitly state that "The available information does not specify..." or "I couldn't find details on..." for that particular aspect.
- Be concise, factual, and extremely literal in your interpretation and presentation of the source material.

**If the tool result indicates `found_answer: true` AND `disambiguation_needed: true` (from `search_and_analyze`):**
- Present the specific scenarios found (from the `scenarios` array in the tool output) EXACTLY as they are described (using titles and descriptions).
- Ask the user to choose which applies to them, using the `disambiguation_question` if provided, or construct a direct and clear question from the `scenarios` titles/descriptions.
- You MUST wait for their response.

### STEP 3: AFTER USER SELECTS SCENARIO
- The `get_scenario_answer` tool will be called automatically.
- Once its results are returned (a `ToolMessage` containing `documents`), apply the same strict extraction rules as described above for `found_answer: true`.

## ABSOLUTE, UNCOMPROMISING RULES FOR ALL RESPONSES:

1. **NEVER ask for clarification BEFORE searching.**
2. **NEVER speculate about scenarios that might exist; only present what tools explicitly report.**
3. **ONLY use information EXPLICITLY stated in the provided tool output's `documents` content for answers.**
   - Refer to the "ABSOLUTELY DO NOT" section above for detailed prohibitions.
   - If information is not in the documents, state its absence.
4. **If content is NOT found, say so clearly and briefly.**
5. **DO NOT mention "documents", "search results", "knowledge base", "according to the documents", "based on my information" etc. when giving an answer.** Just provide the extracted information naturally, but strictly from the source.
6. When asking for clarification (only for disambiguation), be direct and clear.
7. Maintain a neutral, factual, and objective tone.
"""

# ========================================
# GRAPH NODES
# ========================================


def analyze_input(state: AgentState) -> dict:
    """Analyze user input - simplified to avoid premature clarification."""
    messages = state["messages"]

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"interaction_mode": InteractionMode.QUERY.value}

    # Check if we're awaiting scenario selection
    if state.get("awaiting_scenario_selection", False):
        current_options = state.get("current_scenario_options", [])
        selected = QueryAnalyzer.is_scenario_selection(user_message, current_options)

        if selected:
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": selected,
                "awaiting_scenario_selection": False,
            }
        else:
            # User's response didn't match options - treat as new query
            return {
                "interaction_mode": InteractionMode.QUERY.value,
                "awaiting_scenario_selection": False,
                "selected_scenario": user_message,  # Use their response as context
            }

    # Check for greetings
    if QueryAnalyzer.is_greeting(user_message):
        return {"interaction_mode": InteractionMode.GREETING.value}

    # Check for closings
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

    # Default: proceed to search
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
    """Main agent - processes queries by searching first."""
    messages = state["messages"]

    context_info = ""

    # Add context for scenario selection flow
    if (
        state.get("selected_scenario")
        and state.get("interaction_mode") == InteractionMode.DISAMBIGUATION.value
    ):
        context_info = f"\n\nUser selected scenario: {state['selected_scenario']}"
        context_info += f"\nOriginal query: {state.get('original_query', '')}"
        context_info += "\nCall `get_scenario_answer` with this information."

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_and_route(state: AgentState) -> dict:
    """Validate search results and determine routing."""
    messages = state["messages"]

    # Find the last tool message
    last_tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                last_tool_result = json.loads(msg.content)
                break
            except:
                continue

    if last_tool_result is None:
        return {"should_respond_not_found": False}

    found_answer = last_tool_result.get("found_answer", False)
    disambiguation_needed = last_tool_result.get("disambiguation_needed", False)

    if not found_answer:
        # No relevant information found
        message = last_tool_result.get(
            "message", "I don't have information about that."
        )
        return {
            "should_respond_not_found": True,
            "not_found_message": message,
            "found_relevant_info": False,
        }

    if disambiguation_needed:
        # Multiple scenarios found in content
        scenarios = last_tool_result.get("scenarios", [])
        options = [s.get("title", f"Option {i+1}") for i, s in enumerate(scenarios)]
        question = last_tool_result.get("disambiguation_question", "")

        if not question and scenarios:
            question = "I found information about multiple scenarios:\n\n"
            for i, s in enumerate(scenarios, 1):
                question += f"{i}. **{s.get('title', f'Option {i}')}**\n"
                if s.get("description"):
                    question += f"   {s.get('description')}\n"
            question += "\nWhich one applies to your situation?"

        return {
            "has_multiple_scenarios": True,
            "detected_scenarios": scenarios,
            "disambiguation_question": question,
            "awaiting_scenario_selection": True,
            "current_scenario_options": options,
            "should_respond_not_found": False,
            "search_results": json.dumps(last_tool_result),
        }

    # Single scenario or clear answer - proceed normally
    return {
        "should_respond_not_found": False,
        "found_relevant_info": True,
        "search_results": json.dumps(last_tool_result),
    }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    message = state.get(
        "not_found_message",
        "I don't have information about that in my knowledge base. "
        "Is there something else I can help you with?",
    )
    return {"messages": [AIMessage(content=message)]}


def present_scenarios(state: AgentState) -> dict:
    """Present scenario options when disambiguation is needed."""
    question = state.get(
        "disambiguation_question",
        "I found multiple related scenarios. Could you specify which one you're asking about?",
    )
    return {
        "messages": [AIMessage(content=question)],
        "awaiting_scenario_selection": True,
    }


# ========================================
# ROUTING FUNCTIONS
# ========================================


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
    if mode == InteractionMode.CLARIFICATION.value:
        return "ask_clarification"

    return "agent"


def route_after_validation(
    state: AgentState,
) -> Literal["handle_not_found", "agent", "present_scenarios"]:
    """Route based on search result validation."""
    if state.get("should_respond_not_found", False):
        return "handle_not_found"
    if state.get("has_multiple_scenarios", False) and state.get(
        "awaiting_scenario_selection", False
    ):
        return "present_scenarios"
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
    ]

    @classmethod
    def sanitize(cls, response: str) -> str:
        if not response:
            return response
        sanitized = response
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
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
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("validate_and_route", validate_and_route)
    workflow.add_node("handle_not_found", handle_not_found)
    workflow.add_node("present_scenarios", present_scenarios)

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
    workflow.add_edge("present_scenarios", END)

    # Agent -> tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # Tools -> Validate
    workflow.add_edge("tools", "validate_and_route")

    # Validate -> route appropriately
    workflow.add_conditional_edges(
        "validate_and_route",
        route_after_validation,
        {
            "handle_not_found": "handle_not_found",
            "present_scenarios": "present_scenarios",
            "agent": "agent",
        },
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========================================
# CHATBOT CLASS
# ========================================


class SmartChatbot:
    """Chatbot that searches first, then disambiguates only when necessary."""

    def __init__(self):
        self.agent = create_agent()
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {}
        self.topic_history = []
        self.conversation_history = []

        # Scenario tracking
        self.awaiting_scenario_selection = False
        self.current_scenario_options = []
        self.original_query = None
        self.search_results = None

    def chat(self, message: str) -> str:
        """Process a chat message."""
        config = {"configurable": {"thread_id": self.thread_id}}

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "context": self.context,
            "clarification_needed": False,
            "clarification_reason": "",
            "follow_up_questions": [],
            "pending_clarification": False,
            "original_query": self.original_query or message,
            "clarification_attempts": 0,
            "user_intent": "",
            "detected_topics": [],
            "sentiment": "neutral",
            "interaction_mode": InteractionMode.QUERY.value,
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
            "selected_scenario": None,
            "disambiguation_depth": 0,
            "scenario_context": [],
            "awaiting_scenario_selection": self.awaiting_scenario_selection,
            "filtered_search_results": "",
            "current_scenario_options": self.current_scenario_options,
        }

        try:
            result = self.agent.invoke(initial_state, config=config)

            # Update tracking state
            self.awaiting_scenario_selection = result.get(
                "awaiting_scenario_selection", False
            )
            self.current_scenario_options = result.get("current_scenario_options", [])

            if self.awaiting_scenario_selection:
                self.original_query = result.get("original_query", message)
                self.search_results = result.get("search_results", "")
            else:
                self.original_query = None
                self.search_results = None

            # Get response
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    response = ResponseSanitizer.sanitize(msg.content)

                    self.conversation_history.append(
                        {"role": "user", "content": message}
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": response}
                    )

                    return response

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            return "I encountered an issue. Please try again."

        return "I couldn't generate a response. Please try again."

    def run(self):
        """Run interactive chat."""
        print("\n" + "=" * 60)
        print("  🤖 SMART SUPPORT ASSISTANT")
        print("=" * 60)
        print("\nAsk me anything! I'll search my knowledge base first.")
        print("Type 'quit' to exit, 'topics' for available topics.\n")

        if VECTOR_STORE is not None:
            docs = VECTOR_STORE.similarity_search("", k=10000)
            files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
            print(f"📁 {len(files)} document(s) loaded.\n")
        else:
            print("⚠️ No documents indexed.\n")

        print("-" * 60)

        while True:
            try:
                prompt = "\n👤 You: "
                if self.awaiting_scenario_selection:
                    prompt = "\n👤 You (select an option): "

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
                print(f"\n❌ Error: {e}\n")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    chatbot = SmartChatbot()
    chatbot.run()
