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

    # Clarification settings
    MAX_CLARIFICATION_ATTEMPTS = 2
    MIN_QUERY_LENGTH_FOR_SEARCH = 3


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

    # Requires choice flag
    requires_choice: bool

    # Flag to indicate answer was provided
    answer_provided: bool


# ========================================
# LLM SETUP
# ========================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)


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
# QUERY ANALYZER (ENHANCED)
# ========================================
class QueryAnalyzer:
    """Analyzes user queries with improved new question detection."""

    GREETINGS = [
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

    CLOSINGS = [
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

    # Question indicators that suggest a NEW question, not a selection
    QUESTION_STARTERS = [
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "can",
        "could",
        "would",
        "should",
        "is",
        "are",
        "do",
        "does",
        "tell me",
        "explain",
        "describe",
        "show me",
        "help me",
        "i want",
        "i need",
        "i would like",
        "please",
    ]

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        return query.lower().strip().rstrip("!.,") in cls.GREETINGS

    @classmethod
    def is_closing(cls, query: str) -> bool:
        query_clean = query.lower().strip().rstrip("!.,")
        return any(c in query_clean for c in cls.CLOSINGS)

    @classmethod
    def is_too_short(cls, query: str) -> bool:
        """Check if query is too short to be meaningful."""
        return len(query.strip()) < Config.MIN_QUERY_LENGTH_FOR_SEARCH

    @classmethod
    def is_new_question(cls, query: str) -> bool:
        """
        Detect if user's input is a NEW question rather than a scenario selection.
        This is crucial for breaking out of disambiguation flow.
        """
        query_lower = query.lower().strip()

        # Check for question mark
        if "?" in query:
            return True

        # Check for question starters
        for starter in cls.QUESTION_STARTERS:
            if query_lower.startswith(starter + " ") or query_lower.startswith(
                starter + ","
            ):
                return True

        # Check word count - selections are usually short (1-4 words)
        word_count = len(query_lower.split())
        if word_count > 5:
            return True

        # Check for verbs that indicate new intent
        new_intent_verbs = [
            "apply",
            "create",
            "add",
            "remove",
            "delete",
            "update",
            "change",
            "need",
            "want",
        ]
        if any(verb in query_lower for verb in new_intent_verbs):
            return True

        return False

    @classmethod
    def is_scenario_selection(
        cls, query: str, available_options: List[str]
    ) -> Optional[str]:
        """Check if user's response is selecting a scenario from available options."""
        if not available_options:
            return None

        query_lower = query.lower().strip()

        # First check if this looks like a new question
        if cls.is_new_question(query):
            return None

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

        # Check for exact or close match with options
        for option in available_options:
            option_lower = option.lower()

            # Exact match
            if option_lower == query_lower:
                return option

            # Query is contained in option or vice versa
            if option_lower in query_lower or query_lower in option_lower:
                return option

            # Check for significant word overlap (at least 50% of option words)
            option_words = set(option_lower.split())
            query_words = set(query_lower.split())

            # Filter out common stop words
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "on",
                "in",
                "to",
                "for",
                "of",
                "and",
                "or",
                "my",
                "i",
            }
            option_words = option_words - stop_words
            query_words = query_words - stop_words

            if option_words:
                overlap = option_words.intersection(query_words)
                overlap_ratio = len(overlap) / len(option_words)
                if overlap_ratio >= 0.5:
                    return option

        return None

    @classmethod
    def is_negative_response(cls, query: str) -> bool:
        """Check if user responded with 'no', 'none', 'neither', etc."""
        negative_words = [
            "no",
            "none",
            "neither",
            "nope",
            "not",
            "don't",
            "doesn't",
            "didn't",
            "nothing",
        ]
        query_lower = query.lower().strip()
        return query_lower in negative_words or any(
            query_lower.startswith(nw + " ") for nw in negative_words
        )


# ========================================
# SCENARIO DETECTOR (STRICT)
# ========================================


def is_visa_country_scenario(scenarios: List[dict]) -> bool:
    """
    Heuristic: treat as a visa/country disambiguation if multiple scenarios
    mention 'visa' in the title.
    """
    if not scenarios:
        return False

    titles = [s.get("title", "").lower() for s in scenarios]
    visa_titles = [t for t in titles if "visa" in t]

    # At least two visa-related scenarios → treat as visa/country disambiguation
    return len(visa_titles) >= 2


class ScenarioDetector:
    """
    Detects multiple scenarios ONLY from actual content in search results.
    Uses strict validation to avoid unnecessary disambiguation.
    """

    DETECTION_PROMPT = """Analyze the following search results and determine if they contain MULTIPLE DISTINCT scenarios that the user MUST choose between to get a proper answer.

STRICT RULES:
1. ONLY identify scenarios that are EXPLICITLY mentioned in the search results
2. DO NOT guess or infer scenarios that are not in the content
3. If the content describes ONE process/procedure with multiple steps, there is NO disambiguation needed
4. Multiple scenarios exist ONLY if the content explicitly mentions MUTUALLY EXCLUSIVE cases like:
   - "For Student Visa... For Work Visa... For Job Seeker Visa..."
   - "Option A: ... Option B: ..."
   - Different named categories that cannot all apply to the same user
5. If all the information can be presented together without user selection, set requires_choice to FALSE

USER QUERY: {query}

SEARCH RESULTS CONTENT:
{search_results}

RESPOND IN JSON FORMAT ONLY:
{{
    "has_multiple_scenarios": true/false,
    "requires_choice": true/false,
    "scenarios": [
        {{
            "id": "scenario_1",
            "title": "<brief title - e.g., 'Student Visa', 'Work Visa'>",
            "description": "<one-line description from content>",
            "exact_quote": "<EXACT quote from content that defines this scenario>"
        }}
    ],
    "disambiguation_question": "<clear question asking user to choose - ONLY if requires_choice is true>",
    "reason": "<brief explanation>"
}}

CRITICAL: 
- Set requires_choice to TRUE only if the user CANNOT receive a complete answer without first selecting which scenario applies to them.
- Visa type questions (Student/Work/Tourist) typically REQUIRE choice.
- Step-by-step procedures within a single category do NOT require choice.
"""

    @classmethod
    def detect(
        cls, query: str, search_results: List[dict], llm_instance: ChatOpenAI
    ) -> dict:
        """Detect if search results contain multiple scenarios requiring user choice."""
        formatted_results = "\n\n---\n\n".join(
            [doc.get("content", "") for doc in search_results[:5]]
        )

        prompt = cls.DETECTION_PROMPT.format(
            query=query, search_results=formatted_results
        )

        try:
            response = llm_instance.invoke(
                [
                    SystemMessage(
                        content="You are a strict scenario detector. Only identify scenarios EXPLICITLY present in the content that REQUIRE user choice. Return ONLY valid JSON."
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            content = response.content
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Validate scenarios have exact quotes
            scenarios = result.get("scenarios", [])
            valid_scenarios = []

            for s in scenarios:
                quote = s.get("exact_quote", "")
                title = s.get("title", "")
                # Keep scenario if it has a title and some quote
                if title and quote:
                    valid_scenarios.append(s)

            if len(valid_scenarios) < Config.MIN_SCENARIOS_FOR_DISAMBIGUATION:
                result["has_multiple_scenarios"] = False
                result["requires_choice"] = False
                result["disambiguation_needed"] = False
            else:
                result["disambiguation_needed"] = result.get("requires_choice", False)

            result["scenarios"] = valid_scenarios[: Config.MAX_SCENARIOS_TO_SHOW]
            return result

        except (json.JSONDecodeError, Exception) as e:
            return {
                "has_multiple_scenarios": False,
                "requires_choice": False,
                "scenarios": [],
                "disambiguation_needed": False,
                "disambiguation_question": "",
                "reason": f"Detection error: {str(e)}",
            }


def is_visa_country_scenario(scenarios: List[dict]) -> bool:
    """
    Heuristic: treat as a visa/country disambiguation if multiple scenarios
    mention 'visa' in the title.
    """
    if not scenarios:
        return False

    titles = [s.get("title", "").lower() for s in scenarios]
    visa_titles = [t for t in titles if "visa" in t]
    return len(visa_titles) >= 2


def build_dynamic_visa_disambiguation_question(
    state: AgentState, scenarios: List[dict]
) -> str:
    """
    Build ONE clarification question for visa scenarios, using the recent
    conversation + scenario titles.

    It should:
    - Use the conversation to infer what the user ALREADY specified
      (country and/or visa type).
    - Ask ONLY for what is still missing.
    - NOT list internal options or country names.
    - Return 'NO_QUESTION' if nothing else is needed.
    """

    # Get recent conversation turns
    messages = state.get("messages", [])
    conv_lines = []
    # Take last few turns to keep prompt small
    for m in messages[-8:]:
        if isinstance(m, HumanMessage):
            conv_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            conv_lines.append(f"Assistant: {m.content}")
    conversation = "\n".join(conv_lines)

    scenario_titles = [s.get("title", "") for s in scenarios]

    system_msg = SystemMessage(
        content=(
            "You write ONE short clarifying question for a visa support assistant.\n\n"
            "You will be given the recent conversation and internal visa scenario titles.\n\n"
            "Rules:\n"
            "1. Use the conversation to infer what the user has ALREADY specified "
            "(e.g., visa type such as student/work, and/or country).\n"
            "2. Ask ONLY for the information that is still missing to uniquely determine "
            "the visa scenario (usually country and/or visa type).\n"
            "3. Do NOT list or mention any specific scenario titles or country names from the list.\n"
            "4. Do NOT repeat information the user already gave.\n"
            "5. If the conversation already identifies a single clear visa scenario and "
            "no clarification is needed, respond with exactly: NO_QUESTION\n"
            "6. Otherwise, respond with exactly ONE clarifying question sentence, and nothing else."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"Recent conversation:\n{conversation}\n\n"
            f"Internal scenario titles:\n{scenario_titles}\n\n"
            "Decide whether clarification is needed, and if so, ask one question."
        )
    )

    resp = llm.invoke([system_msg, human_msg])
    return resp.content.strip()


# ========================================
# TOOLS
# ========================================
@tool
def search_and_analyze(query: str) -> str:
    """
    Search the document database and analyze if multiple scenarios exist.
    This is the primary tool - always call this first.

    Returns:
        JSON with search results, scenario analysis, and requires_choice flag
    """
    if VECTOR_STORE is None:
        return json.dumps(
            {
                "found_answer": False,
                "quality": SearchQuality.NOT_FOUND.value,
                "confidence": 0.0,
                "documents": [],
                "has_multiple_scenarios": False,
                "requires_choice": False,
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
                "requires_choice": False,
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
                "requires_choice": False,
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
                "requires_choice": False,
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
                "requires_choice": False,
                "disambiguation_needed": False,
                "message": "No relevant information found.",
            }
        )

        # Check for multiple scenarios in the ACTUAL content

    # Check for multiple scenarios in the ACTUAL content
    scenario_result = ScenarioDetector.detect(query, documents, llm)

    scenarios = scenario_result.get("scenarios", [])
    has_multiple = scenario_result.get("has_multiple_scenarios", False)
    requires_choice = scenario_result.get("requires_choice", False)
    disambiguation_needed = requires_choice

    # Default: use whatever the detector produced
    disambiguation_question = ""
    if disambiguation_needed:
        disambiguation_question = scenario_result.get("disambiguation_question", "")

        # --- VISA / COUNTRY SPECIAL HANDLING ---
        # For visa-related scenarios, ask a generic question instead of listing
        # "UK Student Visa, USA Student Visa, ..." etc.
        if "visa" in query.lower() or is_visa_country_scenario(scenarios):
            disambiguation_question = (
                "Please specify the country and visa type you are asking about."
            )
        # ---------------------------------------

    return json.dumps(
        {
            "found_answer": True,
            "should_respond": True,
            "quality": analysis["quality"],
            "confidence": float(analysis["confidence"]),
            "documents": documents,
            "count": len(documents),
            "has_multiple_scenarios": has_multiple,
            "requires_choice": requires_choice,
            "disambiguation_needed": disambiguation_needed,
            "scenarios": scenarios,
            "disambiguation_question": disambiguation_question,
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
    # Remove common words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "on",
        "in",
        "to",
        "for",
        "of",
        "and",
        "or",
        "stuck",
        "my",
        "i",
    }
    scenario_keywords = scenario_keywords - stop_words

    filtered_docs = []

    for doc, score in results:
        content_lower = doc.page_content.lower()
        keyword_matches = sum(
            1 for kw in scenario_keywords if kw in content_lower and len(kw) > 2
        )

        if keyword_matches > 0 or float(score) < Config.GOOD_SCORE:
            filtered_docs.append(
                {
                    "content": doc.page_content,
                    "relevance": "high" if keyword_matches >= 1 else "medium",
                    "score": float(score),
                    "keyword_matches": keyword_matches,
                }
            )

    filtered_docs.sort(key=lambda x: (-x.get("keyword_matches", 0), x["score"]))

    return json.dumps(
        {
            "found_answer": len(filtered_docs) > 0,
            "selected_scenario": selected_scenario,
            "documents": filtered_docs[:3],
            "should_respond": len(filtered_docs) > 0,
            "requires_choice": False,  # Already made choice
            "disambiguation_needed": False,
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
            "Applications": ["application", "submit", "process", "status"],
            "Credits": ["credit", "balance", "payment", "transaction"],
            "Visa": ["visa", "immigration", "passport", "embassy"],
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

llm_with_tools = llm.bind_tools(tools)


# ========================================
# SYSTEM PROMPT
# ========================================
SYSTEM_PROMPT = """You are a highly specialized document-based support assistant. Your SOLE purpose is to provide information by STRICTLY extracting content from search results.

CRITICAL WORKFLOW:

STEP 1: ALWAYS SEARCH FIRST
- For ANY user question, IMMEDIATELY call search_and_analyze tool
- DO NOT ask clarifying questions BEFORE searching
- DO NOT speculate about what scenarios might exist

STEP 2: INTERPRET TOOL RESULTS
After receiving results from search_and_analyze:

A) If found_answer: false
   → Respond: "I don't have information about that in my knowledge base."
   → Do not elaborate or offer alternatives

B) If found_answer: true AND requires_choice: false
   → Generate answer by PRECISELY extracting information from the documents
   → Use the exact wording from documents as much as possible
   → If information is missing, state: "The available information does not specify [X]."
   → Format clearly with numbered steps if applicable

C) If found_answer: true AND requires_choice: true
   → Present ONLY the scenarios found in the tool output
   → Use the exact titles and descriptions from the scenarios array
   → Ask the disambiguation_question provided
   → Wait for user selection - DO NOT provide any answer yet

STEP 3: AFTER USER SELECTS SCENARIO
- Call get_scenario_answer with the selection
- Extract answer ONLY from the returned documents
- Present the answer clearly with steps if applicable
- DO NOT ask further questions unless the tool explicitly requires it

ABSOLUTE RULES:
1. NEVER ask for clarification BEFORE searching
2. NEVER speculate about scenarios - only present what tools explicitly report
3. ONLY use information EXPLICITLY stated in documents
4. DO NOT add examples, elaboration, or "tips" not in the source
5. DO NOT mention "documents", "search results", "knowledge base" in your answer
6. If information is missing, explicitly acknowledge it
7. Maintain neutral, factual tone
8. After providing an answer, the interaction is complete - do not ask follow-up questions
"""


# ========================================
# GRAPH NODES
# ========================================
def analyze_input(state: AgentState) -> dict:
    """Analyze user input and determine interaction mode."""
    messages = state["messages"]

    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"interaction_mode": InteractionMode.QUERY.value}

    # Check for greetings first
    if QueryAnalyzer.is_greeting(user_message):
        return {
            "interaction_mode": InteractionMode.GREETING.value,
            # Reset any pending disambiguation
            "awaiting_scenario_selection": False,
            "current_scenario_options": [],
        }

    # Check for closings
    if QueryAnalyzer.is_closing(user_message):
        return {
            "interaction_mode": InteractionMode.CLOSING.value,
            "awaiting_scenario_selection": False,
            "current_scenario_options": [],
        }

    # ============================================
    # KEY FIX: Check if this is a NEW question
    # even when we're awaiting scenario selection
    # ============================================
    if QueryAnalyzer.is_new_question(user_message):
        # This is a new question - reset disambiguation state and process as new query
        return {
            "interaction_mode": InteractionMode.QUERY.value,
            "original_query": user_message,
            # Reset disambiguation state
            "awaiting_scenario_selection": False,
            "current_scenario_options": [],
            "selected_scenario": None,
            "search_results": "",
        }

    # Check if we're awaiting scenario selection
    if state.get("awaiting_scenario_selection", False):
        current_options = state.get("current_scenario_options", [])

        # Check for negative response
        if QueryAnalyzer.is_negative_response(user_message):
            # User said "no" - treat as needing more context
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": user_message,
                "awaiting_scenario_selection": False,
            }

        # Check if user selected one of the options
        selected = QueryAnalyzer.is_scenario_selection(user_message, current_options)

        if selected:
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": selected,
                "awaiting_scenario_selection": False,
            }
        else:
            # User's response didn't match options - use as additional context
            return {
                "interaction_mode": InteractionMode.DISAMBIGUATION.value,
                "selected_scenario": user_message,
                "awaiting_scenario_selection": False,
            }

    # Check for too short
    if QueryAnalyzer.is_too_short(user_message):
        return {
            "interaction_mode": InteractionMode.CLARIFICATION.value,
            "clarification_needed": True,
            "follow_up_questions": [
                "Could you please provide more details about what you're looking for?"
            ],
        }

    # Default: proceed to search as new query
    return {
        "interaction_mode": InteractionMode.QUERY.value,
        "original_query": user_message,
        "awaiting_scenario_selection": False,
        "current_scenario_options": [],
    }


def handle_greeting(state: AgentState) -> dict:
    """Handle greeting messages."""
    import random

    greetings = [
        "Hello! I'm here to help you find information. What would you like to know?",
        "Hi there! How can I assist you today?",
        "Hey! What can I help you with?",
    ]
    return {
        "messages": [AIMessage(content=random.choice(greetings))],
        "awaiting_scenario_selection": False,
        "answer_provided": True,
    }


def handle_closing(state: AgentState) -> dict:
    """Handle closing messages."""
    import random

    closings = [
        "Goodbye! Feel free to return if you have more questions.",
        "Happy to help! Take care!",
        "Glad I could assist. Have a great day!",
    ]
    return {
        "messages": [AIMessage(content=random.choice(closings))],
        "awaiting_scenario_selection": False,
        "answer_provided": True,
    }


def ask_clarification(state: AgentState) -> dict:
    """Ask for clarification for very short/unclear queries."""
    follow_up = state.get("follow_up_questions", ["What would you like to know?"])
    return {
        "messages": [AIMessage(content=follow_up[0])],
        "pending_clarification": True,
        "awaiting_scenario_selection": False,
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
        search_results = state.get("search_results", "")
        if search_results:
            context_info += f"\nPrevious search results: {search_results}"
        context_info += "\nCall `get_scenario_answer` with this information to get the specific answer. Then provide the answer directly without asking more questions."

    system = SystemMessage(content=SYSTEM_PROMPT + context_info)
    response = llm_with_tools.invoke([system] + list(messages))

    return {"messages": [response], "has_searched": True}


def validate_and_route(state: AgentState) -> dict:
    """Validate search results and determine routing based on requires_choice flag."""
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
    requires_choice = last_tool_result.get("requires_choice", False)

    if not found_answer:
        message = last_tool_result.get(
            "message", "I don't have information about that."
        )
        return {
            "should_respond_not_found": True,
            "not_found_message": message,
            "found_relevant_info": False,
            "awaiting_scenario_selection": False,
        }

    # Handle scenario disambiguation
    if requires_choice:
        scenarios = last_tool_result.get("scenarios", [])
        options = [s.get("title", f"Option {i+1}") for i, s in enumerate(scenarios)]
        question = last_tool_result.get("disambiguation_question", "")

        # ----- DYNAMIC VISA QUESTION LOGIC -----
        if is_visa_country_scenario(scenarios):
            try:
                dyn_q = build_dynamic_visa_disambiguation_question(state, scenarios)
                if dyn_q == "NO_QUESTION":
                    # Conversation already pins down a single visa scenario.
                    # Treat this as if no disambiguation is required.
                    return {
                        "should_respond_not_found": False,
                        "found_relevant_info": True,
                        "search_results": json.dumps(last_tool_result),
                        "requires_choice": False,
                        "awaiting_scenario_selection": False,
                        # Will still go through agent to generate the answer
                        "answer_provided": True,
                    }
                else:
                    # Use the dynamic question; for visa we *don't* want to list options.
                    question = dyn_q
                    options = []
            except Exception:
                # If anything goes wrong, fall back to normal behaviour below
                pass
        # ----- END DYNAMIC VISA LOGIC -----

        # Fallback: if still no question, build the standard options list
        if not question and scenarios:
            question = "I found information about multiple options:\n\n"
            for i, s in enumerate(scenarios, 1):
                question += f"{i}. **{s.get('title', f'Option {i}')}**"
                if s.get("description"):
                    question += f": {s.get('description')}"
                question += "\n"
            question += "\nWhich one applies to your situation?"

        return {
            "has_multiple_scenarios": True,
            "requires_choice": True,
            "detected_scenarios": scenarios,
            "disambiguation_question": question,
            "awaiting_scenario_selection": True,
            "current_scenario_options": options,
            "should_respond_not_found": False,
            "search_results": json.dumps(last_tool_result),
            "answer_provided": False,
        }


def handle_not_found(state: AgentState) -> dict:
    """Handle case when no relevant information was found."""
    message = state.get(
        "not_found_message",
        "I don't have information about that in my knowledge base.",
    )
    return {
        "messages": [AIMessage(content=message)],
        "awaiting_scenario_selection": False,
        "answer_provided": True,
    }


def present_scenarios(state: AgentState) -> dict:
    """Present scenario options when disambiguation is needed."""
    question = state.get(
        "disambiguation_question",
        "I found multiple related scenarios. Could you specify which one applies to you?",
    )
    return {
        "messages": [AIMessage(content=question)],
        "awaiting_scenario_selection": True,
        "answer_provided": False,  # Still waiting for selection
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

    # Check if disambiguation is needed
    if state.get("requires_choice", False) and state.get(
        "awaiting_scenario_selection", False
    ):
        return "present_scenarios"

    return "agent"


# ========================================
# RESPONSE SANITIZER
# ========================================
class ResponseSanitizer:
    """Sanitize responses to remove file references."""

    FILE_PATTERNS = [
        r"\b[\w\-]+\.(pdf|docx?|txt|xlsx?|pptx?|csv|json|xml|html?|md)\b",
        r"\(source:\s*[^)]+\)",
        r"source:\s*[\w\-\.]+",
        r"(?i)according to the [\w\s]+ document",
        r"(?i)based on the [\w\s]+ document",
        r"(?i)the document (states|mentions|indicates)",
        r"(?i)from the search results",
        r"(?i)in my knowledge base",
    ]

    @classmethod
    def sanitize(cls, response: str) -> str:
        if not response:
            return response

        sanitized = response

        # Remove file references
        for pattern in cls.FILE_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Clean up extra whitespace
        sanitized = re.sub(r"\s{2,}", " ", sanitized)
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

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
# CHATBOT CLASS (KEY FIXES HERE)
# ========================================
class SmartChatbot:
    """Enhanced chatbot with proper state management."""

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
        self.selected_scenario = None

    def _reset_disambiguation_state(self):
        """Reset disambiguation-related state."""
        self.awaiting_scenario_selection = False
        self.current_scenario_options = []
        self.original_query = None
        self.search_results = None
        self.selected_scenario = None

    def chat(self, message: str) -> str:
        """Process a chat message."""
        config = {"configurable": {"thread_id": self.thread_id}}

        # ============================================
        # KEY FIX: Detect new questions and reset state
        # ============================================
        if self.awaiting_scenario_selection and QueryAnalyzer.is_new_question(message):
            self._reset_disambiguation_state()

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
            "selected_scenario": self.selected_scenario,
            "disambiguation_depth": 0,
            "scenario_context": [],
            "awaiting_scenario_selection": self.awaiting_scenario_selection,
            "filtered_search_results": "",
            "current_scenario_options": self.current_scenario_options,
            "requires_choice": False,
            "answer_provided": False,
        }

        try:
            result = self.agent.invoke(initial_state, config=config)

            # ============================================
            # KEY FIX: Proper state management after response
            # ============================================

            # Check if we're now awaiting scenario selection
            new_awaiting = result.get("awaiting_scenario_selection", False)
            answer_provided = result.get("answer_provided", False)

            if new_awaiting:
                # Entering disambiguation mode
                self.awaiting_scenario_selection = True
                self.current_scenario_options = result.get(
                    "current_scenario_options", []
                )
                self.original_query = result.get("original_query", message)
                self.search_results = result.get("search_results", "")
            elif answer_provided or not new_awaiting:
                # Answer was provided OR we're not waiting for selection
                # Reset the disambiguation state
                self._reset_disambiguation_state()

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
            # Reset state on error
            self._reset_disambiguation_state()
            return "I encountered an issue processing your request. Please try again."

        return "I couldn't generate a response. Please try again."

    def reset_session(self):
        """Reset the conversation session."""
        self.thread_id = f"session_{datetime.now().timestamp()}"
        self.context = {}
        self.topic_history = []
        self.conversation_history = []
        self._reset_disambiguation_state()

    def run(self):
        """Run interactive chat."""
        print("\n" + "=" * 60)
        print("  🤖 SMART SUPPORT ASSISTANT")
        print("=" * 60)
        print("\nAsk me anything! I'll search my knowledge base first.")
        print(
            "Commands: 'quit' to exit, 'topics' for available topics, 'reset' to start fresh.\n"
        )

        if VECTOR_STORE is not None:
            try:
                docs = VECTOR_STORE.similarity_search("", k=10000)
                files = set(doc.metadata.get("filename", "Unknown") for doc in docs)
                print(f"📁 {len(files)} document(s) loaded.\n")
            except:
                print("📁 Documents loaded.\n")
        else:
            print("⚠️ No documents indexed.\n")

        print("-" * 60)

        while True:
            try:
                # Dynamic prompt based on state
                if self.awaiting_scenario_selection:
                    prompt = "\n👤 You (select an option): "
                else:
                    prompt = "\n👤 You: "

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!\n")
                    break

                if user_input.lower() == "reset":
                    self.reset_session()
                    print("\n🔄 Session reset. Start a new conversation!\n")
                    continue

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
                self._reset_disambiguation_state()


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    chatbot = SmartChatbot()
    chatbot.run()
