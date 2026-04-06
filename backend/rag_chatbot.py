import re   # Used for keyword (pattern) matching
import os
import streamlit as st

# 🔥 Load secrets into environment
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
# ------------------ QUERY EXPANSION ------------------
def expand_query(question):
    # Prompt to LLM to generate similar questions
    prompt = f"Rephrase or expand the following question to cover more ways it could be asked about visa requirements. Provide 3 alternative phrasings, separated by newlines.\nQuestion: {question}"
    
    # Send prompt to configured LLM provider. If this call fails, continue with
    # original query.
    try:
        response = _invoke_llm(prompt, retries=1)
    except Exception:
        return [question.strip()]
    # test change
    expansions = [question.strip()]  # Start with original question
    
    # Add generated variations line by line
    for line in response.content.splitlines():
        if line.strip():
            expansions.append(line.strip())
    
    # Remove duplicates and return
    return list(dict.fromkeys(expansions))


# ------------------ KEYWORD SEARCH ------------------
def keyword_search(question, collection, n_results=10):
    
    # Get documents from database with limit to avoid SQL variable overflow
    all_docs = collection.get(limit=1000, include=["documents", "metadatas"])
    docs = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])
    
    results = []
    
    # Convert question into regex pattern (acts as keyword search)
    pattern = re.compile(re.escape(question), re.IGNORECASE)
    
    # Loop through each document
    for idx, doc in enumerate(docs):
        if pattern.search(doc):   # Check if keyword exists in document
            
            # Get metadata (source, page, etc.)
            meta = metadatas[idx] if idx < len(metadatas) else {}
            
            # Store result (distance = 0 for keyword match)
            results.append((doc, meta, 0.0))
    
    return results[:n_results]


# ------------------ IMPORTS & SETUP ------------------
import os
import json
import time
import importlib
import chromadb
from dotenv import load_dotenv

try:
    _langchain_groq = importlib.import_module("langchain_groq")
    ChatGroq = getattr(_langchain_groq, "ChatGroq", None)
except Exception:
    ChatGroq = None
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
load_dotenv()   # Load API keys

# Enable tracing (for debugging / monitoring)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import os
import chromadb
import subprocess

# 🔥 Build DB
print("Building Chroma DB from PDFs...")
subprocess.run(["python", "backend/store_dataset.py"])

# ✅ Client
client = chromadb.Client()

# Load collections for different countries
us_collection = client.get_or_create_collection("us_visa_collection")
uk_collection = client.get_or_create_collection("uk_visa_collection")
australia_collection = client.get_or_create_collection("australia_collection")


# ------------------ LLM + MEMORY ------------------
# Provider can be switched with: LLM_PROVIDER=groq
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

# Stores previous conversation (chat history)
memory = ConversationBufferMemory(return_messages=True)


def _is_quota_error(error_text):
    text = error_text.lower()
    return (
        "resource_exhausted" in text
        or "quota exceeded" in text
        or "insufficient_quota" in text
        or "rate limit" in text
        or "429" in text
    )


def _ordered_unique(items):
    seen = set()
    result = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _retry_wait_seconds(error_text, default_seconds=10):
    # Provider errors often contain: "Please retry in 38.11s"
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text.lower())
    if not match:
        return default_seconds

    try:
        return max(1, int(float(match.group(1))))
    except Exception:
        return default_seconds


def _invoke_llm(prompt, retries=0):
    """Invoke LLM with Groq model failover."""
    last_exc = None

    if LLM_PROVIDER != "groq":
        raise RuntimeError("LLM_PROVIDER must be set to groq.")

    if ChatGroq is None:
        raise RuntimeError(
            "LLM_PROVIDER=groq requires langchain-groq. "
            "Install with: pip install langchain-groq groq"
        )

    for model_name in _ordered_unique(GROQ_MODEL_CANDIDATES):
        try:
            llm = ChatGroq(model=model_name, temperature=0)
            return llm.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            last_exc = exc

    # Return quickly on repeated failure so UI can show offline fallback.
    raise last_exc


# ------------------ CONFIDENCE SCORE ------------------
def calculate_confidence(distances):
    if not distances:
        return 0.0

    # 🔥 Use only top 5 best matches
    top_distances = distances[:5]

    scores = []
    for d in top_distances:
        d = float(d)

        # 🎯 Strong scoring logic
        if d < 0.2:
            score = 0.95
        elif d < 0.4:
            score = 0.85
        elif d < 0.6:
            score = 0.70
        else:
            score = 0.50

        scores.append(score)

    # 🔥 Weighted importance (top result matters most)
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights[:len(scores)])

    confidence = (weighted_sum / total_weight) * 100

    return round(confidence, 1)


# ------------------ COUNTRY DETECTION ------------------
def detect_country(question):
    q = str(question).lower()
    q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
    q_norm = re.sub(r"\s+", " ", q_norm).strip()
    q_compact = re.sub(r"[^a-z0-9]", "", q)

    # Detect based on keywords in question
    if "australia" in q_norm or "australian" in q_norm:
        return "australia"

    if (
        re.search(r"\buk\b", q_norm)
        or re.search(r"\bu\s*k\b", q_norm)
        or "unitedkingdom" in q_compact
        or "greatbritain" in q_compact
        or "britain" in q_norm
        or "england" in q_norm
        or "british" in q_norm
    ):
        return "uk"

    if (
        re.search(r"\busa\b", q_norm)
        or re.search(r"\bu\s*s\b", q_norm)
        or "unitedstates" in q_compact
        or "america" in q_norm
        or "usvisa" in q_compact
    ):
        return "us"
    
    return "us"   # Default country


# ------------------ RETRIEVAL (CORE FUNCTION) ------------------
def retrieve_context(question, country):
    
    # Select correct collection
    if country == "australia":
        collection = australia_collection
    elif country == "uk":
        collection = uk_collection
    else:
        collection = us_collection

    # Step 1: Expand query
    expanded_queries = expand_query(question)

    # Step 2: Vector search (semantic search)
    all_vector_results = []
    for q in expanded_queries:
        results = collection.query(
            query_texts=[q],
            n_results=20,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Store results
        for doc, meta, dist in zip(docs, metadatas, distances):
            all_vector_results.append((doc, meta, dist))

    # Step 3: Keyword search
    keyword_results = keyword_search(question, collection, n_results=10)

    # Step 4: Merge both results
    all_results = all_vector_results + keyword_results
    
    seen = set()
    merged = []
    
    # Remove duplicates and sort by best match (low distance)
    for doc, meta, dist in sorted(all_results, key=lambda x: float(x[2])):
        key = (doc, meta.get("source", ""), meta.get("page", ""))
        
        if key not in seen:
            merged.append((doc, meta, dist))
            seen.add(key)
        
        if len(merged) >= 15:
            break

    # Step 5: Build context for LLM
    context_blocks = []
    distances = []
    
    TOP_K = 5
    for doc, meta, dist in merged[:TOP_K]:
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        
        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc}")
        distances.append(float(dist))

    context = "\n\n".join(context_blocks)

    # Limit context size (avoid overload)
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length]

    # Step 6: Calculate scores
    confidence = calculate_confidence(distances)

    # Boost confidence when keyword match exists.
    if len(keyword_results) > 0:
        confidence += 10

    # Cap max confidence.
    confidence = min(confidence, 95)

    # Floor non-zero confidence to avoid overly low display.
    if confidence > 0:
        confidence = max(confidence, 60)

    return context, confidence


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start:end])
    except Exception:
        return {}


def _as_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _decision_label(eligibility_score):
    if eligibility_score >= 70:
        return "Likely"
    if eligibility_score >= 40:
        return "Maybe"
    return "Unlikely"


def _country_label(country):
    country_text = str(country or "").strip().lower()
    if country_text in {"australia", "au"}:
        return "Australia"
    if country_text in {"uk", "united kingdom", "great britain", "britain"}:
        return "UK"
    if country_text in {"us", "usa", "united states", "america"}:
        return "USA"
    return "Unknown"


def _normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def _extract_salary_amount(text):
    normalized = _normalize_text(text).replace(",", "")

    salary_window = re.search(
        r"(?:salary|pay|annual|annum|per year|pa)\D{0,20}(?:£\s*)?(\d+(?:\.\d+)?)\s*(k)?",
        normalized,
    )
    if salary_window:
        amount = float(salary_window.group(1))
        if salary_window.group(2):
            amount *= 1000.0
        return amount

    direct_amount = re.search(r"(?:£\s*)?(\d{2,6}(?:\.\d+)?)\s*(k)?", normalized)
    if direct_amount:
        amount = float(direct_amount.group(1))
        if direct_amount.group(2):
            amount *= 1000.0
        return amount

    return None


def _criterion_keywords(criterion):
    text = _normalize_text(criterion)
    keywords = []

    if "passport" in text:
        keywords.extend(["passport"])
    if any(term in text for term in ["sponsor", "sponsorship", "employer"]):
        keywords.extend(["sponsor", "sponsorship", "employer sponsor", "approved employer", "licensed sponsor", "sponsor licence"])
    if any(term in text for term in ["occupation", "job", "role", "position", "occupation"]):
        keywords.extend(["job", "role", "occupation", "position"])
    if "salary" in text or "wage" in text or "pay" in text or "income" in text:
        keywords.extend(["salary", "pay", "annual", "annum", "per year", "wage", "income"])
    if any(term in text for term in ["degree", "qualification", "education", "diploma", "certificate"]):
        keywords.extend(["degree", "qualification", "education", "diploma", "certificate", "m.tech", "b.tech", "masters", "bachelor"])
    if any(term in text for term in ["coS", "certificate of sponsorship", "certificate"]):
        keywords.extend(["certificate of sponsorship", "cos"])
    if any(term in text for term in ["fund", "money", "maintenance", "bank"]):
        keywords.extend(["fund", "money", "maintenance", "bank"])

    return list(dict.fromkeys(keyword for keyword in keywords if keyword))


def _build_explicit_fact_checklist(question, context, data):
    question_text = _normalize_text(question)
    question_salary = _extract_salary_amount(question_text)

    checklist = data.get("eligibility_checklist", [])
    if not isinstance(checklist, list) or not checklist:
        return data, False, 0.0

    updated = []
    overrides = 0
    pass_count = 0

    for item in checklist:
        if not isinstance(item, dict):
            continue

        criterion = str(item.get("criterion", "")).strip()
        status = str(item.get("status", "")).strip().upper()
        note = str(item.get("note", "")).strip()
        criterion_text = _normalize_text(criterion)
        keywords = _criterion_keywords(criterion)

        matched = any(keyword in question_text for keyword in keywords)
        salary_match = False

        if "salary" in criterion_text:
            threshold_match = re.search(r"(\d{2,6}(?:\.\d+)?)", criterion_text.replace(",", ""))
            criterion_threshold = None
            if threshold_match:
                try:
                    criterion_threshold = float(threshold_match.group(1))
                except Exception:
                    criterion_threshold = None
            if question_salary is not None:
                if criterion_threshold is not None:
                    salary_match = question_salary >= criterion_threshold
                else:
                    salary_match = True

        if matched or salary_match:
            if status != "PASS":
                overrides += 1
            status = "PASS"
            if not note:
                note = "Explicitly mentioned in the question."
            elif "explicit" not in note.lower():
                note = f"{note} Explicitly mentioned in the question."

        updated.append({
            "criterion": criterion,
            "status": status if status in {"PASS", "NEED"} else "NEED",
            "note": note or ("Explicitly mentioned in the question." if status == "PASS" else "Not clearly mentioned in the question."),
        })

        if status == "PASS":
            pass_count += 1

    total_items = len(updated)
    passed_ratio = (pass_count / total_items) if total_items else 0.0
    heuristic_score = round(passed_ratio * 100.0, 1)

    data["eligibility_checklist"] = updated
    data["what_user_has"] = [item["criterion"] for item in updated if item["status"] == "PASS"]
    data["what_is_needed"] = [item["criterion"] for item in updated if item["status"] == "NEED"]
    data["overall_eligibility_percent"] = max(
        heuristic_score,
        float(data.get("overall_eligibility_percent", 0) or 0),
    )
    data["eligibility_rationale"] = (
        "Checklist updated from explicit facts in the question so mentioned criteria are marked PASS."
    )

    confidence = min(95.0, max(45.0, 50.0 + (pass_count * 8.0) + (overrides * 2.0) - (total_items * 3.0)))
    return data, True, round(confidence, 1)


def _offline_fallback_answer(question, context, country="Unknown", visa_type="Other"):
    q = question.lower()
    country_label = _country_label(country)

    has_job_offer = any(k in q for k in ["job offer", "offering", "offer"])
    has_sponsorship = any(k in q for k in ["sponsorship", "sponsor"])
    has_passport = "passport" in q
    has_degree = any(k in q for k in ["b.tech", "btech", "m.tech", "mtech", "bachelor", "master", "degree"])

    what_you_have = []
    if has_job_offer:
        what_you_have.append("Job offer mentioned")
    if has_sponsorship:
        what_you_have.append("Employer sponsorship mentioned")
    if has_passport:
        what_you_have.append("Valid passport mentioned")
    if has_degree:
        what_you_have.append("Relevant degree mentioned")

    if not what_you_have:
        what_you_have.append("Basic profile details shared")

    needed = [
        "Exact job title, duties, and salary details",
        "Employer petition status and supporting documents",
        "Education and experience proof matching visa rules",
        "Any previous visa or immigration history"
    ]

    score = 20.0
    if has_job_offer:
        score += 20.0
    if has_sponsorship:
        score += 20.0
    if has_passport:
        score += 10.0
    if has_degree:
        score += 10.0
    score = max(0.0, min(100.0, score))

    decision = _decision_label(score)

    if country_label == "Australia":
        suggestions = [
            "- Skilled Employer Sponsored visa (estimated fit): Requires approved sponsor and skilled role",
            "- Skilled Independent visa (conditional): Depends on points and skills assessment"
        ]
    elif country_label == "UK":
        suggestions = [
            "- Skilled Worker visa (estimated fit): Requires sponsor licence and eligible occupation",
            "- Global Talent visa (conditional): Depends on exceptional profile evidence"
        ]
    else:
        suggestions = [
            "- H-1B (estimated fit): Common path for software roles with sponsor",
            "- L-1 (conditional): If transfer from an overseas branch applies"
        ]

    # Keep this plain text to match the chatbot's current output style.
    answer = (
        f"DECISION\n{decision}\n\n"
        "SUMMARY\n"
            "Live LLM API is currently unavailable or quota-limited, so this is a temporary offline estimate from your question.\n\n"
        "MAIN ELIGIBILITY CRITERIA\n"
        f"- Valid employer-backed role for {country_label}\n"
        "- Qualification match to the role\n"
        "- Employer petition and compliance documents\n"
        "- Applicant background and document consistency\n\n"
        "WHAT YOU HAVE\n"
        + "\n".join(f"- {item}" for item in what_you_have)
        + "\n\nWHAT IS NEEDED\n"
        + "\n".join(f"- {item}" for item in needed)
        + "\n\nELIGIBILITY CHECKLIST\n"
        + "\n".join([
            f"- [{'PASS' if has_job_offer else 'NEED'}] Job offer",
            f"- [{'PASS' if has_sponsorship else 'NEED'}] Employer sponsorship",
            f"- [{'PASS' if has_passport else 'NEED'}] Passport",
            f"- [{'PASS' if has_degree else 'NEED'}] Degree/qualification evidence",
            "- [NEED] Employer petition filing confirmation",
        ])
        + "\n\nVISA SUGGESTIONS\n"
        + "\n".join(suggestions)
        + "\n\nNEXT STEPS\n"
        "- Confirm employer petition status\n"
        "- Prepare degree and experience proofs\n"
        "- Re-run after API quota reset for policy-accurate result\n\n"
        "IMPORTANT NOTES\n"
        "- This is a fallback estimate, not a legal decision\n"
        "- Final eligibility depends on official case-specific review"
    )

    return answer, round(score, 1)


def _format_structured_answer(data):
    summary = str(data.get("summary", "")).strip()
    criteria = _as_list(data.get("main_eligibility_criteria", []))
    user_has = _as_list(data.get("what_user_has", []))
    needed = _as_list(data.get("what_is_needed", []))
    suggestions = data.get("visa_suggestions", [])
    checklist = data.get("eligibility_checklist", [])
    notes = _as_list(data.get("important_notes", []))
    next_steps = _as_list(data.get("next_steps", []))
    rationale = str(data.get("eligibility_rationale", "")).strip()

    sections = []
    if summary:
        sections.append("SUMMARY\n" + summary)

    if criteria:
        sections.append(
            "MAIN ELIGIBILITY CRITERIA\n" + "\n".join(f"- {item}" for item in criteria)
        )

    if user_has:
        sections.append("WHAT YOU HAVE\n" + "\n".join(f"- {item}" for item in user_has))

    if needed:
        sections.append("WHAT IS NEEDED\n" + "\n".join(f"- {item}" for item in needed))

    checklist_lines = []
    if isinstance(checklist, list):
        for item in checklist:
            if not isinstance(item, dict):
                continue
            criterion = str(item.get("criterion", "")).strip()
            status = str(item.get("status", "")).strip().upper()
            note = str(item.get("note", "")).strip()
            if not criterion:
                continue
            if status not in {"PASS", "NEED"}:
                status = "NEED"
            line = f"- [{status}] {criterion}"
            if note:
                line += f": {note}"
            checklist_lines.append(line)

    if checklist_lines:
        sections.append("ELIGIBILITY CHECKLIST\n" + "\n".join(checklist_lines))

    suggestion_lines = []
    if isinstance(suggestions, list):
        sortable = []
        fallback = []
        for entry in suggestions:
            if isinstance(entry, dict):
                visa_name = str(entry.get("visa", "")).strip()
                reason = str(entry.get("reason", "")).strip()
                fit = entry.get("fit_percent", None)
                if visa_name:
                    try:
                        fit_num = float(fit) if fit is not None else -1.0
                    except Exception:
                        fit_num = -1.0
                    sortable.append((fit_num, visa_name, reason, fit))
            elif str(entry).strip():
                fallback.append(str(entry).strip())

        sortable.sort(key=lambda x: x[0], reverse=True)
        for fit_num, visa_name, reason, fit in sortable[:2]:
            line = f"- {visa_name}"
            if fit is not None and fit_num >= 0:
                line += f" ({fit_num:.0f}%)"
            if reason:
                line += f": {reason}"
            suggestion_lines.append(line)

        if len(suggestion_lines) < 2:
            for item in fallback[: 2 - len(suggestion_lines)]:
                suggestion_lines.append(f"- {item}")

    if suggestion_lines:
        sections.append("VISA SUGGESTIONS\n" + "\n".join(suggestion_lines))

    if next_steps:
        sections.append("NEXT STEPS\n" + "\n".join(f"- {item}" for item in next_steps))

    if notes:
        sections.append("IMPORTANT NOTES\n" + "\n".join(f"- {item}" for item in notes))

    if rationale:
        sections.append("ELIGIBILITY RATIONALE\n" + rationale)

    return "\n\n".join(sections)


# ------------------ LLM RESPONSE ------------------
def ask_llm(question, context, country="Unknown", visa_type="Other", use_memory=False):
    # Streamlit assessments should be stateless by default so previous turns do
    # not influence current eligibility scoring.
    history = ""
    if use_memory:
        history = memory.load_memory_variables({}).get("history", "")

    country_label = _country_label(country)
    visa_type_label = str(visa_type or "Other").strip()

    # Prompt for LLM
    prompt = f"""
You are an expert visa assistant helping users understand visa policies.

Destination Country:
{country_label}

Visa Type:
{visa_type_label}

Conversation History:
{history}

Policy Context:
{context}

User Question:
{question}

Return ONLY one valid JSON object with this schema:
{{
    "summary": "short 2-3 line plain-language summary",
    "overall_eligibility_percent": 0,
    "main_eligibility_criteria": ["..."],
    "what_user_has": ["..."],
    "what_is_needed": ["..."],
    "eligibility_checklist": [
        {{"criterion": "...", "status": "PASS|NEED", "note": "short reason"}}
    ],
    "visa_suggestions": [
        {{"visa": "visa name", "fit_percent": 0, "reason": "short reason"}}
    ],
    "next_steps": ["..."],
    "important_notes": ["..."],
    "eligibility_rationale": "brief reason for the percentage"
}}

Rules:
- Base your answer only on provided context and user profile.
- Make the answer specific to the destination country.
- Mention the most relevant visa pathway for that country.
- Be conservative if information is missing.
- Use simple English and avoid legal jargon where possible.
- Keep each bullet short and clear (ideally under 16 words).
- Keep list lengths tight for readability:
    - main_eligibility_criteria: up to 4 items
    - what_user_has: up to 4 items
    - what_is_needed: up to 5 items
    - eligibility_checklist: up to 5 items
    - visa_suggestions: up to 3 items (ranked best first)
    - next_steps: up to 3 items
    - important_notes: up to 3 items
- In eligibility_checklist, use only PASS or NEED for status.
- If the user question explicitly mentions a criterion or clear equivalent, mark it PASS.
- For salary, any explicit amount at or above the threshold counts as PASS.
- If unknown, explicitly state unknown in what_is_needed.
- If strong policy evidence exists, set overall_eligibility_percent above 80.
- Do not add any text before or after JSON.
"""

    # Generate response
    try:
        response = _invoke_llm(prompt, retries=1)
    except Exception as exc:
        err_text = str(exc)
        if _is_quota_error(err_text) or "not found" in err_text.lower() or "not supported" in err_text.lower():
            answer, eligibility_score = _offline_fallback_answer(question, context, country=country, visa_type=visa_type)
            return answer, eligibility_score, min(95.0, max(45.0, eligibility_score + 5.0))
        raise
    data = _extract_json_object(response.content)
    data["country"] = country_label
    data["visa_type"] = visa_type_label

    heuristic_applied, assessment_confidence = False, 0.0
    data, heuristic_applied, assessment_confidence = _build_explicit_fact_checklist(question, context, data)

    formatted_answer = _format_structured_answer(data)

    score = data.get("overall_eligibility_percent", 0)
    try:
        eligibility_score = max(0.0, min(100.0, float(score)))
    except Exception:
        eligibility_score = 0.0

    if heuristic_applied:
        eligibility_score = max(eligibility_score, float(data.get("overall_eligibility_percent", 0) or 0))
    else:
        assessment_confidence = min(95.0, max(45.0, eligibility_score * 0.7 + (5.0 if context else 0.0)))

    if not formatted_answer:
        formatted_answer = response.content

    decision = _decision_label(eligibility_score)
    formatted_answer = f"DECISION\n{decision}\n\n" + formatted_answer

    # Save conversation only when memory is explicitly enabled.
    if use_memory:
        memory.save_context(
            {"input": question},
            {"output": formatted_answer}
        )

    return formatted_answer, round(eligibility_score, 1), round(assessment_confidence, 1)


# ------------------ CHATBOT LOOP ------------------
def chatbot():
    print("\nVisa RAG Chatbot Ready")
    
    while True:
        question = input("You: ")
        
        if question.lower() == "exit":
            break
        
        # Detect country
        country = detect_country(question)
        
        try:
            # Retrieve documents
            context, confidence = retrieve_context(question, country)

            # Generate answer with structured sections and eligibility score
            answer, eligibility_score, assessment_confidence = ask_llm(
                question,
                context,
                country=country,
                use_memory=True,
            )

            print("\nAssistant:", answer)
            print(f"\nAnalysis Scores:")
            print(f"  - Eligibility Score: {eligibility_score}%")
            print(f"  - Confidence Score: {assessment_confidence}%")
            print(f"  - Retrieval Confidence: {confidence}%")
            print()
        except Exception as exc:
            print("\nAssistant: I could not process this request right now.")
            print(f"Reason: {exc}")
            print("Please try again in a moment.\n")


# ------------------ MAIN ------------------
if __name__ == "__main__":
    chatbot()