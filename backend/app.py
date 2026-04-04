import streamlit as st
import json
import os
import re
from datetime import datetime

# 🔥 IMPORT YOUR BACKEND FUNCTIONS
from rag_chatbot import retrieve_context, ask_llm, detect_country

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ELIGIBILITY_LOG_FILE = os.path.join(BASE_DIR, "eligibility_records.jsonl")

VISA_TYPES = [
    "Work Visa",
    "Student Visa",
    "Tourist Visa",
    "Business Visa",
    "Dependent Visa",
]

DEFAULT_CURRENCY = "INR"
USD_TO_INR_RATE = 83

def normalize_score(raw_score):
    try:
        return max(0.0, min(100.0, float(raw_score)))
    except Exception:
        return 0.0

def score_band(score):
    if score >= 80:
        return "Most Likely"
    if score >= 50:
        return "Likely"
    return "Unlikely"

def extract_profile_bullets(question, max_items=5):
    parts = re.split(r"[\n.;]+", str(question))
    bullets = [segment.strip(" -\t") for segment in parts if segment.strip()]
    filtered = [item for item in bullets if len(item) >= 6]
    return filtered[:max_items]

def extract_answer_bullets(answer, max_items=8):
    lines = str(answer).splitlines()
    bullets = []
    for line in lines:
        text = line.strip()
        if text.startswith("- "):
            bullets.append(text[2:].strip())
    return bullets[:max_items]


def parse_context_sources(context):
    pattern = re.compile(r"\[Source:\s*([^,\]]+)\s*,\s*Page:\s*([^\]]+)\]")
    found = pattern.findall(str(context))
    deduped = []
    seen = set()
    for source, page in found:
        key = (source.strip(), page.strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"source": key[0], "page": key[1]})
    return deduped


def extract_section_block(answer, section_name, all_sections):
    escaped_section = re.escape(section_name)
    next_sections = [name for name in all_sections if name != section_name]
    next_pattern = "|".join(re.escape(name) for name in next_sections)

    if next_pattern:
        pattern = rf"(?ms)^\s*{escaped_section}\s*$\n(.*?)(?=^\s*(?:{next_pattern})\s*$|\Z)"
    else:
        pattern = rf"(?ms)^\s*{escaped_section}\s*$\n(.*)\Z"

    match = re.search(pattern, str(answer))
    if not match:
        return ""
    return match.group(1).strip()


def parse_answer_sections(answer):
    section_names = [
        "DECISION",
        "SUMMARY",
        "MAIN ELIGIBILITY CRITERIA",
        "WHAT USER HAS",
        "WHAT IS NEEDED",
        "ELIGIBILITY CHECKLIST",
        "VISA SUGGESTIONS",
        "NEXT STEPS",
        "IMPORTANT NOTES",
        "ELIGIBILITY RATIONALE",
    ]
    parsed = {}
    for name in section_names:
        parsed[name] = extract_section_block(answer, name, section_names)
    return parsed


def strip_decision_block(answer):
    return re.sub(r"(?ms)^\s*DECISION\s*\n\s*[^\n]+\n*", "", str(answer), count=1).strip()


def normalize_yes_no(value):
    return "Yes" if str(value).strip().lower() == "yes" else "No"


def clean_key(label):
    key = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
    return key.strip("_")


def render_dynamic_fields(visa_type):
    details = {}

    if visa_type == "Work Visa":
        col1, col2 = st.columns(2)
        with col1:
            details["job_offer"] = normalize_yes_no(st.selectbox("Job Offer", ["Yes", "No"], key="work_job_offer"))
            details["salary"] = float(st.number_input(f"Salary ({DEFAULT_CURRENCY})", min_value=0.0, step=500.0, key="work_salary"))
            details["occupation_or_job_role"] = st.text_input("Occupation / Job Role", key="work_role")
        with col2:
            details["employer_sponsorship"] = normalize_yes_no(
                st.selectbox("Employer Sponsorship", ["Yes", "No"], key="work_sponsorship")
            )
            details["work_experience_years"] = int(
                st.slider("Work Experience (years)", min_value=0, max_value=30, value=2, key="work_experience")
            )

    elif visa_type == "Student Visa":
        col1, col2 = st.columns(2)
        with col1:
            details["university_name"] = st.text_input("University Name", key="student_university")
            details["offer_letter"] = normalize_yes_no(st.selectbox("Offer Letter", ["Yes", "No"], key="student_offer"))
            details["course_level"] = st.selectbox(
                "Course Level",
                ["Bachelors", "Masters"],
                key="student_course_level",
            )
        with col2:
            details["ielts_toefl_score"] = float(
                st.number_input("IELTS/TOEFL Score", min_value=0.0, max_value=120.0, value=6.5, step=0.5, key="student_test")
            )
            funds_usd = float(
                st.number_input("Financial Support (USD)", min_value=0.0, step=100.0, key="student_financial_support")
            )
            funds_inr = funds_usd * USD_TO_INR_RATE
            st.write(f"${funds_usd:,.2f} (Rs {funds_inr:,.0f})")
            details["financial_support_available_usd"] = funds_usd
            details["financial_support_available"] = funds_inr

    elif visa_type == "Tourist Visa":
        col1, col2 = st.columns(2)
        with col1:
            details["travel_duration_days"] = int(
                st.slider("Travel Duration (days)", min_value=1, max_value=180, value=14, key="tourist_duration")
            )
            details["purpose_of_visit"] = st.text_input("Purpose of Visit", key="tourist_purpose")
            details["travel_history"] = normalize_yes_no(
                st.selectbox("Travel History", ["Yes", "No"], key="tourist_history")
            )
        with col2:
            funds_usd = float(
                st.number_input("Proof of Funds (USD)", min_value=0.0, step=100.0, key="tourist_funds")
            )
            funds_inr = funds_usd * USD_TO_INR_RATE
            st.write(f"${funds_usd:,.2f} (Rs {funds_inr:,.0f})")
            details["proof_of_funds_usd"] = funds_usd
            details["proof_of_funds"] = funds_inr
            details["return_ticket"] = normalize_yes_no(st.selectbox("Return Ticket", ["Yes", "No"], key="tourist_ticket"))

    elif visa_type == "Business Visa":
        col1, col2 = st.columns(2)
        with col1:
            details["company_name"] = st.text_input("Company Name", key="business_company")
            details["purpose_of_visit"] = st.text_input("Purpose of Visit", key="business_purpose")
            details["invitation_letter"] = normalize_yes_no(
                st.selectbox("Invitation Letter", ["Yes", "No"], key="business_invitation")
            )
        with col2:
            details["business_experience"] = st.text_input("Business Experience", key="business_experience")
            details["annual_income"] = float(
                st.number_input(f"Annual Income ({DEFAULT_CURRENCY})", min_value=0.0, step=1000.0, key="business_income")
            )

    elif visa_type == "Dependent Visa":
        col1, col2 = st.columns(2)
        with col1:
            details["sponsor_name"] = st.text_input("Sponsor Name", key="dependent_sponsor_name")
            details["relationship_to_sponsor"] = st.selectbox(
                "Relationship to Sponsor",
                ["Spouse", "Child", "Parent", "Other"],
                key="dependent_relationship",
            )
            details["sponsor_visa_status"] = st.text_input("Sponsor Visa Status", key="dependent_visa_status")
        with col2:
            details["proof_of_relationship"] = normalize_yes_no(
                st.selectbox("Proof of Relationship", ["Yes", "No"], key="dependent_proof")
            )
            details["sponsor_income"] = float(
                st.number_input(f"Sponsor Income ({DEFAULT_CURRENCY})", min_value=0.0, step=1000.0, key="dependent_income")
            )

    return details


def build_structured_profile(name, age, nationality, target_country, visa_type, details, question, currency=DEFAULT_CURRENCY):
    return {
        "name": name or "N/A",
        "age": int(age),
        "nationality": nationality,
        "target_country": target_country,
        "visa_type": visa_type,
        "currency": currency,
        "details": details,
        "question": question,
    }


def build_llm_question(profile):
    detail_lines = []
    for key, value in profile.get("details", {}).items():
        label = key.replace("_", " ").title()
        detail_lines.append(f"- {label}: {value}")

    detail_text = "\n".join(detail_lines) if detail_lines else "- No additional visa details provided"

    return (
        "User Profile:\n"
        f"- Name: {profile.get('name', 'N/A')}\n"
        f"- Age: {profile.get('age', 0)}\n"
        f"- Country: {profile.get('nationality', 'Unknown')}\n"
        f"- Target Country: {profile.get('target_country', 'Unknown')}\n"
        f"- Visa Type: {profile.get('visa_type', 'Other')}\n"
        f"- Currency: {profile.get('currency', DEFAULT_CURRENCY)}\n"
        f"{detail_text}\n\n"
        "User Question:\n"
        f"{profile.get('question', '')}\n\n"
        "Based on visa policy documents, evaluate eligibility and provide: "
        "eligibility score, confidence score, explanation, and suggestions to improve."
    )


def build_report_text(record):
    details = record.get("structured_input", {}).get("details", {})
    detail_lines = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in details.items()]

    section_lines = [
        "SwiftVisa Eligibility Report",
        "",
        f"Timestamp (UTC): {record.get('timestamp_utc', 'N/A')}",
        f"Name: {record.get('name', 'N/A')}",
        f"Age: {record.get('age', 'N/A')}",
        f"Nationality: {record.get('user_country', 'N/A')}",
        f"Target Country: {record.get('visa_destination_country', 'N/A')}",
        f"Visa Type: {record.get('visa_type', 'N/A')}",
        f"Currency: {record.get('structured_input', {}).get('currency', DEFAULT_CURRENCY)}",
        f"Eligibility Score: {record.get('eligibility_score', 0)}%",
        f"Confidence Score: {record.get('confidence_score', 0)}%",
        f"Retrieval Confidence Score: {record.get('retrieval_confidence_score', 0)}%",
        "",
        "Question:",
        record.get("question", ""),
        "",
        "Dynamic Visa Details:",
    ]

    if detail_lines:
        section_lines.extend(detail_lines)
    else:
        section_lines.append("- No dynamic details")

    section_lines.extend(
        [
            "",
            "Assistant Response:",
            record.get("answer", ""),
        ]
    )

    return "\n".join(section_lines)


def save_eligibility_record(record):
    with open(ELIGIBILITY_LOG_FILE, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


def normalize_country_key(country_value):
    value = str(country_value).strip().lower()
    mapping = {
        "us": "us",
        "usa": "us",
        "uk": "uk",
        "australia": "australia",
    }
    return mapping.get(value, value)


def country_display_name(country_key):
    mapping = {
        "us": "USA",
        "uk": "UK",
        "australia": "Australia",
    }
    return mapping.get(str(country_key).lower(), str(country_key).upper())

st.set_page_config(
    page_title="SwiftVisa AI Assistant",
    layout="wide",
)

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "last_saved_record" not in st.session_state:
    st.session_state.last_saved_record = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "reset_target_country" not in st.session_state:
    st.session_state.reset_target_country = False

if "try_another_country_mode" not in st.session_state:
    st.session_state.try_another_country_mode = False

if "pending_target_country" not in st.session_state:
    st.session_state.pending_target_country = None

if "preferred_new_country" not in st.session_state:
    st.session_state.preferred_new_country = "USA"

st.markdown(
    """
    <style>
        :root {
            --bg-base: var(--background-color);
            --card-bg: var(--secondary-background-color);
            --body-text: var(--text-color);
            --hero-title: var(--text-color);
            --hero-subtitle: var(--text-color);
            --card-border: rgba(148, 163, 184, 0.35);
            --hero-border: rgba(148, 163, 184, 0.35);
            --muted-text: rgba(148, 163, 184, 0.92);
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.12) 0%, transparent 44%),
                radial-gradient(circle at 85% 5%, rgba(16, 185, 129, 0.10) 0%, transparent 38%),
                var(--bg-base);
            color: var(--body-text);
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: var(--body-text);
        }

        .stCaption {
            color: var(--muted-text) !important;
        }

        .chat-card {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: var(--card-bg);
            margin-bottom: 0.7rem;
        }
        .hero {
            border: 1px solid var(--hero-border);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(140deg, var(--card-bg) 0%, rgba(59, 130, 246, 0.08) 100%);
            margin-bottom: 1rem;
        }
        .result-card {
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: var(--card-bg);
            color: var(--body-text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HERO ----------------
st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0 0 0.3rem 0; color:var(--hero-title);">SwiftVisa AI Assistant</h2>
        <p style="margin:0; color:var(--hero-subtitle);">
            Check your visa eligibility with AI-generated guidance and confidence scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Input Settings")

    if st.session_state.pending_target_country:
        st.session_state.target_country = st.session_state.pending_target_country
        st.session_state.pending_target_country = None

    if st.session_state.reset_target_country:
        st.session_state.target_country = "Auto Detect"
        st.session_state.reset_target_country = False

    country_override = st.selectbox(
        "Target country",
        ["Auto Detect", "USA", "UK", "Australia"],
        key="target_country",
        help="Use Auto Detect to infer country from your question.",
    )

    st.caption("Quick examples")
    if st.button("Use USA example", use_container_width=True):
        st.session_state.visa_type = "Work Visa"
        st.session_state.question_input = (
            "I have a US software job offer, employer sponsorship, 5 years of "
            "experience, and a valid passport. Am I eligible for H-1B?"
        )
    if st.button("Use UK example", use_container_width=True):
        st.session_state.visa_type = "Student Visa"
        st.session_state.question_input = (
            "I completed a UK degree and have a confirmed employer sponsorship "
            "for a skilled role. Can I qualify for a UK Skilled Worker visa?"
        )

    if st.button("Try Another Country", use_container_width=True):
        st.session_state.try_another_country_mode = True
        st.session_state.reset_target_country = False
        st.rerun()

    if st.session_state.try_another_country_mode:
        st.caption("Choose your preferred country for the next eligibility check.")
        st.selectbox(
            "Which country do you prefer?",
            ["USA", "UK", "Australia"],
            key="preferred_new_country",
        )
        if st.button("Apply Preferred Country", use_container_width=True):
            st.session_state.pending_target_country = st.session_state.preferred_new_country
            st.session_state.try_another_country_mode = False
            st.rerun()

    if st.button("Reset Form", use_container_width=True):
        keep_keys = {"query_history", "chat_messages"}
        for state_key in list(st.session_state.keys()):
            if state_key not in keep_keys:
                del st.session_state[state_key]
        st.rerun()

    st.markdown("---")
    st.caption("Tip: Include education, job offer, sponsorship, funds, and passport status.")

# ---------------- INPUT FORM ----------------
st.subheader("Your Profile")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    user_name = st.text_input(
        "Full Name",
        key="user_name",
        placeholder="e.g., John Doe",
    )
with col2:
    user_age = st.number_input(
        "Age",
        key="user_age",
        min_value=18,
        max_value=99,
    )
with col3:
    user_country = st.selectbox(
        "Nationality",
        options=["USA", "UK", "Canada", "India", "Australia", "Other"],
        key="user_country",
    )

col4, col5 = st.columns([1, 1])
with col4:
    visa_type = st.selectbox(
        "Visa Type",
        key="visa_type",
        options=VISA_TYPES,
    )
with col5:
    st.info(f"Target Country: {country_override}")

st.subheader("Visa-Specific Details")
st.caption(f"All money fields use {DEFAULT_CURRENCY} (Indian Rupees).")
visa_details = render_dynamic_fields(visa_type)

st.markdown("---")
st.subheader("Your Question")
question = st.text_area(
    "Describe your profile and ask your question",
    key="question_input",
    placeholder="Example: I am an M.Tech graduate with a US software job offer, employer sponsorship, and valid passport. Am I eligible for H-1B?",
    height=180,
)

if len(question.strip()) > 20:
    st.info("AI is understanding your profile details and preparing policy-based checks.")

st.subheader("Chat Preview")
if st.session_state.chat_messages:
    for msg in st.session_state.chat_messages[-6:]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
else:
    st.caption("No chat messages yet. Submit your profile and question to begin.")

submit = st.button("Check Eligibility", type="primary")

# ---------------- BACKEND CONNECTION ----------------
if submit:
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    structured_profile = build_structured_profile(
        name=user_name,
        age=user_age,
        nationality=user_country,
        target_country=country_override,
        visa_type=visa_type,
        details=visa_details,
        question=question,
        currency=DEFAULT_CURRENCY,
    )
    llm_question = build_llm_question(structured_profile)

    with st.spinner("Analyzing your eligibility..."):
        # Let users force target country or auto-detect from question.
        detected_country_key = (
            detect_country(llm_question)
            if country_override == "Auto Detect"
            else normalize_country_key(country_override)
        )

        detected_country_label = country_display_name(detected_country_key)

        context, retrieval_confidence = retrieve_context(llm_question, detected_country_key)
        answer, eligibility_score, assessment_confidence = ask_llm(
            llm_question,
            context,
            country=detected_country_key,
            visa_type=visa_type,
        )

    eligibility_score = normalize_score(eligibility_score)
    retrieval_confidence = normalize_score(retrieval_confidence)
    assessment_confidence = normalize_score(assessment_confidence)
    band = score_band(eligibility_score)
    profile_bullets = extract_profile_bullets(question)
    answer_bullets = extract_answer_bullets(answer)
    parsed_sections = parse_answer_sections(answer)
    policy_sources = parse_context_sources(context)

    explanation_text = (
        parsed_sections.get("ELIGIBILITY RATIONALE")
        or parsed_sections.get("SUMMARY")
        or answer
    )
    suggestion_candidates = []
    for section_name in ["VISA SUGGESTIONS", "NEXT STEPS", "WHAT IS NEEDED"]:
        block = parsed_sections.get(section_name, "")
        if block:
            for line in block.splitlines():
                cleaned = line.strip().lstrip("- ").strip()
                if cleaned:
                    suggestion_candidates.append(cleaned)
    suggestions = suggestion_candidates[:5]

    record = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "name": user_name or "N/A",
        "age": int(user_age),
        "user_country": user_country,
        "visa_type": visa_type,
        "visa_destination_country": detected_country_label,
        "visa_destination_country_key": detected_country_key,
        "structured_input": structured_profile,
        "llm_question": llm_question,
        "question": question,
        "context": context,
        "answer": answer,
        "explanation": explanation_text,
        "suggestions": suggestions,
        "policy_sources": policy_sources,
        "eligibility_score": round(eligibility_score, 1),
        "confidence_score": round(assessment_confidence, 1),
        "retrieval_confidence_score": round(retrieval_confidence, 1),
        "score_band": band,
        "profile_bullets": profile_bullets,
        "answer_bullets": answer_bullets,
    }
    save_eligibility_record(record)
    st.session_state.last_saved_record = record
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": f"Visa Type: {visa_type}\n\n{question}",
        }
    )
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.success("✅ Analysis complete")

    # -------- PROFILE SUMMARY --------
    st.subheader("📋 Your Profile")
    profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
    with profile_col1:
        st.metric("Full Name", user_name or "N/A", delta=None)
    with profile_col2:
        st.metric("Age", int(user_age), delta=None)
    with profile_col3:
        st.metric("Nationality", user_country, delta=None)
    with profile_col4:
        st.metric("Visa Category", visa_type, delta=None)

    with st.expander("View structured JSON sent to backend"):
        st.json(structured_profile)

    # -------- ELIGIBILITY SCORES --------
    st.subheader("🎯 Eligibility Assessment")
    score_col1, score_col2, score_col3 = st.columns([1.3, 1.3, 1.2])
    with score_col1:
        st.metric("Destination Country", detected_country_label)
    with score_col2:
        st.metric("Eligibility Score", f"{eligibility_score:.1f}%")
    with score_col3:
        st.metric("Confidence Score", f"{assessment_confidence:.1f}%")

    st.caption("Confidence reflects how much of your checklist is " \
    "explicitly supported by the question and policy context.")

    bounded_score = max(0, min(100, int(eligibility_score))) / 100
    st.progress(bounded_score, text=f"Eligibility Progress - {band} Assessment")
    st.progress(
        max(0, min(100, int(assessment_confidence))) / 100,
        text=f"Confidence Progress - {assessment_confidence:.1f}%",
    )

    # -------- DETAILED ANALYSIS --------
    st.markdown("---")
    st.subheader("📄 Explanation")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.write(explanation_text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("💡 Suggestions")
    if suggestions:
        for tip in suggestions:
            st.markdown(f"- {tip}")
    else:
        st.write("- Keep profile details explicit (funds, sponsorship, offer letters, and timelines).")

    st.subheader("📊 Full AI Response")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.write(strip_decision_block(answer))
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📚 Policy Sources (RAG)"):
        if policy_sources:
            for source_item in policy_sources:
                st.markdown(f"- {source_item['source']} (page {source_item['page']})")
        else:
            st.write("No explicit policy sources were parsed from retrieved context.")

    report_text = build_report_text(record)
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"swiftvisa_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    st.caption(f"Saved record: {ELIGIBILITY_LOG_FILE}")

    st.session_state.query_history.insert(
        0,
        {
            "name": user_name or "N/A",
            "age": int(user_age),
            "user_country": user_country,
            "visa_type": visa_type,
            "question": question,
            "destination_country": detected_country_label,
            "eligibility": round(eligibility_score, 1),
            "confidence": round(assessment_confidence, 1),
            "retrieval_confidence": round(retrieval_confidence, 1),
            "band": band,
            "profile_bullets": profile_bullets,
            "answer_bullets": answer_bullets,
            "suggestions": suggestions,
        },
    )
    st.session_state.query_history = st.session_state.query_history[:5]

if st.session_state.query_history:
    st.markdown("---")
    st.subheader("Recent Assessments")
    for idx, item in enumerate(st.session_state.query_history, start=1):
        st.caption(
            f"{idx}. {item['name']} ({item['age']}) | {item['user_country']} → {item['destination_country']} | "
            f"Visa: {item['visa_type']} | Eligibility: {item['eligibility']}% | "
            f"Confidence: {item.get('confidence', 0)}% | Band: {item.get('band', 'N/A')}"
        )
        with st.expander("View details"):
            st.write("Question")
            st.write(item["question"])
            st.write(f"Eligibility Score: {item['eligibility']}%")
            st.write(f"Confidence Score: {item.get('confidence', 0)}%")
            st.write(f"Retrieval Confidence: {item.get('retrieval_confidence', 0)}%")

            st.write("Profile bullet points")
            if item.get("profile_bullets"):
                for bullet in item["profile_bullets"]:
                    st.markdown(f"- {bullet}")
            else:
                st.write("- No profile bullet points")

            st.write("Suggestions")
            if item.get("suggestions"):
                for tip in item["suggestions"]:
                    st.markdown(f"- {tip}")
            else:
                st.write("- No suggestions")

            st.write("Answer bullet points")
            if item.get("answer_bullets"):
                for bullet in item["answer_bullets"]:
                    st.markdown(f"- {bullet}")
            else:
                st.write("- No answer bullet points")