import os
import chromadb
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
try:
    from groq import Groq
except Exception:
    Groq = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
us_collection = client.get_or_create_collection("us_visa_collection")
uk_collection = client.get_or_create_collection("uk_visa_collection")
australia_collection = client.get_or_create_collection("australia_collection")


class State(TypedDict, total=False):
    question: str
    country: str
    context: str
    answer: str
    messages: list


def normalize_country(country_value):
    country_text = str(country_value or "").strip().lower()

    if country_text in ["uk", "u.k", "united kingdom", "britain", "great britain", "england", "british", "scotland", "wales"]:
        return "UK"
    if country_text in ["us", "u.s", "usa", "u.s.a", "united states", "america", "american"]:
        return "USA"
    return "Unknown"


def extract_question(state: State):
    question = state.get("question", "")
    if question:
        return str(question)

    messages = state.get("messages", [])
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))
        return str(getattr(last_message, "content", ""))

    return ""


def detect_Country(state: State):
    explicit_country = normalize_country(state.get("country", ""))

    question = extract_question(state)

    q = str(question).lower()

    uk_keywords = ["uk", "united kingdom", "britain", "great britain", "england", "british", "scotland", "wales"]
    us_keywords = ["us visa", "usa", "united states", "america", "american visa", "us immigration", "uscis", "green card", "h1b", "h-1b", "f1 visa", "f-1"]

    if explicit_country != "Unknown":
        country = explicit_country
    elif any(kw in q for kw in uk_keywords):
        country = "UK"
    elif any(kw in q for kw in us_keywords):
        country = "USA"
    else:
        country = "Unknown"

    return {"question": str(question), "country": country}


def retireve_document(state: State):
    question = extract_question(state)
    country = normalize_country(state.get("country", "Unknown"))

    if country == "Unknown":
        return {"question": question, "country": "Unknown", "context": ""}

    if not question:
        return {"question": "", "country": country, "context": ""}

    if country == "UK":
        collection = uk_collection
    elif country == "USA":
        collection = us_collection
    elif country == "AUSTRALIA":
        collection = australia_collection
    else:
        return {"question": question, "country": country, "context": ""}

    results = collection.query(query_texts=[question], n_results=20, include=["documents", "metadatas"])
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    # Combine context with metadata for traceability
    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_blocks.append(f"[Source: {source}, Page: {page}]\n{doc}")
    context = "\n\n".join(context_blocks)
    return {"question": question, "country": country, "context": context}


def generate_answer(state: State):
    question = extract_question(state)
    country = normalize_country(state.get("country", "Unknown"))
    context = state.get("context", "")

    if not question:
        return {"answer": "Please provide your question. Example input: {\"question\": \"What is the visa process?\", \"country\": \"UK\"}."}

    if country == "Unknown":
        return {"answer": "Please provide country in JSON format. Example: {\"question\": \"What is the visa process?\", \"country\": \"UK\"} or {\"country\": \"USA\"}."}

    if not context:
        return {"answer": f"I could not find policy documents for {country}. Please try rephrasing your question."}

    if Groq is None:
        return {"answer": "Groq SDK is not installed. Run: pip install groq==0.9.0"}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"answer": "GROQ_API_KEY is missing. Add it to your environment or Streamlit secrets."}

    client = Groq(api_key=api_key)

    prompt = f"""You are an expert visa assistant helping users understand visa policies.

Policy Context:
{context}

User Question:
{question}

Instructions:
- Answer using the policy context provided.
- If the context does not contain the answer, say so clearly.
- Be clear and concise."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {"answer": response.choices[0].message.content}


builder = StateGraph(State)

builder.add_node("detect_Country", detect_Country)
builder.add_node("retireve_document", retireve_document)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("detect_Country")

builder.add_edge("detect_Country", "retireve_document")
builder.add_edge("retireve_document", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

if __name__ == "__main__":
    # Example input for the graph
    input_state = {
        "question": "What is the visa process for the UK?",
        "country": "UK"
    }
    result = graph.invoke(input_state)
    print("Result:", result)