from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class State(TypedDict, total=False):
    question: str
    country: str
    answer: str
    messages: list


def detect_country(state: State):
    question = state.get("question", "")
    if not question:
        messages = state.get("messages", [])
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                question = str(last_message.get("content", ""))
            else:
                question = str(getattr(last_message, "content", ""))

    q = str(question).lower()

    if "us" in q:
        country = "USA"
    elif "uk" in q or "united kingdom" in q:
        country = "UK"
    else:
        country = "Unknown"

    return {"question": str(question), "country": country}


def generate_answer(state: State):
    answer = f"Visa information for {state.get('country', 'Unknown')}"
    return {"answer": answer}


builder = StateGraph(State)

builder.add_node("detect_country", detect_country)
builder.add_node("generate_answer", generate_answer)

builder.set_entry_point("detect_country")

builder.add_edge("detect_country", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()