"""
The goal here is not to build an agent, but to build a simple FAQ bot using the LangGraph's StateGraph. That's what is accomplished here.
FAQ Bot
1. Accept question from the user
2. Compare it against a list of FAQs
3. Pick the best match
4. if the match is good enough, return the answer, otherwise say "I don't know"
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# this is the overall state of the agent, which will be passed between nodes
# each tool of the agent will read and write to this state
class FAQAgentState(TypedDict):
    question: str
    best_match: str
    answer: str

# should be properly implemented using a vector database and a more sophisticated similarity function, but I don't care :)
sample_faqs = {
    "What is your name?": "My name is FAQ Bot.",
    "How are you?": "I'm doing well, thank you!",
    "What can you do?": "I can answer your questions based on my FAQ database.",
}

def similarity(a: str, b: str) -> float:
    # A simple similarity function based on common words
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    common_words = a_words.intersection(b_words)
    return len(common_words) / max(len(a_words), len(b_words))

def find_best_match(question: str) -> str:
    best_match = None
    best_score = 0
    for faq in sample_faqs.keys():
        score = similarity(question, faq)
        if score > best_score and score > 0.5:  # Threshold for a good match
            best_score = score
            best_match = faq
    return best_match

# Each node of the graph is a function that takes the state, does something, and returns the new state
def get_question(state: FAQAgentState) -> FAQAgentState:
    question = input("Please enter your question: ")
    state['question'] = question
    return state

# This node finds the best matching FAQ for the user's question and updates the state with it
def find_best_match_node(state: FAQAgentState) -> FAQAgentState:
    best_match = find_best_match(state['question'])
    state['best_match'] = best_match
    return state

# This node provides the answer based on the best match found, or says "I don't know" if no good match was found
def provide_answer(state: FAQAgentState) -> FAQAgentState:
    if state['best_match']:
        state["answer"] = sample_faqs[state['best_match']]
        print(f"Answer: {state['answer']}")
    else:
        print("I don't know the answer to that question.")
    return state

def main():
    graph = StateGraph(FAQAgentState)
    graph.add_node("get_question", get_question)
    graph.add_node("find_best_match", find_best_match_node)
    graph.add_node("provide_answer", provide_answer)
    graph.add_edge(START, "get_question")
    graph.add_edge("get_question", "find_best_match")
    graph.add_edge("find_best_match", "provide_answer")
    graph.add_edge("provide_answer", END)

    initial_state: FAQAgentState = {"question": "", "best_match": "", "answer": ""}
    app = graph.compile()
    app.invoke(initial_state)


if __name__ == "__main__":
    main()