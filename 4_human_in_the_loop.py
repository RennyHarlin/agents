"""
Goal here is not a sophisticated agent,
but to just have a bare implementation of HITL.

Let's say you work in customer service in a laptop firm and you have to answer customer queries.
1. You have a vector database of all the laptop manuals and troubleshooting guides.
2. You have access to the web to search for any information that is not in the vector database.
3. But, then the information from the web has to be verified by you before you can use it to answer the customer's query.

Here, we will implement a simple agent that mimics this behavior.
"""

from typing import TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import ChatOllama

client = QdrantClient("http://localhost:6333")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./cache")    

vector_size = len(embeddings.embed_query("sample text"))

llm = ChatOllama(model="llama3")

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

class CustomerServiceAgentState(TypedDict):
    customer_query: str
    related_docs: list[str]
    web_search_results: list[str]
    summarized_info: str
    summarized_web_info: str

    is_web_search_needed: bool
    verified_info: bool


def get_customer_query(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    state["customer_query"] = input("Please enter the customer's query: ")
    return state

def retrieve_documents_from_vector_store(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    print("Retrieving documents from vector store...")
    results = vector_store.similarity_search(state["customer_query"], k=5)
    if not results:
        state["is_web_search_needed"] = True
        print("No relevant documents found in vector store. Web search is needed.")
    else:
        print(f"Documents retrieved from vector store: {results}")
        state["related_docs"] = [str(doc.page_content) for doc in results]
    return state

def search_web(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    print("Searching the web for subtopics that were not found in the vector store...")
    if not state["is_web_search_needed"]:
        return state
    search_tool = DuckDuckGoSearchResults()
    results = search_tool.invoke(state["customer_query"])
    state["web_search_results"] = [str(results)]
    print(f"Documents retrieved from web search: {state['web_search_results']}")
    return state

def summarize_all_info(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    if state["is_web_search_needed"] and state["web_search_results"]:
        print("Summarizing the information retrieved from the web search...")
        # Summarize web results into one concise answer draft.
        summarized_response = llm.invoke("\n".join(state["web_search_results"]))
        state["summarized_web_info"] = summarized_response.content
        print(f"Summarized web information: {state['summarized_web_info']}")
    else:
        summarized_response = llm.invoke("\n".join(state["related_docs"]))
        state["summarized_info"] = summarized_response.content
        print(f"Summarized information from vector store: {state['summarized_info']}")
    return state

def verify_web_info(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    if state["is_web_search_needed"]:
        print("Verifying the information retrieved from the web search...")
        print(f"Web search results: {state['summarized_web_info']}")
        verification = input("Please verify the information retrieved from the web search (yes/no): ")
        if verification.lower() == "yes":
            state["verified_info"] = True
            print("Information verified successfully.")
    return state

def ans_query(state: CustomerServiceAgentState) -> CustomerServiceAgentState:
    if state["related_docs"]:
        print(f"Answering the customer's query using the related documents from the vector store: {state['summarized_info']}")
    elif state["verified_info"]:
        print(f"Answering the customer's query using the verified information from the web search: {state['summarized_web_info']}")
    else:
        print("Sorry, I couldn't find any relevant information to answer the customer's query.")
    return state

def main():
    graph = StateGraph(CustomerServiceAgentState)
    graph.add_node("get_customer_query", get_customer_query)
    graph.add_node("retrieve_documents_from_vector_store", retrieve_documents_from_vector_store)
    graph.add_node("search_web", search_web)
    graph.add_node("summarize_all_info", summarize_all_info)
    graph.add_node("verify_web_info", verify_web_info)
    graph.add_node("ans_query", ans_query)

    graph.add_edge(START, "get_customer_query")
    graph.add_edge("get_customer_query", "retrieve_documents_from_vector_store")
    graph.add_edge("retrieve_documents_from_vector_store", "search_web")
    graph.add_edge("search_web", "summarize_all_info")
    graph.add_edge("summarize_all_info", "verify_web_info")
    graph.add_edge("verify_web_info", "ans_query")
    graph.add_edge("ans_query", END)

    initial_state: CustomerServiceAgentState = {
        "customer_query": "",
        "related_docs": [],
        "web_search_results": [],
        "summarized_info": "",
        "summarized_web_info": "",
        "is_web_search_needed": False,
        "verified_info": False,
    }
    app = graph.compile()
    app.invoke(initial_state)


if __name__ == "__main__":
    main()
    