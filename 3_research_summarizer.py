"""
Assumption: Ur vector database contains ur work, projects, reports etc.
that u have done in the past, and also contains information about the world that u have researched in the past. 
It is basically ur memory of all the research u have done in the past.

1. Get a research topic as input from user
2. Plan
    - what are the subtopics I need to research to write about this topic?
    - for each subtopic, check if its there in ur work in vector database
    - if not, search the web for it and store it in the vector database
3. Write a summary of the research topic based on the information in the vector database and the output of the web search
"""

from typing import TypedDict

from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import START, END, StateGraph


client = QdrantClient("http://localhost:6333")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./cache")    

vector_size = len(embeddings.embed_query("sample text"))

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

# store a pdf in the vector store, done only once when storing a new document, not needed for querying
# pdf_path = "docs/attention.pdf"
# reader = PdfReader(pdf_path)
# documents = []
# for page in reader.pages:
#     text = page.extract_text()
#     documents.append(text)


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
# texts = text_splitter.split_text("\n".join(documents))

# # embed and store the chunks in the vector store
# metadatas = [{"source": f"{pdf_path}_chunk_{i}"} for i in range(len(texts))]
# vector_store.add_texts(texts=texts, metadatas=metadatas)

# print("Done storing PDF in vector store")

class LLMPlan(BaseModel):
    subtopics: list[str]
 
llm = ChatOllama(model="llama3")

class ResearchSummarizerAgentState(TypedDict):
    research_topic: str
    subtopics: list[str]
    documents: list[dict[str, list[str]]]
    search_docs: list[str]
    summary: str

def clean(text: str) -> str:
    return text.strip().lower()

def get_research_topic(state: ResearchSummarizerAgentState) -> ResearchSummarizerAgentState:
    print("Getting research topic from user...")
    research_topic = input("Please enter the research topic: ")
    state['research_topic'] = clean(research_topic)
    return state

def plan(state: ResearchSummarizerAgentState) -> ResearchSummarizerAgentState:
    print("Planning...")
    structured_llm = llm.with_structured_output(LLMPlan)
    planning_prompt = f"""
    You are a research assistant.
    Your task is to break down the research topic into subtopics that need to be researched in order to write a comprehensive summary about the research topic.

    The research topic is: {state['research_topic']}
    Please provide a list of subtopics that need to be researched in order to write a comprehensive summary about the research topic.
    """
    plan = structured_llm.invoke(planning_prompt)
    state['subtopics'] = plan.subtopics
    return state

def retrieve_documents_from_vector_store(state: ResearchSummarizerAgentState) -> ResearchSummarizerAgentState:
    print("Retrieving documents from vector store...")
    documents = []
    for subtopic in state['subtopics']:
        results = vector_store.similarity_search(subtopic, k=3)
        if(results == []):
            print(f"No documents found for subtopic: {subtopic}")
            print(f"Will be searched from web")
            state['search_docs'].append(subtopic)
        else:
            documents.append({subtopic: [result.page_content for result in results]})
    state['documents'] = documents
    return state

def search_web(state: ResearchSummarizerAgentState) -> ResearchSummarizerAgentState:
    print("Searching the web for subtopics that were not found in the vector store...")
    search_tool = DuckDuckGoSearchResults()
    search_docs = []
    for subtopic in state['search_docs']:
        results = search_tool.invoke(subtopic)
        search_docs.append({subtopic: results})
    state['search_docs'] = search_docs
    return state

def write_summary(state: ResearchSummarizerAgentState) -> ResearchSummarizerAgentState:
    print("Writing summary...")
    summary_prompt = f"""
    You are a research assistant.
    Your task is to write a comprehensive summary about the research topic based on the information provided.

    The research topic is: {state['research_topic']}
    The subtopics that were researched are: {state['subtopics']}
    The documents retrieved from the vector store are: {state['documents']}
    The documents retrieved from the web search are: {state['search_docs']}

    Please write a comprehensive summary about the research topic based on the information provided.
    """
    summary = llm.invoke(summary_prompt)
    state['summary'] = summary.content
    return state

def main():
    graph = StateGraph(ResearchSummarizerAgentState)
    graph.add_node("get_research_topic", get_research_topic)
    graph.add_node("plan", plan)
    graph.add_node("retrieve_documents_from_vector_store", retrieve_documents_from_vector_store)
    graph.add_node("search_web", search_web)
    graph.add_node("write_summary", write_summary)

    graph.add_edge(START, "get_research_topic")
    graph.add_edge("get_research_topic", "plan")
    graph.add_edge("plan", "retrieve_documents_from_vector_store")
    graph.add_edge("retrieve_documents_from_vector_store", "search_web")
    graph.add_edge("search_web", "write_summary")
    graph.add_edge("write_summary", END)

    initial_state: ResearchSummarizerAgentState = {
        "research_topic": "",
        "subtopics": [],
        "documents": [],
        "search_docs": [],
        "summary": "",
    }
    app = graph.compile()
    final_state = app.invoke(initial_state)

    print("Summary:")
    print(final_state['summary'])


if __name__ == "__main__":
    main()







