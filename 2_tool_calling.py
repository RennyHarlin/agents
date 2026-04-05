"""
Agent that decides when to calls tools
like:
1. calculator
2. date
keeping it simple...the goal is not how complex the tools are,
but how the agent decides when to call them and how to use their output.
"""

from typing import TypedDict

from langchain.tools import tool
from datetime import datetime
from langgraph.graph import START, END, StateGraph, StateGraph

@tool 
def calculator(expression: str) -> str:
    """
     A simple calculator tool that evaluates basic arithmetic expressions.
    """
    # this is a very naive implementation of a calculator, just for demonstration purposes
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"    


@tool
def date() -> str:
    """
    A simple date tool that returns the current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ToolCallingAgentState(TypedDict):
    question: str
    tool_to_use: str
    tool_output: str

def get_input(state: ToolCallingAgentState) -> ToolCallingAgentState:
    question = input("Please enter your question: ")
    state['question'] = question
    return state

def decide_tool(state: ToolCallingAgentState) -> ToolCallingAgentState:
    question = state['question'].lower()
    if "calculate" in question:
        state['tool_to_use'] = "calculator"
    elif "date" in question:
        state['tool_to_use'] = "date"
    else:
        state['tool_to_use'] = "none"
    return state

def call_tool(state: ToolCallingAgentState) -> ToolCallingAgentState:
    if state['tool_to_use'] == "calculator":
        expression = state['question'].replace("calculate", "").strip()
        state['tool_output'] = calculator.invoke({"expression": expression})
    elif state['tool_to_use'] == "date":
        state['tool_output'] = date.invoke({})
    else:
        state['tool_output'] = "I don't know how to answer that."
    return state

def provide_answer(state: ToolCallingAgentState) -> ToolCallingAgentState:
    print(f"Answer: {state['tool_output']}")
    return state

def main():
    graph = StateGraph(ToolCallingAgentState)
    graph.add_node("get_input", get_input)
    graph.add_node("decide_tool", decide_tool)
    graph.add_node("call_tool", call_tool)
    graph.add_node("provide_answer", provide_answer)

    graph.add_edge(START, "get_input")
    graph.add_edge("get_input", "decide_tool")
    graph.add_edge("decide_tool", "call_tool")
    graph.add_edge("call_tool", "provide_answer")
    graph.add_edge("provide_answer", END)

    initial_state: ToolCallingAgentState = {"question": "", "tool_to_use": "", "tool_output": ""}
    app = graph.compile()
    app.invoke(initial_state)

if __name__ == "__main__":
    main()