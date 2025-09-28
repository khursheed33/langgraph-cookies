LangGraph Time Travel
This document summarizes the time travel functionality in LangGraph, as detailed at LangGraph Time Travel Tutorial. It covers rewinding graphs using checkpoints, adding steps, replaying state history, and resuming from specific points in time.
Note: This tutorial builds on the "Customize state" concept.
Overview
In typical chatbot workflows, users interact sequentially. Memory and human-in-the-loop enable checkpoints for controlling responses. Time travel allows starting from previous responses to explore alternatives or rewind to fix mistakes, useful in applications like autonomous software engineers.
LangGraph's time travel uses checkpoints to rewind and resume execution.
1. Rewind Your Graph
Fetch checkpoints with get_state_history and resume from a previous state.
Setup
Install dependencies and initialize LLM (examples for OpenAI, Anthropic, Azure OpenAI, Google GenAI, AWS Bedrock provided; truncated for brevity).
API Reference: TavilySearch | BaseMessage | InMemorySaver | StateGraph | START | END | add_messages | ToolNode | tools_condition
from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

2. Add Steps
Add interactions; each step is checkpointed.
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

(Example output truncated; shows human message, AI response with tool call, tool message, and final AI summary.)
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

(Example output truncated; similar flow with tool call for autonomous agents.)
3. Replay the Full State History
Iterate through history to inspect all checkpoints.
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        to_replay = state

(Example output shows message counts and next nodes.)
Resume from a Checkpoint
Resume from selected state (e.g., after chatbot in second invocation).
print(to_replay.next)
print(to_replay.config)

for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()

(Resumes from tools node; output truncated.)
4. Load a State from a Moment-in-Time
Use checkpoint_id to load specific state.
The graph resumes from the tools node, as indicated by the first printed value being the search engine tool response.
Learn More
Explore deployment and advanced features for further LangGraph usage.
Conclusion
Time travel in LangGraph enables debugging, experimentation, and interactive applications by rewinding and exploring alternative paths via checkpoints.