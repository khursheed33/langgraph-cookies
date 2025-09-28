LangGraph Streaming
This document summarizes the streaming capabilities in LangGraph, as detailed at LangGraph Streaming. It covers supported stream modes, basic usage, streaming graph state, subgraph outputs, LLM tokens, custom data, and considerations for older Python versions.
Alpha Notice: The documentation covers the v1-alpha release and is subject to change. For the latest stable version, refer to the current LangGraph Python or JavaScript documentation.
Overview
Streaming in LangGraph provides real-time updates, improving UX by displaying progressive output despite LLM latency. Capabilities include streaming graph state, subgraph outputs, LLM tokens, custom data, and multiple modes.
Supported Stream Modes



Mode
Description



values
Streams full state after each step.


updates
Streams state deltas after each step (separate for multiple updates).


custom
Streams custom data from nodes.


messages
Streams LLM tokens with metadata.


debug
Streams detailed execution info.


Basic Usage
Use .stream() (sync) or .astream() (async) to yield outputs as iterators:
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)

Stream Multiple Modes
Pass a list for multiple modes; outputs are (mode, chunk) tuples:
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
    print(chunk)

Stream Graph State
Use updates or values:
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = StateGraph(State).add_node(refine_topic).add_node(generate_joke)
graph.add_edge(START, "refine_topic").add_edge("refine_topic", "generate_joke").add_edge("generate_joke", END)
graph = graph.compile()

for chunk in graph.stream({"topic": "ice cream"}, stream_mode="updates"):
    print(chunk)

Stream Subgraph Outputs
Set subgraphs=True for outputs as (namespace, data):
for chunk in graph.stream({"foo": "foo"}, subgraphs=True, stream_mode="updates"):
    print(chunk)

Debugging
Use debug for detailed traces:
for chunk in graph.stream({"topic": "ice cream"}, stream_mode="debug"):
    print(chunk)

LLM Tokens
Use messages for token streams as (message_chunk, metadata):
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

@dataclass
class MyState:
    topic: str
    joke: str = ""

llm = init_chat_model(model="openai:gpt-4o-mini")

def call_model(state: MyState):
    llm_response = llm.invoke([{"role": "user", "content": f"Generate a joke about {state.topic}"}])
    return {"joke": llm_response.content}

graph = StateGraph(MyState).add_node(call_model).add_edge(START, "call_model").compile()

for message_chunk, metadata in graph.stream({"topic": "ice cream"}, stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)

Filter by LLM Invocation
Tag LLMs and filter by metadata["tags"]:
llm_1 = init_chat_model(model="openai:gpt-4o-mini", tags=['joke'])
# ... similar for llm_2 with ['poem']

async for msg, metadata in graph.astream({"topic": "cats"}, stream_mode="messages"):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="|", flush=True)

Filter by Node
Filter by metadata["langgraph_node"]:
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    if msg.content and metadata["langgraph_node"] == "some_node_name":
        # Process

Stream Custom Data
Use get_stream_writer() to emit custom data; set stream_mode="custom":
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    writer = get_stream_writer()
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

for chunk in graph.stream({"query": "example"}, stream_mode="custom"):
    print(chunk)

Use with Any LLM
Stream from non-LangChain LLMs using custom:
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    writer = get_stream_writer()
    for chunk in your_custom_streaming_client(state["topic"]):
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

# ... compile and stream with "custom"

Disable Streaming for Specific Models
Set disable_streaming=True:
model = init_chat_model("anthropic:claude-3-7-sonnet-latest", disable_streaming=True)

Async with Python < 3.11
Explicitly pass RunnableConfig to async calls; pass writer manually instead of get_stream_writer().
Conclusion
LangGraph's streaming enhances responsiveness, supporting various modes for state, tokens, and custom data, with flexible filtering and integration options.