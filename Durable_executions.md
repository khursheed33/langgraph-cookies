LangGraph Durable Execution
This document summarizes the durable execution features in LangGraph, as detailed at LangGraph Durable Execution. It covers the concept of durable execution, requirements, determinism, durability modes, using tasks in nodes, resuming workflows, and starting points for resumption.
Alpha Notice: The documentation covers the v1-alpha release and is subject to change. For the latest stable version, refer to the current LangGraph Python or JavaScript documentation.
Overview
Durable execution allows a process or workflow to save progress at key points, enabling it to pause and resume from where it left off. This is useful for human-in-the-loop scenarios, long-running tasks, or handling interruptions like LLM timeouts. LangGraph's persistence layer ensures state is saved, allowing resumption without reprocessing completed steps, even after delays.
If using LangGraph with a checkpointer, durable execution is enabled by default. Design workflows to be deterministic and idempotent, wrapping side effects or non-deterministic operations in tasks.
Requirements
To enable durable execution:

Specify a checkpointer to save progress.
Provide a thread identifier for tracking execution history.
Wrap non-deterministic operations or side effects in tasks or nodes for consistent replay.

For details on determinism, see the section below.
Determinism and Consistent Replay
Resumption starts from an appropriate point, replaying steps from there. Wrap non-deterministic operations (e.g., random generation) and side effects (e.g., API calls) in tasks or nodes to avoid repetition.
Guidelines:

Avoid Repeating Work: Wrap each side-effect operation in separate tasks.
Encapsulate Non-Deterministic Operations: Ensure consistent outcomes on resumption.
Use Idempotent Operations: Make side effects repeatable without duplication, using keys or verification.

See common pitfalls in the functional API for examples; principles apply to StateGraph.
Durability Modes
LangGraph supports three modes for balancing performance and consistency (added in v0.6.0; use durability instead of deprecated checkpoint_during):

"exit": Persist only on completion (best performance, no mid-execution recovery).
"async": Persist asynchronously (good performance/durability, minor crash risk).
"sync": Persist synchronously (high durability, performance overhead).

Specify mode in execution methods:
graph.stream({"input": "test"}, durability="sync")

Using Tasks in Nodes
Convert operations to tasks for easier management:
from typing import NotRequired
from typing_extensions import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
import requests

class State(TypedDict):
    url: str
    result: NotRequired[str]

def call_api(state: State):
    result = requests.get(state['url']).text[:100]
    return {"result": result}

builder = StateGraph(State)
builder.add_node("call_api", call_api)
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
graph.invoke({"url": "https://www.example.com"}, config)

Resuming Workflows
Resume for:

Pausing/Resuming: Use interrupts and Commands for human intervention.
Failure Recovery: Resume from last checkpoint with None input.

Starting points:

StateGraph: Beginning of the interrupted node.
Subgraph Call: Parent node calling the subgraph; within subgraph, the interrupted node.

Conclusion
LangGraph's durable execution, powered by checkpointers, ensures resilient workflows. By following determinism guidelines and using appropriate durability modes, applications can handle interruptions, human interventions, and failures effectively.