# LangGraph Features Overview

This README provides a concise summary of key LangGraph features: Persistence, Durable Execution, Streaming, Human-in-the-Loop, and Time Travel, based on the LangChain documentation. These features enable robust, interactive, and resilient AI workflows.

**Alpha Notice**: The documentation covers the v1-alpha release and is subject to change. For the latest stable version, refer to the current LangGraph Python or JavaScript documentation.

## 1. Persistence

LangGraph's persistence layer uses **checkpointers** to save graph state checkpoints per super-step in a `thread`, enabling human-in-the-loop, memory, time travel, and fault tolerance.

- **Threads**: Unique IDs (`thread_id`) track checkpoint sequences. Specify in `configurable`:
  ```python
  {"configurable": {"thread_id": "1"}}
  ```
- **Checkpoints**: `StateSnapshot` objects capture state (`config`, `metadata`, `values`, `next`, `tasks`). Example:
  ```python
  from langgraph.graph import StateGraph, START, END
  from langgraph.checkpoint.memory import InMemorySaver
  from typing import Annotated
  from typing_extensions import TypedDict
  from operator import add

  class State(TypedDict):
      foo: str
      bar: Annotated[list[str], add]

  def node_a(state: State):
      return {"foo": "a", "bar": ["a"]}

  def node_b(state: State):
      return {"foo": "b", "bar": ["b"]}

  workflow = StateGraph(State).add_node(node_a).add_node(node_b)
  workflow.add_edge(START, "node_a").add_edge("node_a", "node_b").add_edge("node_b", END)
  checkpointer = InMemorySaver()
  graph = workflow.compile(checkpointer=checkpointer)
  config = {"configurable": {"thread_id": "1"}}
  graph.invoke({"foo": ""}, config)
  ```
- **Memory Store**: Shares data across threads using `InMemoryStore` or semantic search.
- **Checkpointer Libraries**: Includes `langgraph-checkpoint` (in-memory), `langgraph-checkpoint-sqlite`, and `langgraph-checkpoint-postgres`. Supports encryption via `EncryptedSerializer`.

**Capabilities**:
- **Human-in-the-Loop**: Inspect, interrupt, approve steps.
- **Memory**: Retain conversation history.
- **Time Travel**: Replay or fork executions.
- **Fault Tolerance**: Resume from last successful step.

## 2. Durable Execution

Durable execution saves progress for pausing and resuming workflows, ideal for human-in-the-loop or long-running tasks.

- **Requirements**:
  - Use a checkpointer.
  - Specify `thread_id`.
  - Wrap non-deterministic operations/side effects in tasks.
- **Determinism**: Ensure idempotent operations and encapsulate side effects for consistent replay.
- **Durability Modes** (v0.6.0+):
  - `exit`: Persist on completion (fastest, no mid-execution recovery).
  - `async`: Persist asynchronously (balanced).
  - `sync`: Persist synchronously (most durable, slower).
  ```python
  graph.stream({"input": "test"}, durability="sync")
  ```
- **Tasks in Nodes**: Convert operations to tasks for easier management.

**Resumption**:
- Pause/resume with interrupts.
- Recover from failures using last checkpoint.

## 3. Streaming

Streaming provides real-time updates, enhancing UX for LLM-based applications.

- **Stream Modes**:
  - `values`: Full state after each step.
  - `updates`: State deltas.
  - `custom`: User-defined data.
  - `messages`: LLM tokens with metadata.
  - `debug`: Detailed traces.
  ```python
  for chunk in graph.stream(inputs, stream_mode="updates"):
      print(chunk)
  ```
- **Features**:
  - Stream subgraph outputs with `subgraphs=True`.
  - Stream LLM tokens or custom data using `get_stream_writer()`.
  - Filter by LLM tags or node names.

**Considerations**: For Python < 3.11, pass `RunnableConfig` explicitly for async; disable streaming for non-supported models.

## 4. Human-in-the-Loop

Interrupts pause graphs for human intervention, using persistence for indefinite pauses.

- **Pause with `interrupt`**:
  ```python
  from langgraph.types import interrupt

  def human_node(state: State):
      value = interrupt({"text_to_revise": state["some_text"]})
      return {"some_text": value}
  ```
- **Resume with `Command`**:
  ```python
  graph.invoke(Command(resume="Edited text"), config)
  ```
- **Patterns**:
  - Approve/reject actions.
  - Edit state.
  - Review tool calls.
  - Validate input.
- **Debugging**: Use static interrupts (`interrupt_before`/`interrupt_after`) or LangGraph Studio.

**Considerations**:
- Place side effects after interrupts.
- Subgraphs resume from parent node; avoid dynamic interrupt changes.

## 5. Time Travel

Time travel allows rewinding to explore alternative outcomes or fix mistakes.

- **Rewind**: Use `get_state_history` to fetch checkpoints.
  ```python
  for state in graph.get_state_history(config):
      print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
  ```
- **Resume**: Load state with `checkpoint_id`:
  ```python
  graph.stream(None, to_replay.config, stream_mode="values")
  ```
- **Use Case**: Debug, experiment, or build interactive applications like autonomous agents.

## Conclusion

LangGraph's features—Persistence, Durable Execution, Streaming, Human-in-the-Loop, and Time Travel—enable robust, interactive, and resilient AI workflows. Persistence supports state management, durable execution ensures reliability, streaming enhances responsiveness, human-in-the-loop allows oversight, and time travel facilitates debugging and exploration.
