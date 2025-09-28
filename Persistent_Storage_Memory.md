LangChain Persistence
This document summarizes the persistence mechanisms in LangGraph, as detailed at LangChain Persistence. It covers the built-in persistence layer, implemented through checkpointers, and the memory store for cross-thread information sharing. Key concepts include threads, checkpoints, state management, memory store, and checkpointer libraries, along with their capabilities such as human-in-the-loop workflows, memory, time travel, and fault tolerance.
Alpha Notice: The documentation covers the v1-alpha release and is subject to change. For the latest stable version, refer to the current LangGraph Python or JavaScript documentation.
Overview
LangGraph's persistence layer uses checkpointers to save a checkpoint of the graph state at each super-step, stored in a thread. This enables capabilities like human-in-the-loop workflows, memory retention, time travel, and fault tolerance. The LangGraph API handles checkpointing automatically, so manual configuration is not required.
Threads
A thread is a unique identifier (thread_id) assigned to a sequence of checkpoints, representing the accumulated state of a graph's execution. To persist state, a thread_id must be specified in the configurable portion of the config when invoking a graph:
{"configurable": {"thread_id": "1"}}

Threads allow retrieval of current and historical states. The LangGraph Platform API provides endpoints for managing threads and their states.
Checkpoints
A checkpoint is a snapshot of the graph state at a specific point, represented by a StateSnapshot object with the following properties:

config: Configuration associated with the checkpoint.
metadata: Metadata for the checkpoint.
values: State channel values at the checkpoint.
next: Tuple of node names to execute next.
tasks: Tuple of PregelTask objects with information about upcoming tasks, including error details or interrupts if applicable.

Example: Checkpoint Creation
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
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

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)

This creates four checkpoints:

Empty checkpoint with START as the next node.
Checkpoint with user input {'foo': '', 'bar': []} and node_a as the next node.
Checkpoint with node_a outputs {'foo': 'a', 'bar': ['a']} and node_b as the next node.
Checkpoint with node_b outputs {'foo': 'b', 'bar': ['a', 'b']} and no next nodes.

Get State
Retrieve the latest state or a specific checkpoint using graph.get_state(config):
config = {"configurable": {"thread_id": "1"}}
state = graph.get_state(config)  # Latest state

config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
state = graph.get_state(config)  # Specific checkpoint

Example output:
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
    tasks=()
)

Get State History
Retrieve the full history of checkpoints for a thread using graph.get_state_history(config):
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))

This returns a list of StateSnapshot objects, ordered chronologically with the most recent first.
Replay
Replay a prior execution up to a specific checkpoint using checkpoint_id:
config = {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
graph.invoke(None, config=config)

Steps before the checkpoint_id are replayed, while steps after are executed anew, creating a new fork.
Update State
Update the graph state using graph.update_state(config, values, as_node):

config: Specifies thread_id and optionally checkpoint_id to fork a specific checkpoint.
values: Values to update, processed by reducer functions if defined.
as_node: Optional node name to simulate the update coming from a specific node.

Example:
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# Current state: {"foo": 1, "bar": ["a"]}
graph.update_state(config, {"foo": 2, "bar": ["b"]})
# New state: {"foo": 2, "bar": ["a", "b"]}

The foo channel is overwritten (no reducer), while bar is appended (due to the add reducer).
Memory Store
The memory store enables sharing information across threads, unlike checkpointers, which are thread-specific. The LangGraph API handles stores automatically.
Basic Usage
from langgraph.store.memory import InMemoryStore
import uuid

in_memory_store = InMemoryStore()
user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()

Output:
{
    'value': {'food_preference': 'I like pizza'},
    'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
    'namespace': ['1', 'memories'],
    'created_at': '2024-10-02T17:22:31.590602+00:00',
    'updated_at': '2024-10-02T17:22:31.590605+00:00'
}

Semantic Search
Enable semantic search with an embedding model:
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"]
    }
)

memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3
)

Control embedding fields:
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"food_preference": "I love Italian cuisine", "context": "Discussing dinner plans"},
    index=["food_preference"]
)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)

Using in LangGraph
Compile the graph with both a checkpointer and a memory store:
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)

user_id = "1"
config = {"configurable": {"thread_id": "1", "user_id": user_id}}
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
):
    print(update)

Access the store in a node:
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"memory": "User mentioned pizza"})

Search memories in a node:
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    memories = store.search(namespace, query=state["messages"][-1].content, limit=3)
    info = "\n".join([d.value["memory"] for d in memories])
    # Use memories in model call

Memories persist across threads with the same user_id:
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]}, config, stream_mode="updates"
):
    print(update)

For LangGraph Platform, configure indexing in langgraph.json:
{
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}

Checkpointer Libraries
LangGraph provides several checkpointer implementations:

langgraph-checkpoint: Base interface (BaseCheckpointSaver) and InMemorySaver for experimentation.
langgraph-checkpoint-sqlite: Uses SQLite (SqliteSaver/AsyncSqliteSaver) for local workflows.
langgraph-checkpoint-postgres: Uses Postgres (PostgresSaver/AsyncPostgresSaver) for production.

Checkpointer Interface
Methods include:

.put: Store a checkpoint.
.put_writes: Store pending writes.
.get_tuple: Fetch a checkpoint tuple for graph.get_state().
.list: List checkpoints for graph.get_state_history().

Asynchronous methods (aput, aput_writes, aget_tuple, alist) support async execution.
Serializer
The JsonPlusSerializer handles serialization, with a pickle_fallback option for unsupported types:
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
graph.compile(checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True)))

Encryption
Enable encryption with EncryptedSerializer:
import sqlite3
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)

For Postgres:
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()

Encryption is automatic on LangGraph Platform if LANGGRAPH_AES_KEY is set.
Capabilities
Human-in-the-Loop
Checkpointers enable inspection, interruption, and approval of graph steps by saving and resuming state.
Memory
Checkpointers retain conversation history within a thread, accessible in subsequent interactions.
Time Travel
Replay or fork executions at specific checkpoints to review or explore alternative paths.
Fault Tolerance
Resume from the last successful step if a node fails, with pending writes stored to avoid re-running successful nodes.
Conclusion
LangGraph's persistence layer, powered by checkpointers and memory stores, provides robust state management for threads and cross-thread information sharing. It supports advanced workflows through human-in-the-loop interactions, memory retention, time travel, and fault tolerance, with automatic handling via the LangGraph API.