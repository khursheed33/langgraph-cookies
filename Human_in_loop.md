LangGraph Human-in-the-Loop
This document summarizes the human-in-the-loop features in LangGraph, as detailed at LangGraph Add Human-in-the-Loop. It covers pausing with interrupts, resuming with Commands, common patterns, debugging, and considerations.
Alpha Notice: The documentation covers the v1-alpha release and is subject to change. For the latest stable version, refer to the current LangGraph Python or JavaScript documentation.
Overview
Use interrupts to pause graphs for human review, editing, or approval of tool calls in agents or workflows. Interrupts leverage LangGraph’s persistence layer to save state and pause indefinitely until resumption.
Pause Using interrupt
Dynamic interrupts trigger based on graph state by calling interrupt() function. The graph pauses for human input, then resumes.
As of v1.0, interrupt is recommended; NodeInterrupt is deprecated (removal in v2.0).
Steps:

Specify a checkpointer for state saving.
Call interrupt() appropriately (see patterns).
Run with thread ID until interrupt.
Resume via invoke/stream (see Command primitive).

Example:
from langgraph.types import interrupt, Command

def human_node(state: State):
    value = interrupt({"text_to_revise": state["some_text"]})
    return {"some_text": value}

graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "some_id"}}
result = graph.invoke({"some_text": "original text"}, config=config)
print(result['__interrupt__'])  # Interrupt object

print(graph.invoke(Command(resume="Edited text"), config=config))  # {'some_text': 'Edited text'}

Extended Example:
from typing import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    some_text: str

def human_node(state: State):
    value = interrupt({"text_to_revise": state["some_text"]})
    return {"some_text": value}

graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": uuid.uuid4()}}
result = graph.invoke({"some_text": "original text"}, config=config)

print(result['__interrupt__'])  # Interrupt details

print(graph.invoke(Command(resume="Edited text"), config=config))  # {'some_text': 'Edited text'}

Interrupts resemble Python’s input(), but rerun the node on resume. Place at node start or in dedicated nodes.
Resume Using the Command Primitive
Resume with Command via invoke/stream. Execution restarts from node beginning, with interrupt() returning the resume value.
Example:
graph.invoke(Command(resume={"age": "25"}), thread_config)

Resume Multiple Interrupts
For parallel nodes, resume all with Command(resume=resume_map) where map keys are interrupt IDs.
Example:
from typing import TypedDict
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    text_1: str
    text_2: str

def human_node_1(state: State):
    value = interrupt({"text_to_revise": state["text_1"]})
    return {"text_1": value}

def human_node_2(state: State):
    value = interrupt({"text_to_revise": state["text_2"]})
    return {"text_2": value}

graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
result = graph.invoke({"text_1": "original text 1", "text_2": "original text 2"}, config=config)

resume_map = {i.id: f"edited text for {i.value['text_to_revise']}" for i in graph.get_state(config).interrupts}
print(graph.invoke(Command(resume=resume_map), config=config))  # Edited texts

Common Patterns
Four patterns:

Approve/Reject: Pause before critical steps; route based on input.
Edit State: Review and update state.
Review Tool Calls: Edit LLM tool calls before execution.
Validate Input: Validate human input.

Approve or Reject
from typing import Literal
from langgraph.types import interrupt, Command

def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
    is_approved = interrupt({"question": "Is this correct?", "llm_output": state["llm_output"]})
    if is_approved:
        return Command(goto="some_node")
    else:
        return Command(goto="another_node")

# Add to graph
graph_builder.add_node("human_approval", human_approval)
graph = graph_builder.compile(checkpointer=checkpointer)

graph.invoke(Command(resume=True), config=thread_config)

(Extended examples truncated for brevity; refer to original for full details.)
Edit Graph State
Update state with Command(resume=..., update=...).
Review Tool Calls
Pause before tool execution to edit calls.
Validate Human Input
Loop until valid input.
Debug with Interrupts
Use static interrupts (interrupt_before/interrupt_after) for debugging.
Example:
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["step_3"])

# Run until interrupt
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)

# Resume
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)

Use Static Interrupts in LangGraph Studio
Set breakpoints in UI for inspection.
Considerations

Side-Effects: Place after interrupt to avoid duplication.
Subgraphs: Parent resumes from calling node; subgraph from interrupt node.
Multiple Interrupts in Node: Avoid dynamic changes to prevent mismatches.

Extended examples provided for side-effects, subgraphs, and multiple interrupts.
Conclusion
Human-in-the-loop enables oversight in LangGraph workflows via interrupts and Commands, supporting approval, editing, and validation with persistence for indefinite pauses.