LangChain Workflows and Agents Summary
This document summarizes the key concepts from the LangChain documentation on workflows and agents, as detailed at LangChain Workflows and Agents. It covers the setup, patterns, and examples for building workflows and agents using LangGraph, including prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer, and agent configurations.
Overview

Workflows: Predetermined code paths designed to execute in a specific order, ideal for structured tasks.
Agents: Dynamic systems that define their own processes and tool usage, suitable for unpredictable scenarios.
LangGraph Benefits: Offers persistence, streaming, debugging support, and deployment capabilities.

Setup
To build workflows or agents, use a chat model supporting structured outputs and tool calling (e.g., Anthropic's Claude).
Installation
pip install langchain_core langchain-anthropic langgraph

Initialize LLM
import os
import getpass
from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

LLMs and Augmentations
Workflows and agents rely on LLMs augmented with tools, structured outputs, or memory. Examples include:
Structured Output
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query optimized for web search.")
    justification: str = Field(None, description="Why this query is relevant.")

structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

Tool Augmentation
def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])
msg = llm_with_tools.invoke("What is 2 times 3?")
msg.tool_calls

Workflow Patterns
1. Prompt Chaining
Processes outputs sequentially through multiple LLM calls for well-defined tasks (e.g., translation, content verification).
Example: Joke Generation Workflow
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: State):
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: State):
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()
display(Image(chain.get_graph().draw_mermaid_png()))
state = chain.invoke({"topic": "cats"})
print("Initial joke:", state["joke"])
if "improved_joke" in state:
    print("Improved joke:", state["improved_joke"])
    print("Final joke:", state["final_joke"])
else:
    print("Joke failed quality gate - no punchline detected!")

2. Parallelization
Runs multiple LLM tasks simultaneously to increase speed or verify outputs.
Example: Parallel Content Generation
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_1(state: State):
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}

parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])

3. Routing
Directs inputs to context-specific tasks based on initial processing.
Example: Content Type Router
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(None, description="The next step in the routing process")

router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_1(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    decision = router.invoke([
        SystemMessage(content="Route the input to story, joke, or poem based on the user's request."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

router_builder = StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges("llm_call_router", route_decision, {
    "llm_call_1": "llm_call_1",
    "llm_call_2": "llm_call_2",
    "llm_call_3": "llm_call_3",
})
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()
display(Image(router_workflow.get_graph().draw_mermaid_png()))
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])

4. Orchestrator-Worker
Orchestrator breaks tasks into subtasks, delegates to workers, and synthesizes results. Ideal for dynamic tasks with unpredictable subtasks.
Example: Report Generation
from typing import Annotated, List
import operator
from langgraph.types import Send

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics and concepts.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {state['topic']}"),
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    section = llm.invoke([
        SystemMessage(content="Write a report section following the provided name and description. Include no preamble. Use markdown formatting."),
        HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"),
    ])
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()
display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
from IPython.display import Markdown
Markdown(state["final_report"])

5. Evaluator-Optimizer
One LLM generates a response, another evaluates it, and iterations occur until the response meets criteria.
Example: Joke Optimization
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
    feedback: str = Field(description="If the joke is not funny, provide feedback on how to improve it.")

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state: State):
    if state.get("feedback"):
        msg = llm.invoke(f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges("llm_call_evaluator", route_joke, {
    "Accepted": END,
    "Rejected + Feedback": "llm_call_generator",
})

optimizer_workflow = optimizer_builder.compile()
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])

6. Agents
Agents use LLMs with tools in a feedback loop, autonomously deciding actions and tools for unpredictable tasks.
Example: Arithmetic Agent
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def llm_call(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke([
                SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
            ] + state["messages"])
        ]
    }

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

Conclusion
LangChain and LangGraph provide flexible frameworks for building workflows and agents. Workflows (prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer) are suited for structured tasks, while agents excel in dynamic, tool-driven scenarios. The examples demonstrate practical implementations for various use cases, leveraging structured outputs, tool calling, and graph-based workflows.