now based on the above context, I wanted you to create supervisor of supervisor workflow (render mermaid png)
add agents like: database-supervisor (agents: neo4j, tools: [get_schema, execute_query]), coding-supervisor (agents: python, tools: []), file-system-supervisor (agents: file_system_agent, tools: [file system related tools  with @tool decorator])

the workflow should be cyclic:

START --handoff_to--> master supervisor  --handoff_to--> [domain supervisors] ----> domain workers/agents --handoff_to-->  domain supervisor --handoff_to--> Master Supervisor --handoff_to--> END

Key consideration:
1. master agent = intelligently decision making and routing to domain supervisors
2. domain supervisor = ingelligently deciding which worker/agent should be used within that domain
3. worker/agents = they are responsible for generation, they have appropriate tools
4. memory management - manage state correctly, master supervisor should have proper memory
5. main.py should have cli support to run the system

create this implementation in python must use above context for this,
