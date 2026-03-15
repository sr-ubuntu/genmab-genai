# Simple MCP Architecture - Demo View

```mermaid
graph LR
    User["👤 User<br/>Query"]

    API["🌐 FastAPI<br/>Server"]

    Claude["🤖 Claude<br/>Orchestrator"]

    subgraph Tools["🛠️ MCP Tools"]
        DBTool["Database<br/>get_patients<br/>get_records"]
        SumTool["Summarizer<br/>summarize"]
    end

    Response["📊 Response"]

    User -->|"1. Ask question"| API
    API -->|"2. Send query"| Claude
    Claude -->|"3. Use tools"| Tools
    Tools -->|"4. Return data"| Claude
    Claude -->|"5. Answer"| Response
    Response -->|"Show result"| User
```

## Flow 

1. **User asks a question** via REST API
2. **Claude receives the query** and sees what tools are available
3. **Claude decides which tools to use** (database or summarizer)
4. **Tools execute and return results**
5. **Claude synthesizes the answer** and returns it to the user

That's it! MCP (Model Context Protocol) is just a way for Claude to safely call external tools.
