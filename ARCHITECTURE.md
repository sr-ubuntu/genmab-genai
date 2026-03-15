# MCP Architecture Diagram

```mermaid
graph TB
    User["User/Client<br/>(REST API)"]

    subgraph FastAPI["FastAPI Server<br/>(api_server.py)"]
        API["POST /query<br/>Endpoint"]
        Agent["Agent Loop<br/>(_run_agent)"]
    end

    subgraph MCPClient["MCP Client Layer"]
        DBClient["DB MCP Client<br/>(stdio_client)"]
        SummarizerClient["Summarizer MCP Client<br/>(stdio_client)"]
    end

    subgraph MCPServers["MCP Servers<br/>(Subprocesses)"]
        DBServer["DB Server<br/>(mcp_db_server.py)<br/>- get_patients<br/>- get_records"]
        SummarizerServer["Summarizer Server<br/>(mcp_summarizer_server.py)<br/>- summarize_clinical"]
    end

    subgraph Claude["Anthropic SDK"]
        ClaudeModel["Claude 3.5 Sonnet<br/>messages.create()"]
    end

    subgraph Data["Data Sources"]
        DB["SQLite Database<br/>(clinical.db)"]
        AWS["AWS API Gateway<br/>→ Bedrock<br/>(optional)"]
    end

    User -->|"1. POST query"| API
    API -->|"2. Call"| Agent

    Agent -->|"3a. Connect"| DBClient
    Agent -->|"3b. Connect"| SummarizerClient

    DBClient -->|"4a. stdio"| DBServer
    SummarizerClient -->|"4b. stdio"| SummarizerServer

    Agent -->|"5. Send message<br/>+ tools list"| ClaudeModel
    ClaudeModel -->|"6. Tool use request"| Agent

    Agent -->|"7. Call tool"| DBClient
    Agent -->|"7. Call tool"| SummarizerClient

    DBServer -->|"8. Query"| DB
    SummarizerServer -->|"8a. Call API"| AWS

    Agent -->|"9. Tool result"| ClaudeModel
    ClaudeModel -->|"10. Final response"| Agent
    Agent -->|"11. JSON response"| User
```

## Data Flow

1. **User Request** → FastAPI receives query via REST
2. **Agent Initialization** → Creates MCP client connections to both servers
3. **Tool Discovery** → Gets list of available tools from each MCP server
4. **Claude Loop** → Sends query + tools to Claude API
5. **Tool Calls** → Claude requests tool execution
6. **Tool Execution** → Agent calls appropriate MCP server
7. **Results** → MCP servers return data (from DB or Bedrock)
8. **Response Loop** → Results sent back to Claude
9. **Final Answer** → Claude returns text response to user

## Key Components

- **api_server.py**: FastAPI server, orchestrates MCP connections
- **mcp_db_server.py**: Exposes database tools via MCP
- **mcp_summarizer_server.py**: Exposes summarization tools via MCP
- **stdio transport**: Lightweight communication between processes
- **ClientSession**: Manages MCP protocol handshake & messaging
