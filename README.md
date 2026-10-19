# LangGraph durable procedural + episodic memory, guarded MCP file access (CLI)

## Setup

Requires Python 3.11+ and a valid Groq API key.

```bash
cd langgraph_memory_mcp_demo
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # on Windows: copy .env.example .env
# edit .env with your API key
python app.py --user alice --thread session1
```

Try:
- `List my policy files.`
- `What's my refund window? Read my refund policy.`
- `Do refunds above $500 need approval?`
- `/quit`; restart same command and ask `What did I ask earlier?`
- Run `python app.py --user bob --thread session1` and ask for Bob's refund policy. Bob cannot access Alice's security file.

## Architecture

```
CLI (principal selected locally; demo only)
  -> recall -> agent (procedural skill + user-scoped episodic recall)
                  | tool calls
                  v
              middleware (allowlist, validation, quota, audit events)
                  | mcp stdio
                  v
              policy_mcp_server.py (PRINCIPAL-bound allowlist -> files)
                  | tool results
                  +----> agent -> memorize -> END
```

### Startup & wiring

```

 $ python agent.py --user alice --thread main
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ argparse: --user {alice,bob}   --thread (default "main")│
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 load_dotenv(ROOT/.env)  →  require GROQ_API_KEY
        │
        ▼
 init_db()  ── creates:
   • skills(user_id, name, instructions)   → seeded "policy_reading" for alice & bob
   • episodes(id, user_id, thread_id, ts, user_request, answer, outcome, policy_ids)
   • tool_events(id, user_id, thread_id, ts, tool_name, args, status, detail)
        │
        ▼
 MultiServerMCPClient({ "policy": {
     transport: "stdio",
     command: python,
     args: ["policy_mcp_server.py"],
     env: { DEMO_PRINCIPAL = <user> }        ◄── model CANNOT pick principal
 }})
        │  exposes tools:
        │    • list_policy_files()
        │    • read_policy(file_id)
        ▼
 ChatGroq(model=MODEL)   +   AsyncSqliteSaver(checkpoints.sqlite3)
        │
        ▼
 build_graph(model, tools, principal, thread_id, saver)
        │
        ▼
 thread_id = f"{user}:{thread}"       ◄── per-user, per-thread checkpoint bucket
        │
        ▼
 REPL: read "You>" → graph.ainvoke(...) → print answer → loop

```

### Graph topology (compiled StateGraph)

```

                            START
                              │
                              ▼
                       ┌────────────┐
                       │   recall   │   resets per-turn counters:
                       └────────────┘   tool_calls_used=0,
                              │         fetched_policy_ids=[],
                              │         outcome="pending"
                              ▼
                       ┌────────────┐
                       │   agent    │   • search_episodes(user, new_turn)
                       │            │   • load_skills(user)
                       │            │   • build SystemMessage
                       │            │   • bind_tools(model).ainvoke(last 12 msgs)
                       └────────────┘   (or unbound model if budget exhausted)
                              │
                              ▼
                     ┌──────────────────┐
                     │ route(state)     │
                     │ last.tool_calls? │
                     └──────────────────┘
                        │            │
                     yes│            │no
                        ▼            ▼
                 ┌────────────┐  ┌────────────┐
                 │ middleware │  │  memorize  │
                 └────────────┘  └────────────┘
                        │              │
                        │ append       │ save_episode()
                        │ ToolMessage  │ outcome = policy_consulted
                        │ (+ log_event)│   if fetched_policy_ids else
                        │              │ no_policy_consulted
                        └──► agent     ▼
                             (loop)    END

```

### One turn, end-to-end (with memory reads/writes)

```

 USER INPUT "..."                 (REPL)
      │
      ▼
 graph.ainvoke({
     messages          : [HumanMessage(q)]
     user_id           : "alice"
     new_turn          : q
     tool_calls_used   : 0
     fetched_policy_ids: []
     outcome           : "pending"
 }, config={ thread_id:"alice:main", recursion_limit:20 })
      │
      ├── AsyncSqliteSaver  ── loads prior messages for this thread  ──┐
      │                                                                │
      ▼                                                                │
 ┌─────────┐   reads skills + episodes (SQL, per-user filter)         │
 │ agent   │◄───────────────────────────────────────────────────────  │
 └─────────┘                                                          │
      │  AIMessage                                                       │
      ▼                                                                  │
 ┌──────────────┐  for each tool_call:                                │
 │ middleware   │  validate → invoke → log_event → ToolMessage         │
 └──────────────┘  writes to tool_events  ───────────────────────────►┘
      │
      ▼ (loop until no tool_calls or budget hit)
 ┌──────────────┐
 │  memorize    │  save_episode(...)  → episodes table
 └──────────────┘
      │
      ▼
 print result["messages"][-1].content
 print [outcome=... ; policies=[...]]
      │
      ▼
 AsyncSqliteSaver writes checkpoint for thread "alice:main"

```

#### Where durable state lives

```
                ┌───────────────────────────┐
   read  ◄──────│ skills      (procedural)  │  ← seeded, never mutated by model
                │ episodes    (episodic)    │  ← appended in memorize()
                │ tool_events (audit log)   │  ← appended in middleware()
                │ checkpoints (langgraph)   │  ← AutoSqliteSaver per turn
                └───────────────────────────┘
   write ──────► episodes, tool_events, checkpoints

```

### Middleware guard ladder (deterministic, per tool call)

```
 for call in last_ai_message.tool_calls:
   name, args, call_id = ...
   ┌───────────────────────────────────────────────────────────────┐
   │ 1. name ∈ {list_policy_files, read_policy}  AND in tool_map ? │
   └───────────────────────────────────────────────────────────────┘
             │no                                 │yes
             ▼                                   ▼
   status=blocked                   ┌───────────────────────────────┐
   "tool not allowlisted"           │ 2. tool_calls_used < 4 ?      │
                                    └───────────────────────────────┘
                                             │no              │yes
                                             ▼                ▼
                                   status=blocked    ┌──────────────────────────┐
                                   "budget exceeded" │ 3. arg validation:       │
                                                     │  read_policy: {file_id}  │
                                                     │    matching [a-z0-9_-]{1,40}
                                                     │  list_policy_files: {}   │
                                                     └──────────────────────────┘
                                                              │invalid   │valid
                                                              ▼          ▼
                                                     status=blocked   await tool.ainvoke(args)
                                                     "invalid args"        │
                                                                           ▼
                                                     ┌────────────────────────────────┐
                                                     │ "ACCESS_DENIED" in result?     │
                                                     │   yes → status=denied          │
                                                     │   no  → if read_policy:        │
                                                     │           fetched_policy_ids   │
                                                     │             .append(file_id)   │
                                                     │   truncate content to 8000ch   │
                                                     │ exception → status=error       │
                                                     └────────────────────────────────┘
                                                                           │
                                                                           ▼
                                                     log_event(user, thread, name, args,
                                                               status, detail)
                                                     → INSERT INTO tool_events
                                                                           │
                                                                           ▼
                                                     ToolMessage(content, tool_call_id, name)
                                                     → appended to state.messages
                                                     → back to agent node
```

  **LangGraph** = the control plane: recall → agent ↔ middleware → memorize.

  **agent** = LLM, may emit tool calls; sees skills + episodes + budget in the system prompt.

  **middleware** = the only place a tool ever runs; it's an allowlist + budget + arg-schema + audit wall — the model can't bypass it.

  **memorize** = writes the turn to episodes so the next turn's search_episodes can recall it.

  **AsyncSqliteSaver** = the conversation's durable message history, keyed by alice:main etc.

  **MCP server** = untrusted data source; the principal is fixed by DEMO_PRINCIPAL in the subprocess env, so the model can never choose who it reads policies for.

### Memory persistence
- `data/checkpoints.sqlite3`: LangGraph durable conversation checkpoints (user-prefixed thread IDs).
- `data/memory.sqlite3` / `skills`: durable procedural instructions/skills scoped per principal.
- `data/memory.sqlite3` / `episodes`: bounded transcripts (request + answer), policy IDs, and outcomes, scoped per principal and shared across that user's threads.
- `data/memory.sqlite3` / `tool_events`: auditable before/after decision for each tool call, including blocked/denied errors.

Memory retrieval is **bounded lexical matching** over most recent 40 episodes; substitute embeddings and a DB for production. Procedural skills are seeded once with `INSERT OR IGNORE` and survive restarts. Policy docs are NOT cached as authoritative episodic facts.

### Safety properties and limitations
- MCP policy tools receive only logical IDs, not arbitrary filesystem paths. Server identity is fixed by its process environment and cannot be passed by the model. `read_policy` separately enforces the ACL.
- Middleware validates allowlisted tools, argument shape, tool budget, and records audit events. The model is instructed to treat policy content as untrusted; this is **not** a complete prompt-injection defense.
- `--user alice|bob` is only for local demonstration. It is **not authentication**. Real deployments must bind a verified principal from trusted identity and enforce ACLs server-side.
- `user_id` in LangGraph state is illustrative, not trusted as an authorization source; closures capture the locally selected principal.
- `tool_events` may include arguments; avoid sensitive inputs in real audit logs.
- A denied read is surfaced to the model as `ACCESS_DENIED`; it never retrieves the unauthorized file contents.
- SQLite is appropriate for a local demonstration, not concurrent production. In production use Postgres checkpointers, real auth, encrypted storage, retention, explicit approvals for write actions, and stronger injection controls.
- **No refund-changing tool is provided**; assistant only reads and summarizes.
- Prompt messages are truncated to 12 latest for cost control; memory persists outside that prompt window. Long conversation references beyond 12 messages depend on episodic retrieval.

## Inspect state and audit trail

```bash
python - <<'PY'
import sqlite3
c=sqlite3.connect('data/memory.sqlite3')
for table in ('skills','episodes','tool_events'):
 print('\n',table)
 for row in c.execute('SELECT * FROM '+table+' LIMIT 10'): print(row)
PY
```

## Files
- `app.py`: state graph, persistence, memory retrieval, tool middleware, CLI.
- `policy_mcp_server.py`: local MCP stdio server with principal-bound file ACL.
- `policies/*.md`: sample per-user policy files.
- `requirements.txt`, `.env.example`: dependencies and config.

**Verification note:** source has been syntax-checked; API integration and end-to-end model calls need running in your configured Python environment with dependencies and an API key.

### Run the application & Test

```bash
# run for alice
python app.py --user alice --thread session1
# run for bob
python app.py --user bob --thread session1

You> List my available policies.

You> Read my refund policy.

You> What is the approval threshold?

You> What did we discuss earlier?
```

### Next Step

```text

Task
  ↓
Recall procedural skill
  ↓
Retrieve relevant episodes
  ↓
Execute tools
  ↓
Evaluate outcome
  ↓
Extract reusable experience
  ↓
Propose skill improvement
  ↓
Human approval
  ↓
Update procedural memory

```

### What should be improved for production


| Component | Current Implementation | Production Direction |
|---|---|---|
| **Checkpointing** | SQLite | PostgreSQL |
| **Procedural memory** | SQLite skills | Versioned, approved skill registry |
| **Episodic memory** | Lexical search | Hybrid/vector retrieval with metadata filters |
| **User identity** | CLI argument | Verified authentication |
| **MCP authorization** | Static ACL | Identity-bound RBAC/ABAC |
| **Middleware** | Graph node | Centralized policy enforcement and observability |
| **Human approval** | Documented approval rules | LangGraph `interrupt()` / resume |
| **Memory lifecycle** | Persistent records | Retention, summarization, deletion |
| **Auditability** | SQLite events | Structured traces and centralized audit store |
