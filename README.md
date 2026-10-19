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

## Groq model backend

This project uses `langchain_groq.ChatGroq` and reads `GROQ_API_KEY` from `.env`. The default model is `llama-3.3-70b-versatile`, which supports tool calls; set `MODEL` in `.env` to another Groq tool-capable model if preferred. The MCP server, user ACL, middleware, checkpoints and episodic/procedural memory remain unchanged.

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
