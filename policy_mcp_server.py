"""Local stdio MCP policy server. User identity is bound by parent process, not model arguments."""
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent / "policies"
# PRINCIPAL == "alice" when we run python app.py --user alice --thread session1
PRINCIPAL = os.environ.get("DEMO_PRINCIPAL", "")
ACL = {
    "alice": {"refund": "alice_refund.md", "security": "alice_security.md"},
    "bob": {"refund": "bob_refund.md"},
}
if PRINCIPAL not in ACL:
    raise RuntimeError("No valid DEMO_PRINCIPAL bound to MCP server")

mcp = FastMCP("policy-files")

@mcp.tool()
def list_policy_files() -> list[str]:
    """List policy IDs visible to the authenticated local principal."""
    return sorted(ACL[PRINCIPAL])

@mcp.tool()
def read_policy(file_id: str) -> str:
    """Read an allowed policy by logical file_id (e.g. refund, security)."""
    filename = ACL[PRINCIPAL].get(file_id)
    if filename is None:
        return "ACCESS_DENIED: policy ID not authorized for this principal"
    # No model-supplied paths; resolve only from fixed allowlist mapping.
    return (ROOT / filename).read_text(encoding="utf-8")

# local MCP
if __name__ == "__main__":
    mcp.run(transport="stdio")
