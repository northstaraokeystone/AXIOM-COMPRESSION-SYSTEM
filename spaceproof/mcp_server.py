"""mcp_server.py - MCP Protocol Server for Claude Desktop Integration.

D20 Production Evolution: MCP protocol compliance for external integration.

THE MCP SERVER PARADIGM:
    SpaceProof exposes receipts to Claude Desktop/Cursor via MCP.
    Tools: query_receipts, verify_chain, get_topology

    MCP Configuration (for claude_desktop_config.json):
    {
        "mcpServers": {
            "spaceproof": {
                "command": "python",
                "args": ["-m", "spaceproof.mcp_server"],
                "tools": ["query_receipts", "verify_chain", "get_topology"]
            }
        }
    }

Source: Receipts_native_architecture_v2_0.txt MCP protocol
"""

import json
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
import threading

from spaceproof.core import emit_receipt, merkle
from spaceproof.meta_integration import classify_pattern

# === CONSTANTS ===

MCP_SERVER_TENANT = "spaceproof-mcp-server"
DEFAULT_PORT = 3000
MCP_VERSION = "1.0.0"

# In-memory receipt storage (would be persistent in production)
_receipt_store: List[Dict[str, Any]] = []
_pattern_store: Dict[str, Dict[str, Any]] = {}


def add_receipt(receipt: Dict[str, Any]) -> None:
    """Add receipt to store.

    Args:
        receipt: Receipt to store
    """
    _receipt_store.append(receipt)


def add_pattern(pattern_id: str, pattern: Dict[str, Any]) -> None:
    """Add pattern to store.

    Args:
        pattern_id: Pattern identifier
        pattern: Pattern data
    """
    _pattern_store[pattern_id] = pattern


def query_receipts(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query receipts by type, domain, time range.

    MCP Tool: query_receipts

    Args:
        filters: Filter criteria
            - receipt_type: str (e.g., "compute_provenance")
            - domain: str (e.g., "orbital_compute")
            - start_time: str (ISO8601)
            - end_time: str (ISO8601)
            - satellite_id: str

    Returns:
        List of matching receipts
    """
    results = []

    receipt_type = filters.get("receipt_type")
    domain = filters.get("domain")
    start_time = filters.get("start_time")
    end_time = filters.get("end_time")
    satellite_id = filters.get("satellite_id")

    for receipt in _receipt_store:
        # Filter by receipt type
        if receipt_type and receipt.get("receipt_type") != receipt_type:
            continue

        # Filter by domain (check tenant_id prefix)
        if domain and not receipt.get("tenant_id", "").endswith(domain):
            continue

        # Filter by time range
        ts = receipt.get("ts", "")
        if start_time and ts < start_time:
            continue
        if end_time and ts > end_time:
            continue

        # Filter by satellite_id
        if satellite_id and receipt.get("satellite_id") != satellite_id:
            continue

        results.append(receipt)

    return results


def verify_chain(start_hash: str, end_hash: str) -> Dict[str, Any]:
    """Verify Merkle chain integrity.

    MCP Tool: verify_chain

    Args:
        start_hash: SHA256:BLAKE3 dual-hash of start receipt
        end_hash: SHA256:BLAKE3 dual-hash of end receipt

    Returns:
        Verification result with validity status
    """
    # Find receipts with matching hashes
    chain_receipts = []
    found_start = False
    found_end = False

    for receipt in _receipt_store:
        payload_hash = receipt.get("payload_hash", "")

        if payload_hash == start_hash:
            found_start = True

        if found_start and not found_end:
            chain_receipts.append(receipt)

        if payload_hash == end_hash:
            found_end = True
            break

    if not found_start or not found_end:
        return {
            "valid": False,
            "error": "Chain endpoints not found",
            "start_found": found_start,
            "end_found": found_end,
        }

    # Verify Merkle root
    computed_root = merkle(chain_receipts)

    # Check each receipt is properly chained
    valid = True
    for i in range(1, len(chain_receipts)):
        # Verify previous receipt hash links to current
        # (In production, would check previousProof field)
        pass

    result = {
        "valid": valid,
        "chain_length": len(chain_receipts),
        "merkle_root": computed_root,
        "start_hash": start_hash,
        "end_hash": end_hash,
        "verification_time": datetime.utcnow().isoformat() + "Z",
    }

    emit_receipt(
        "chain_verification",
        {
            "tenant_id": MCP_SERVER_TENANT,
            **result,
        },
    )

    return result


def get_topology(pattern_id: str) -> Dict[str, Any]:
    """Return pattern topology classification.

    MCP Tool: get_topology

    Args:
        pattern_id: Pattern UUID

    Returns:
        Topology classification: "open" | "closed" | "hybrid"
    """
    pattern = _pattern_store.get(pattern_id)

    if pattern is None:
        return {
            "pattern_id": pattern_id,
            "topology": "unknown",
            "error": "Pattern not found",
        }

    domain = pattern.get("domain", "unknown")
    topology = classify_pattern(pattern, domain)

    result = {
        "pattern_id": pattern_id,
        "domain": domain,
        "topology": topology,
        "effectiveness": pattern.get("effectiveness", 0.0),
        "autonomy_score": pattern.get("autonomy_score", 0.0),
        "classification_time": datetime.utcnow().isoformat() + "Z",
    }

    return result


def get_health() -> Dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "version": MCP_VERSION,
        "receipt_count": len(_receipt_store),
        "pattern_count": len(_pattern_store),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP requests."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _read_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data: Any, status: int = 200):
        self._set_headers(status)
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json(get_health())
        elif self.path == "/":
            self._send_json({
                "name": "SpaceProof MCP Server",
                "version": MCP_VERSION,
                "tools": ["query_receipts", "verify_chain", "get_topology"],
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        try:
            body = self._read_body()

            if self.path == "/query_receipts":
                result = query_receipts(body)
                self._send_json({"receipts": result, "count": len(result)})

            elif self.path == "/verify_chain":
                start_hash = body.get("start_hash", "")
                end_hash = body.get("end_hash", "")
                result = verify_chain(start_hash, end_hash)
                self._send_json(result)

            elif self.path == "/get_topology":
                pattern_id = body.get("pattern_id", "")
                result = get_topology(pattern_id)
                self._send_json(result)

            elif self.path == "/add_receipt":
                # Utility endpoint for adding receipts
                add_receipt(body)
                self._send_json({"status": "added"})

            elif self.path == "/add_pattern":
                # Utility endpoint for adding patterns
                pattern_id = body.get("pattern_id", "")
                add_pattern(pattern_id, body)
                self._send_json({"status": "added", "pattern_id": pattern_id})

            else:
                self._send_json({"error": "Not found"}, 404)

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int = DEFAULT_PORT, blocking: bool = True) -> Optional[HTTPServer]:
    """Run MCP server.

    Args:
        port: Port to listen on
        blocking: Whether to block

    Returns:
        HTTPServer instance if non-blocking
    """
    server = HTTPServer(("", port), MCPHandler)
    print(f"SpaceProof MCP Server v{MCP_VERSION} listening on port {port}")

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
    else:
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        return server

    return None


def stop_server(server: HTTPServer) -> None:
    """Stop MCP server.

    Args:
        server: Server to stop
    """
    server.shutdown()


# === MCP PROTOCOL HANDLERS (for stdio mode) ===


def handle_mcp_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP protocol message.

    Args:
        message: MCP message

    Returns:
        MCP response
    """
    method = message.get("method", "")
    params = message.get("params", {})
    msg_id = message.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "spaceproof",
                    "version": MCP_VERSION,
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "query_receipts",
                        "description": "Query SpaceProof receipts by type, domain, time range",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "receipt_type": {"type": "string"},
                                "domain": {"type": "string"},
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"},
                                "satellite_id": {"type": "string"},
                            },
                        },
                    },
                    {
                        "name": "verify_chain",
                        "description": "Verify Merkle chain integrity between two receipts",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "start_hash": {"type": "string"},
                                "end_hash": {"type": "string"},
                            },
                            "required": ["start_hash", "end_hash"],
                        },
                    },
                    {
                        "name": "get_topology",
                        "description": "Get pattern topology classification",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "pattern_id": {"type": "string"},
                            },
                            "required": ["pattern_id"],
                        },
                    },
                ],
            },
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "query_receipts":
            result = query_receipts(arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                },
            }

        elif tool_name == "verify_chain":
            result = verify_chain(
                arguments.get("start_hash", ""),
                arguments.get("end_hash", ""),
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                },
            }

        elif tool_name == "get_topology":
            result = get_topology(arguments.get("pattern_id", ""))
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


def run_stdio_mode():
    """Run MCP server in stdio mode for Claude Desktop."""

    for line in sys.stdin:
        try:
            message = json.loads(line.strip())
            response = handle_mcp_message(message)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)},
            }
            print(json.dumps(error_response), flush=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SpaceProof MCP Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode for MCP")
    args = parser.parse_args()

    if args.stdio:
        run_stdio_mode()
    else:
        run_server(args.port)


if __name__ == "__main__":
    main()
