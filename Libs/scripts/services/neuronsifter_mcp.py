"""MCP-shaped typed scientific tool servers.

The runtime keeps the transport boundary deliberately small: callers receive
typed receipts from :class:`ScientificToolFabric`, while deployment-specific
MCP transports can wrap the same server objects without exposing a shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from Libs.models.modules.scientific_tool_fabric import (
    TRUST_BOUNDARIES,
    ScientificToolFabric,
    ScientificToolReceipt,
)


@dataclass(frozen=True)
class ScientificMCPServer:
    server_name: str
    fabric: ScientificToolFabric

    def __post_init__(self) -> None:
        if self.server_name not in TRUST_BOUNDARIES:
            raise ValueError(f"Unknown scientific MCP trust boundary: {self.server_name}")

    def list_tools(self) -> tuple[str, ...]:
        return self.fabric.list_tools(boundary=self.server_name)

    def call(
        self,
        tool_name: str,
        *,
        posterior_uid: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> ScientificToolReceipt:
        if tool_name not in self.list_tools():
            raise PermissionError(f"Tool {tool_name!r} is outside {self.server_name}")
        return self.fabric.invoke(tool_name, posterior_uid=posterior_uid, inputs=inputs)


def build_scientific_mcp_servers(
    fabric: ScientificToolFabric | None = None,
) -> dict[str, ScientificMCPServer]:
    shared = fabric or ScientificToolFabric()
    return {name: ScientificMCPServer(name, shared) for name in sorted(TRUST_BOUNDARIES)}


__all__ = ["ScientificMCPServer", "build_scientific_mcp_servers"]
