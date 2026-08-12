"""Unit tests for the strict MCP namespace adapter."""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
os.environ.setdefault(
    "BCP_CAMPAIGN_CONFIG",
    str(PACKAGE_ROOT / "configs" / "campaign.example.json"),
)
os.environ.setdefault("BCP_RUN_ID", "offline-test")
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))

from query_worker import CampaignAssistant


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {"docid": {"type": "string"}},
    "required": ["docid"],
}


def advertised():
    return [
        {"name": "search", "inputSchema": copy.deepcopy(SEARCH_SCHEMA)},
        {"name": "get_document", "inputSchema": copy.deepcopy(DOCUMENT_SCHEMA)},
    ]


def offered():
    return [
        {
            "name": "search-server-search",
            "function": {"name": "search-server-search", "parameters": copy.deepcopy(SEARCH_SCHEMA)},
        },
        {
            "name": "search-server-get_document",
            "function": {
                "name": "search-server-get_document",
                "parameters": copy.deepcopy(DOCUMENT_SCHEMA),
            },
        },
    ]


class MCPNamespaceCanonicalizationTest(unittest.TestCase):
    def test_two_exact_direct_name_mappings(self):
        tools = offered()
        self.assertEqual(
            CampaignAssistant._canonicalize_mcp_tool_name("search", tools, advertised()),
            ("search-server-search", "mcp_namespace_suffix_exact_schema"),
        )
        self.assertEqual(
            CampaignAssistant._canonicalize_mcp_tool_name("get_document", tools, advertised()),
            ("search-server-get_document", "mcp_namespace_suffix_exact_schema"),
        )

    def test_unknown_name_is_rejected(self):
        result = CampaignAssistant._canonicalize_mcp_tool_name(
            "get_documents", offered(), advertised()
        )
        self.assertEqual(result, (None, "unknown_mcp_tool_name"))

    def test_ambiguous_suffix_is_rejected(self):
        tools = offered()
        tools.append(
            {
                "name": "other-server-get_document",
                "function": {"name": "other-server-get_document", "parameters": copy.deepcopy(DOCUMENT_SCHEMA)},
            }
        )
        result = CampaignAssistant._canonicalize_mcp_tool_name(
            "get_document", tools, advertised()
        )
        self.assertEqual(result, (None, "mcp_namespace_ambiguous"))

    def test_schema_mismatch_is_rejected(self):
        tools = offered()
        tools[1]["function"]["parameters"]["properties"]["docid"]["type"] = "number"
        result = CampaignAssistant._canonicalize_mcp_tool_name(
            "get_document", tools, advertised()
        )
        self.assertEqual(result, (None, "mcp_namespace_schema_mismatch"))


if __name__ == "__main__":
    unittest.main()
