"""Tool abstraction: Pydantic args schema -> OpenAI function schema, validated execution, structured errors.

Every tool result returned to the model is JSON:
    {"ok": true,  "data": ...}
    {"ok": false, "error": {"type": ..., "message": ..., "hint": ...}}
so the model can read the error and self-correct instead of the loop crashing.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class ToolError(Exception):
    type = "tool_error"

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message, self.hint = message, hint


class NotFoundError(ToolError):
    type = "not_found"


class PermissionDeniedError(ToolError):
    type = "permission_denied"


class BusinessRuleError(ToolError):
    type = "business_rule_violation"


@dataclass
class ToolContext:
    """Runtime context injected by the server, never by the LLM (prevents IDOR via prompt injection)."""
    conn: sqlite3.Connection
    customer_id: str
    today: date
    retriever: Any = None
    side_effects: list[dict] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], Any]
    writes: bool = False  # has side effects (ticket, handoff)

    def openai_schema(self) -> dict:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for p in params.get("properties", {}).values():
            p.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self.tools.values()]

    def execute(self, name: str, raw_args: str | dict, ctx: ToolContext) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return _err("unknown_tool", f"tool '{name}' does not exist", f"available: {sorted(self.tools)}")
        try:
            args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as e:
            return _err("invalid_json", f"arguments are not valid JSON: {e.msg}", "return a JSON object")
        try:
            parsed = tool.args_model.model_validate(args)
        except ValidationError as e:
            details = "; ".join(f"{'.'.join(map(str, x['loc'])) or '(root)'}: {x['msg']}" for x in e.errors())
            return _err("invalid_arguments", details, "fix the arguments according to the schema and retry")
        try:
            return {"ok": True, "data": tool.fn(ctx, parsed)}
        except ToolError as e:
            return _err(e.type, e.message, e.hint)
        except Exception as e:  # never let a tool crash the agent loop
            log.exception("tool %s crashed", name)
            return _err("internal_error", f"{type(e).__name__}: {e}", "do not retry; apologise or hand off to human")


def _err(type_: str, message: str, hint: str = "") -> dict:
    return {"ok": False, "error": {"type": type_, "message": message, "hint": hint}}
