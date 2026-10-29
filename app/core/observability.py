"""Dependency-free request correlation and safe HTTP access logging."""

from __future__ import annotations

from contextvars import ContextVar
import json
import logging
from time import perf_counter
from uuid import uuid4


request_id_context: ContextVar[str | None] = ContextVar("vehicle_iq_request_id", default=None)
logger = logging.getLogger("app.request")


class RequestObservabilityMiddleware:
    """Attach a server-generated request ID and log route timing without payloads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        context_token = request_id_context.set(request_id)
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", ()))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            route = scope.get("route")
            route_template = getattr(route, "path", "unmatched")
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope.get("method", ""),
                        "route": route_template,
                        "status_code": status_code,
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    },
                    separators=(",", ":"),
                )
            )
            request_id_context.reset(context_token)
