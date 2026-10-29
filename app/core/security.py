from __future__ import annotations

import hmac
import re
import threading
import time
from collections import OrderedDict, deque

from starlette.responses import JSONResponse

from app.core.config import settings


_ASSESSMENT_IMAGE_UPLOAD = re.compile(r"^/api/v1/assessments/[0-9a-fA-F-]{36}/images$")
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_RATE_WINDOW_SECONDS = 60.0
_MAX_RATE_LIMIT_CLIENTS = 10_000


class ApiSecurityMiddleware:
    """Protect configured API deployments and cap upload bodies before parsing."""

    def __init__(self, app):
        self.app = app
        self._request_times: OrderedDict[str, deque[float]] = OrderedDict()
        self._rate_lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path.startswith("/api/v1/"):
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            retry_after = self._consume_rate_limit(client_ip, time.monotonic())
            if retry_after is not None:
                await JSONResponse(
                    {"detail": "API request rate limit exceeded."},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )(scope, receive, send)
                return

            token = settings.api_access_token
            production_without_token = (
                settings.app_env.casefold() == "production" and token is None
            )
            if production_without_token:
                await JSONResponse(
                    {"detail": "API access is not configured."}, status_code=503
                )(scope, receive, send)
                return
            if token is not None and not self._valid_bearer(
                scope.get("headers", ()), token.get_secret_value()
            ):
                await JSONResponse(
                    {"detail": "Authentication required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
                return

        if scope.get("method") == "POST" and _ASSESSMENT_IMAGE_UPLOAD.fullmatch(path):
            limit = settings.max_image_upload_bytes + _MULTIPART_OVERHEAD_BYTES
            content_length = self._content_length(scope.get("headers", ()))
            if content_length is not None and content_length > limit:
                await self._too_large(scope, receive, send)
                return

            consumed = 0

            async def bounded_receive():
                nonlocal consumed
                message = await receive()
                if message["type"] == "http.request":
                    consumed += len(message.get("body", b""))
                    if consumed > limit:
                        raise _UploadRequestTooLarge
                return message

            try:
                await self.app(scope, bounded_receive, send)
            except _UploadRequestTooLarge:
                await self._too_large(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _consume_rate_limit(self, client_ip: str, now: float) -> int | None:
        with self._rate_lock:
            calls = self._request_times.get(client_ip)
            if calls is None:
                calls = deque()
                self._request_times[client_ip] = calls
            else:
                self._request_times.move_to_end(client_ip)
            while calls and now - calls[0] >= _RATE_WINDOW_SECONDS:
                calls.popleft()
            if len(calls) >= settings.api_rate_limit_per_minute:
                return max(1, int(_RATE_WINDOW_SECONDS - (now - calls[0])))
            calls.append(now)
            if len(self._request_times) > _MAX_RATE_LIMIT_CLIENTS:
                self._request_times.popitem(last=False)
            return None

    @staticmethod
    def _valid_bearer(headers, expected: str) -> bool:
        authorization = next(
            (value for name, value in headers if name.lower() == b"authorization"),
            b"",
        )
        try:
            scheme, supplied = authorization.decode("ascii").split(None, 1)
        except (UnicodeDecodeError, ValueError):
            return False
        return scheme.casefold() == "bearer" and hmac.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        )

    @staticmethod
    def _content_length(headers) -> int | None:
        value = next(
            (value for name, value in headers if name.lower() == b"content-length"),
            None,
        )
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    async def _too_large(scope, receive, send) -> None:
        await JSONResponse(
            {"detail": "Image upload request exceeds the configured size limit."},
            status_code=413,
        )(scope, receive, send)


class _UploadRequestTooLarge(Exception):
    pass
