"""Small helpers for bounded upstream HTTP responses."""

from __future__ import annotations

import httpx


class ExternalResponseTooLarge(Exception):
    """An upstream response exceeded the configured byte ceiling."""


def request_with_response_limit(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_bytes: int,
    **kwargs,
) -> httpx.Response:
    """Read a response incrementally and stop before buffering too much data."""
    with client.stream(method, url, **kwargs) as response:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > max_bytes:
                    raise ExternalResponseTooLarge
            except ValueError:
                pass

        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_bytes():
            received += len(chunk)
            if received > max_bytes:
                raise ExternalResponseTooLarge
            chunks.append(chunk)

        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=b"".join(chunks),
            request=response.request,
            extensions=response.extensions,
        )
