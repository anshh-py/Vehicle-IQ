import httpx
import pytest

from app.core.http import ExternalResponseTooLarge, request_with_response_limit


def test_response_body_is_capped_while_streaming():
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b"1234"
            yield b"5678"

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=Chunks())
        )
    )

    with pytest.raises(ExternalResponseTooLarge):
        request_with_response_limit(
            client, "GET", "https://provider.example", max_bytes=6
        )


def test_declared_oversized_response_is_rejected_before_body_read():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, headers={"content-length": "1000"}, stream=httpx.ByteStream(b"x" * 1000)
            )
        )
    )

    with pytest.raises(ExternalResponseTooLarge):
        request_with_response_limit(
            client, "GET", "https://provider.example", max_bytes=10
        )
