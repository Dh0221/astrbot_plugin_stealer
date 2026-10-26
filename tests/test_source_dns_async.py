"""外部源 DNS 预检的线程边界与事件循环调度回归。"""

import asyncio
import socket
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from core.sources.github_source import GitHubSource
from core.sources.http_source import HTTPSource, _ValidatingResolver
from core.sources.models import ExternalSourceSecurityError, SourceInspection
from core.sources.source_service import SourceService
from tests.test_external_sources import _FakeResponse, _FakeSession


def _addresses(address, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]


@pytest.fixture
def worker_dns(monkeypatch):
    loop_thread = threading.get_ident()
    calls = []

    def resolve(host, port, **kwargs):
        if threading.get_ident() == loop_thread:
            raise RuntimeError("DNS preflight ran on the event-loop thread")
        calls.append(host)
        return _addresses("127.0.0.1" if host == "private.example" else "8.8.8.8", port)

    monkeypatch.setattr("core.sources.http_source.socket.getaddrinfo", resolve)
    return calls


class _ContextSession(_FakeSession):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


@pytest.fixture
def fake_client(monkeypatch):
    def install(session):
        monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: session)
        monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kwargs: object())
        monkeypatch.setattr(aiohttp.resolver, "ThreadedResolver", lambda: SimpleNamespace())

    return install


@pytest.mark.asyncio
async def test_http_factory_does_not_resolve_dns_on_event_loop(tmp_path, monkeypatch, worker_dns):
    service = SourceService(SimpleNamespace(base_dir=tmp_path, plugin_config=SimpleNamespace()))
    result = SourceInspection(source_type="http_json", source_id="catalog", name="Catalog")
    monkeypatch.setattr(HTTPSource, "inspect", AsyncMock(return_value=result))
    inspection, reader = await service.inspect({"type": "http_json", "endpoint": "https://catalog.example/list"})
    try:
        assert inspection is result
        assert reader.endpoint == "https://catalog.example/list"
        assert worker_dns == ["catalog.example"]
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_http_requests_and_redirects_offload_dns(worker_dns):
    source = await asyncio.to_thread(HTTPSource, "https://catalog.example/list")
    source._session = _FakeSession([
        _FakeResponse(302, headers={"Location": "https://cdn.example/image.png"}),
        _FakeResponse(200, chunks=[b"image"]),
    ])
    try:
        assert await source._request_bytes(source.endpoint, max_bytes=1024) == b"image"
        assert "catalog.example" in worker_dns and "cdn.example" in worker_dns
    finally:
        await source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_github_requests_and_redirects_offload_dns(tmp_path, worker_dns, fake_client, stream):
    source = GitHubSource({"repository": "example/catalog", "ref": "main"}, cache_dir=tmp_path)
    source.headers = {"Authorization": "Bearer test-token", "X-API-Key": "test-key"}
    session = _ContextSession([
        _FakeResponse(302, headers={"Location": "https://cdn.example/archive.zip"}),
        _FakeResponse(200, chunks=[b"archive"]),
    ])
    fake_client(session)
    if stream:
        target = tmp_path / "archive.part"
        assert await source._stream_to_file("https://api.github.com/catalog", target) == 7
        assert target.read_bytes() == b"archive"
    else:
        assert await source._request_bytes("https://api.github.com/catalog", max_bytes=1024) == b"archive"
    assert "api.github.com" in worker_dns and "cdn.example" in worker_dns
    assert "Authorization" in session.calls[0][1]
    assert "Authorization" not in session.calls[1][1]
    assert "X-API-Key" not in session.calls[1][1]


@pytest.mark.asyncio
async def test_github_archive_preflight_offloads_dns(tmp_path, monkeypatch, worker_dns):
    source = GitHubSource({"repository": "example/catalog", "ref": "main"}, cache_dir=tmp_path)

    async def stream(url, target):
        target.write_bytes(b"archive")
        return 7

    monkeypatch.setattr(source, "_stream_to_file", stream)
    target = await source._download_archive()
    assert target.read_bytes() == b"archive"
    assert worker_dns == ["codeload.github.com"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["http", "github", "github_stream"])
async def test_private_dns_still_rejected_before_request(tmp_path, worker_dns, fake_client, kind):
    session = _ContextSession([])
    fake_client(session)
    if kind == "http":
        source = await asyncio.to_thread(HTTPSource, "https://catalog.example/list")
        source._session = session
    else:
        source = GitHubSource({"repository": "example/catalog", "ref": "main"}, cache_dir=tmp_path)
    try:
        with pytest.raises(ExternalSourceSecurityError):
            if kind == "github_stream":
                await source._stream_to_file("https://private.example/image", tmp_path / "image.part")
            else:
                await source._request_bytes("https://private.example/image", max_bytes=1024)
        assert not session.calls
    finally:
        await source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["http", "github", "github_stream"])
async def test_redirect_to_private_dns_is_rejected(tmp_path, worker_dns, fake_client, kind):
    session = _ContextSession([
        _FakeResponse(302, headers={"Location": "https://private.example/image"}),
    ])
    fake_client(session)
    if kind == "http":
        source = await asyncio.to_thread(HTTPSource, "https://catalog.example/list")
        source._session = session
    else:
        source = GitHubSource({"repository": "example/catalog", "ref": "main"}, cache_dir=tmp_path)
    try:
        with pytest.raises(ExternalSourceSecurityError):
            if kind == "github_stream":
                await source._stream_to_file("https://public.example/image", tmp_path / "image.part")
            else:
                await source._request_bytes("https://public.example/image", max_bytes=1024)
        assert len(session.calls) == 1
    finally:
        await source.close()


@pytest.mark.asyncio
async def test_slow_dns_keeps_event_loop_schedulable(monkeypatch):
    monkeypatch.setattr("core.sources.http_source.socket.getaddrinfo", lambda host, port, **kwargs: _addresses("8.8.8.8", port))
    source = HTTPSource("https://catalog.example/list")
    source._session = _FakeSession([_FakeResponse(200, chunks=[b"image"])])
    started, release = threading.Event(), threading.Event()

    def slow_dns(host, port, **kwargs):
        started.set()
        if not release.wait(3):
            raise TimeoutError("test DNS was not released")
        return _addresses("8.8.8.8", port)

    monkeypatch.setattr("core.sources.http_source.socket.getaddrinfo", slow_dns)
    task = asyncio.create_task(source._request_bytes(source.endpoint, max_bytes=1024))
    try:
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 2), timeout=3)
        await asyncio.sleep(0)
        assert not task.done(), "the event loop only resumed after blocking DNS finished"
        release.set()
        assert await task == b"image"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await source.close()


@pytest.mark.asyncio
async def test_connection_resolver_still_rejects_dns_rebinding():
    delegate = SimpleNamespace(resolve=AsyncMock(return_value=[{"host": "127.0.0.1"}]), close=AsyncMock())
    resolver = _ValidatingResolver(delegate)
    with pytest.raises(OSError, match="private address"):
        await resolver.resolve("public.example", 443)
    await resolver.close()
    delegate.close.assert_awaited_once()
