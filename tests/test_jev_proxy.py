"""用本地服务验证 JEV 的环境代理、HTTPS 隧道和直连行为。"""

import asyncio
import json
import os
from types import SimpleNamespace
from urllib.request import getproxies_environment

import pytest

from core.search.jev_selector import JevSelector, JevSelectionError


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["http_proxy", "https_proxy", "no_proxy", "direct"])
async def test_jev_environment_proxy_routing(monkeypatch, mode):
    # 隔离机器已有的环境代理及 Windows 注册表代理，只读取本测试的环境。
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setattr("aiohttp.helpers.getproxies", getproxies_environment)
    origin_requests = []
    proxy_requests = []

    async def respond(reader, writer, requests):
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            request_line, *header_lines = head.split("\r\n")
            headers = dict(
                line.split(": ", 1) for line in header_lines if ": " in line
            )
            body = await reader.readexactly(int(headers.get("Content-Length", "0")))
            requests.append((request_line, headers, json.loads(body) if body else None))
            status = "502 Bad Gateway" if request_line.startswith("CONNECT ") else "200 OK"
            payload = json.dumps(
                {"answers": {"meme": {"type": "choice", "choice": "m01"}}}
            ).encode()
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(
        lambda reader, writer: respond(reader, writer, origin_requests), "127.0.0.1", 0
    ) as origin, await asyncio.start_server(
        lambda reader, writer: respond(reader, writer, proxy_requests), "127.0.0.1", 0
    ) as proxy:
        origin_port = origin.sockets[0].getsockname()[1]
        proxy_url = f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
        scheme = "https" if mode == "https_proxy" else "http"
        base_url = f"{scheme}://127.0.0.1:{origin_port}"
        if mode != "direct":
            # AstrBot 面板会同时写入这两个变量。
            monkeypatch.setenv("http_proxy", proxy_url)
            monkeypatch.setenv("https_proxy", proxy_url)
        if mode == "no_proxy":
            monkeypatch.setenv("no_proxy", "127.0.0.1")
        selector = JevSelector(SimpleNamespace(plugin_config=SimpleNamespace(
            typesafe_api_base_url=base_url, typesafe_api_key="test-key",
        )))
        call = selector.select(
            [("a.png", {})], history=[], user_message="用户", reply="回复",
        )
        if mode == "https_proxy":
            # 代理收到 CONNECT 即证明 HTTPS 请求走了代理，无需真实 TLS 或外网。
            with pytest.raises(JevSelectionError, match="ClientHttpProxyError"):
                await call
            assert proxy_requests[0][0] == f"CONNECT 127.0.0.1:{origin_port} HTTP/1.1"
            assert not origin_requests
        else:
            assert await call == "a.png"
            if mode == "http_proxy":
                assert not origin_requests
                requests = proxy_requests
                expected_target = f"{base_url}/v1/systemone"
            else:
                assert not proxy_requests
                requests = origin_requests
                expected_target = "/v1/systemone"
            assert len(requests) == 1
            request_line, headers, body = requests[0]
            assert request_line == f"POST {expected_target} HTTP/1.1"
            assert headers["Authorization"] == "Bearer test-key"
            assert body["model"] == "jev-latest"
