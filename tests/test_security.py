import asyncio
import httpx
import httpcore
import pytest

from detector_worker.security import PublicBackend, SafeTransport, UnsafeTarget, validate_url


@pytest.mark.parametrize('url', [
    'http://example.com', 'https://user:pass@example.com', 'https://example.com/?secret=x',
    'https://example.com/#x', 'https://127.0.0.1', 'https://[::1]',
    'https://[::ffff:8.8.8.8]', 'https://169.254.169.254', 'https://10.0.0.1',
    'https://100.64.0.1', 'https://192.0.0.1', 'https://[2001:db8::1]',
    'https://example.com:8443', 'https://example.com\\@127.0.0.1',
    'https://[64:ff9b::a00:1]', 'https://[64:ff9b:1::a00:1]',
    'https://192.0.0.9', 'https://192.88.99.1',
])
def test_reject_unsafe_urls(url):
    with pytest.raises(UnsafeTarget):
        validate_url(url)


async def test_connection_resolves_each_time_and_pins_public_address():
    answers = iter([['8.8.8.8'], ['127.0.0.1']])
    calls = []
    async def resolve(host, port):
        return next(answers)
    class Dialer:
        async def connect_tcp(self, host, port, **kwargs):
            calls.append((host, port))
            return 'stream'
    backend = PublicBackend(resolver=resolve, delegate=Dialer())
    assert await backend.connect_tcp('relay.example', 443) == 'stream'
    with pytest.raises(UnsafeTarget):
        await backend.connect_tcp('relay.example', 443)
    assert calls == [('8.8.8.8', 443)]


async def test_mixed_dns_answer_denies_whole_connection():
    async def resolve(host, port):
        return ['8.8.8.8', '10.0.0.2']
    with pytest.raises(UnsafeTarget):
        await PublicBackend(resolver=resolve).connect_tcp('relay.example', 443)


async def test_preserves_tls_hostname_and_host_and_blocks_redirect():
    seen = []
    async def handler(req):
        seen.append((req.url.host, req.headers['host']))
        return httpx.Response(302, headers={'location': 'https://127.0.0.1/'})
    transport = SafeTransport('https://relay.example', inner=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(UnsafeTarget):
            await client.get('https://relay.example/v1')
    assert seen == [('relay.example', 'relay.example')]


async def test_response_byte_limit_streaming():
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'x' * 600
            yield b'x' * 600
    transport = SafeTransport('https://relay.example', max_response=1000,
        inner=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UnsafeTarget):
            await client.get('https://relay.example/')


async def test_never_send_key_to_other_origin():
    calls = []
    transport = SafeTransport('https://relay.example',
        inner=httpx.MockTransport(lambda r: calls.append(r)))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(UnsafeTarget):
            await client.get('https://other.example/', headers={'authorization': 'secret'})
    assert not calls


async def test_stream_first_chunk_is_available_before_second_chunk():
    gate = asyncio.Event()
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'first'
            await gate.wait()
            yield b'second'
    transport = SafeTransport('https://relay.example',
        inner=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())))
    async with httpx.AsyncClient(transport=transport) as c:
        response = await asyncio.wait_for(c.send(c.build_request('GET', 'https://relay.example/'), stream=True), .1)
        try:
            stream = response.aiter_bytes()
            assert await asyncio.wait_for(anext(stream), .1) == b'first'
            gate.set()
            assert await anext(stream) == b'second'
        finally:
            await response.aclose()


async def test_real_httpcore_tls_sni_and_host_preserved_with_pinned_tcp():
    seen = {}
    class Stream:
        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            seen['sni'] = server_hostname
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == 2
            return self
        async def write(self, buffer, timeout=None):
            seen.setdefault('wire', b'')
            seen['wire'] += buffer
        async def read(self, max_bytes, timeout=None):
            return b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok'
        async def aclose(self):
            pass
        def get_extra_info(self, info):
            return None
    class Dialer:
        async def connect_tcp(self, host, port, **kwargs):
            seen['tcp_host'] = host
            return Stream()
    async def resolve(host, port):
        return ['8.8.8.8']
    inner = httpx.AsyncHTTPTransport(trust_env=False)
    inner._pool = httpcore.AsyncConnectionPool(network_backend=PublicBackend(resolve, Dialer()))
    async with httpx.AsyncClient(transport=SafeTransport('https://relay.example', inner=inner)) as c:
        assert (await c.get('https://relay.example/v1')).text == 'ok'
    assert seen['tcp_host'] == '8.8.8.8'
    assert seen['sni'] == 'relay.example'
    assert b'Host: relay.example\r\n' in seen['wire']
