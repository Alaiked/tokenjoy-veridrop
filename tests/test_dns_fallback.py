import asyncio
import httpx
import pytest

from detector_worker import security


def system_answers(monkeypatch, addresses):
    async def resolve(*args, **kwargs):
        return [(0, 0, 0, '', (address, 443)) for address in addresses]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)


async def test_all_benchmark_fake_ips_use_doh_and_validate_pinned_public_result(monkeypatch):
    system_answers(monkeypatch, ['198.18.0.9', '198.19.255.254'])
    seen = []
    async def doh(host):
        seen.append(host)
        return ['8.8.8.8']
    monkeypatch.setattr(security, 'resolve_doh', doh, raising=False)
    assert await security.resolve_public('relay.example', 443) == ['8.8.8.8']
    assert seen == ['relay.example']


async def test_normal_public_system_dns_does_not_contact_doh(monkeypatch):
    system_answers(monkeypatch, ['8.8.8.8', '2606:4700:4700::1111'])
    async def doh(host):
        pytest.fail('unnecessary third-party DNS query')
    monkeypatch.setattr(security, 'resolve_doh', doh, raising=False)
    assert await security.resolve_public('relay.example', 443) == ['8.8.8.8', '2606:4700:4700::1111']


@pytest.mark.parametrize('addresses', [['127.0.0.1'], ['10.0.0.1'], ['8.8.8.8', '10.0.0.1'],
    ['198.18.0.9', '8.8.8.8'], ['198.18.0.9', '127.0.0.1'], []])
async def test_private_or_mixed_dns_does_not_trigger_fallback(monkeypatch, addresses):
    system_answers(monkeypatch, addresses)
    async def doh(host):
        pytest.fail('must not rescue mixed or private DNS')
    monkeypatch.setattr(security, 'resolve_doh', doh, raising=False)
    with pytest.raises(security.UnsafeTarget):
        await security.resolve_public('relay.example', 443)


async def test_fixed_doh_endpoint_only_receives_hostname_and_a_query():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={'Status': 0, 'Answer': [
            {'name': 'relay.example.', 'type': 5, 'data': 'canonical.example.'},
            {'name': 'canonical.example.', 'type': 1, 'data': '8.8.8.8'}]})
    result = await security.resolve_doh('relay.example', transport=httpx.MockTransport(handler))
    assert result == ['8.8.8.8']
    assert len(seen) == 1
    request = seen[0]
    assert request.url.host == '1.1.1.1' and request.url.path == '/dns-query'
    assert dict(request.url.params) == {'name': 'relay.example', 'type': 'A'}
    assert request.headers['accept'] == 'application/dns-json'
    assert 'authorization' not in request.headers and 'x-api-key' not in request.headers
    assert not request.content


@pytest.mark.parametrize('response', [
    httpx.Response(302, headers={'location': 'https://127.0.0.1/'}),
    httpx.Response(200, text='not-json'), httpx.Response(200, json={'Status': 3}),
    httpx.Response(200, json={'Status': 0, 'Answer': []}),
    httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '127.0.0.1'}]}),
    httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '8.8.8.8'}, {'type': 1, 'data': '10.0.0.1'}]}),
    httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': 'not-an-ip'}]}),
    httpx.Response(200, content=b'x' * 16385),
])
async def test_bad_doh_reply_fails_closed_without_retry(response):
    calls = []
    def handler(request):
        calls.append(request)
        return response
    with pytest.raises(security.UnsafeTarget):
        await security.resolve_doh('relay.example', transport=httpx.MockTransport(handler))
    assert len(calls) == 1


async def test_doh_stream_size_and_total_timeout_are_bounded(monkeypatch):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'x' * 9000
            yield b'x' * 9000
    with pytest.raises(security.UnsafeTarget):
        await security.resolve_doh('relay.example', transport=httpx.MockTransport(
            lambda r: httpx.Response(200, stream=Stream())))
    monkeypatch.setattr(security, 'DOH_TIMEOUT', .02)
    async def slow(request):
        await asyncio.Event().wait()
    with pytest.raises(security.UnsafeTarget):
        await security.resolve_doh('relay.example', transport=httpx.MockTransport(slow))


async def test_failed_fallback_and_private_fallback_result_are_not_dialed(monkeypatch):
    system_answers(monkeypatch, ['198.18.0.9'])
    for value in (['198.18.0.9'], ['10.0.0.1']):
        async def doh(host):
            return value
        monkeypatch.setattr(security, 'resolve_doh', doh, raising=False)
        with pytest.raises(security.UnsafeTarget):
            await security.resolve_public('relay.example', 443)
