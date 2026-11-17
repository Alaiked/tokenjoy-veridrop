import asyncio
import json

import httpx
import httpcore
import pytest

from detector_worker.app import create_app
from detector_worker.discovery import CatalogError, ModelCatalog
from detector_worker.security import PublicBackend, SafeTransport

KEY = 'synthetic-catalog-secret'
TOKEN = {'X-Detector-Token': 'test-service-token'}
BODY = {'base_url': 'https://relay.example', 'api_key': KEY}


def factory(handler):
    def build(base_url, **kwargs):
        return SafeTransport(base_url, inner=httpx.MockTransport(handler), **kwargs)
    return build


async def test_authentication_and_invalid_input_are_generic():
    calls = []
    app = create_app(service_token='test-service-token',
                     catalog_transport_factory=factory(lambda r: calls.append(r)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        assert (await c.post('/models', json=BODY)).status_code == 401
        for bad in [dict(BODY, base_url='http://relay.example'), dict(BODY, base_url='https://127.0.0.1'),
                    dict(BODY, api_key='bad\nkey'), dict(BODY, extra=KEY)]:
            r = await c.post('/models', json=bad, headers=TOKEN)
            assert r.status_code == 422
            assert KEY not in r.text
        assert (await c.post('/models', content=b'x' * 17000, headers=TOKEN)).status_code == 413
    assert not calls


async def test_catalog_filters_deduplicates_exact_ids_and_discards_remote_fields():
    ids = ['claude-sonnet-4-6', 'gpt-4o', 'o3', 'o1-mini', 'o4-mini', 'codex-mini',
           'models/gemini-2.5-pro', 'vendor/Claude-3.7', 'anthropic.claude-3-haiku',
           'gpt-4o', 'deepseek-r1', 'llama-3', 'custom', 'o30', 'gpt-unsafe<script>',
           KEY, 'gpt-' + KEY, 'gpt-' + KEY.upper(), 'gpt-' + 'a' * 130, 'gpt-hi\nthere']
    app = create_app(service_token='test-service-token', catalog_transport_factory=factory(
        lambda r: httpx.Response(200, json={'data': [{'id': ident, 'secret': KEY} for ident in ids], 'secret': KEY})))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        r = await c.post('/models', json=BODY, headers=TOKEN)
    assert r.status_code == 200
    assert r.json() == {'models': [
        {'id': 'claude-sonnet-4-6', 'family': 'anthropic'},
        {'id': 'gpt-4o', 'family': 'openai'}, {'id': 'o3', 'family': 'openai'},
        {'id': 'o1-mini', 'family': 'openai'}, {'id': 'o4-mini', 'family': 'openai'},
        {'id': 'codex-mini', 'family': 'openai'},
        {'id': 'models/gemini-2.5-pro', 'family': 'gemini'},
        {'id': 'vendor/Claude-3.7', 'family': 'anthropic'},
        {'id': 'anthropic.claude-3-haiku', 'family': 'anthropic'}]}
    assert KEY not in r.text
    assert not app.state.jobs.records and not app.state.jobs.secrets


@pytest.mark.parametrize('base,path', [
    ('https://relay.example', '/v1/models'),
    ('https://relay.example/v1/', '/v1/models'),
    ('https://relay.example/proxy/v1', '/proxy/v1/models'),
    ('https://relay.example/v1beta/openai', '/v1beta/openai/models'),
    ('https://relay.example/proxy', '/proxy/v1/models')])
async def test_one_get_to_fixed_catalog_path_with_both_auth_headers(base, path):
    seen = []
    def handler(r):
        seen.append(r)
        return httpx.Response(200, json={'data': []})
    catalog = ModelCatalog(transport_factory=factory(handler))
    assert await catalog.discover(base, KEY) == []
    assert len(seen) == 1
    request = seen[0]
    assert request.method == 'GET' and request.url.path == path
    assert request.headers['authorization'] == 'Bearer ' + KEY
    assert request.headers['x-api-key'] == KEY
    assert request.headers['anthropic-version'] == '2023-06-01'
    assert not request.content and not request.url.query


@pytest.mark.parametrize('response', [
    httpx.Response(401, json={'error': KEY}), httpx.Response(429, text=KEY),
    httpx.Response(500, text=KEY), httpx.Response(200, text=KEY),
    httpx.Response(200, json={'error': KEY}), httpx.Response(200, json={'data': [1]}),
    httpx.Response(302, headers={'location': 'https://127.0.0.1/'})])
async def test_remote_errors_never_reflect_key_or_retry(response):
    seen = []
    def handler(r):
        seen.append(r)
        return response
    app = create_app(service_token='test-service-token', catalog_transport_factory=factory(handler))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        r = await c.post('/models', json=BODY, headers=TOKEN)
    assert r.status_code == 502
    assert r.json() == {'detail': 'model_catalog_unavailable'}
    assert KEY not in r.text and len(seen) == 1


async def test_stream_size_bound_and_catalog_row_limit():
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'x' * 600000
            yield b'x' * 600000
    for response in [httpx.Response(200, stream=Stream()),
                     httpx.Response(200, json={'data': [{'id': 'gpt-' + str(i)} for i in range(2049)]})]:
        with pytest.raises(CatalogError) as err:
            await ModelCatalog(transport_factory=factory(lambda r: response)).discover(BODY['base_url'], KEY)
        assert err.value.code == 'model_catalog_unavailable'


async def test_catalog_concurrency_and_total_timeout_release_slot():
    entered = asyncio.Event()
    async def handler(r):
        entered.set()
        await asyncio.sleep(100)
    catalog = ModelCatalog(transport_factory=factory(handler), concurrency=1, timeout=.05)
    task = asyncio.create_task(catalog.discover(BODY['base_url'], KEY))
    await entered.wait()
    with pytest.raises(CatalogError) as err:
        await catalog.discover(BODY['base_url'], KEY)
    assert err.value.status == 429
    with pytest.raises(CatalogError) as err:
        await task
    assert err.value.code == 'model_catalog_timeout'
    assert catalog.active == 0


@pytest.mark.parametrize('depth', range(4))
async def test_url_key_echo_rejected_before_network_including_encoding(depth):
    encoded = ''.join('%' + format(ord(char), '02x') for char in KEY)
    for _ in range(max(0, depth - 1)):
        encoded = encoded.replace('%', '%25')
    if depth == 0:
        encoded = KEY
    seen = []
    app = create_app(service_token='test-service-token',
                     catalog_transport_factory=factory(lambda r: seen.append(r)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        r = await c.post('/models', json=dict(BODY, base_url=BODY['base_url'] + '/' + encoded), headers=TOKEN)
    assert r.status_code == 422
    assert KEY not in r.text and not seen


async def test_catalog_result_bound_and_empty_supported_catalog():
    response = httpx.Response(200, json={'data': [{'id': 'gpt-' + str(i)} for i in range(600)]})
    catalog = ModelCatalog(transport_factory=factory(lambda r: response))
    rows = await catalog.discover(BODY['base_url'], KEY)
    assert len(rows) == 512 and rows[-1]['id'] == 'gpt-511'
    catalog = ModelCatalog(transport_factory=factory(lambda r: httpx.Response(200, json={'data': [{'id':'llama-3'}]})))
    assert await catalog.discover(BODY['base_url'], KEY) == []


async def test_actual_connection_backend_denies_private_dns_before_headers_leave():
    resolutions, dials = [], []
    async def resolve(host, port):
        resolutions.append((host, port))
        return ['8.8.8.8', '127.0.0.1']
    class Dialer:
        async def connect_tcp(self, host, port, **kwargs):
            dials.append(host)
            pytest.fail('private DNS catalog must not connect')
    def build(base_url, **kwargs):
        inner = httpx.AsyncHTTPTransport(trust_env=False, retries=0)
        inner._pool = httpcore.AsyncConnectionPool(network_backend=PublicBackend(resolve, Dialer()), retries=0)
        return SafeTransport(base_url, inner=inner, **kwargs)
    catalog = ModelCatalog(transport_factory=build)
    with pytest.raises(CatalogError):
        await catalog.discover(BODY['base_url'], KEY)
    assert resolutions == [('relay.example', 443)] and not dials


async def test_transport_timeout_is_generic_504_and_not_retried():
    calls = []
    async def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout(KEY)
    app = create_app(service_token='test-service-token', catalog_transport_factory=factory(handler))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        r = await c.post('/models', json=BODY, headers=TOKEN)
    assert r.status_code == 504
    assert r.json() == {'detail': 'model_catalog_timeout'}
    assert len(calls) == 1 and KEY not in r.text
