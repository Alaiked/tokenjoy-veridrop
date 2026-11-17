import asyncio
import json
import httpx
import pytest
from detector_worker.app import create_app

PAYLOAD = dict(id='test-job-1', protocol='openai', base_url='https://relay.example',
               api_key='synthetic-secret-key', model='gpt-4o-mini', long_context=False)
HEADERS = {'X-Detector-Token': 'test-service-token'}


async def poll(client, ident='test-job-1'):
    for _ in range(100):
        job = (await client.get('/jobs/' + ident, headers=HEADERS)).json()
        if job['state'] not in ('queued', 'running'):
            return job
        await asyncio.sleep(.005)
    pytest.fail('job did not finish')


async def test_auth_validation_health_and_no_key_reflection():
    app = create_app(service_token='test-service-token')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        assert (await c.get('/healthz')).status_code == 200
        assert (await c.post('/jobs', json=PAYLOAD)).status_code == 401
        bad = dict(PAYLOAD, protocol='bad', api_key='synthetic-secret-key')
        r = await c.post('/jobs', json=bad, headers=HEADERS)
        assert r.status_code == 422
        assert 'synthetic-secret-key' not in r.text


async def test_idempotency_and_credential_release():
    gate = asyncio.Event()
    calls = []
    async def engine(spec, key):
        calls.append(key)
        await gate.wait()
        return {'score': 100, 'verdict': 'passed', 'engine_version': 'fixture', 'checks': []}
    app = create_app(service_token='test-service-token', engine=engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        a = await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        b = await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        assert a.json()['id'] == b.json()['id']
        gate.set()
        job = await poll(c)
        assert job['state'] == 'done'
        assert calls == ['synthetic-secret-key']
        assert 'synthetic-secret-key' not in json.dumps(app.state.jobs.records)
        assert not app.state.jobs.secrets


async def test_cancel_terminates_inflight_and_drops_secret():
    cancelled = asyncio.Event()
    async def engine(spec, key):
        try:
            await asyncio.sleep(100)
        finally:
            cancelled.set()
    app = create_app(service_token='test-service-token', engine=engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        await asyncio.sleep(.01)
        r = await c.delete('/jobs/test-job-1', headers=HEADERS)
        assert r.json()['state'] == 'cancelled'
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not app.state.jobs.secrets


async def test_queue_bound_and_generic_timeout():
    async def engine(spec, key):
        await asyncio.sleep(100)
    app = create_app(service_token='test-service-token', engine=engine,
                     concurrency=1, capacity=1, job_timeout=.02)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        r = await c.post('/jobs', json=dict(PAYLOAD, id='other'), headers=HEADERS)
        assert r.status_code == 429
        job = await poll(c)
        assert job['state'] == 'error' and job['error_code'] == 'timeout'
        assert not app.state.jobs.secrets


async def test_internal_exception_never_leaks():
    async def engine(spec, key):
        raise RuntimeError(key)
    app = create_app(service_token='test-service-token', engine=engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        job = await poll(c)
        assert job['error_code'] == 'upstream_error'
        assert 'synthetic-secret-key' not in json.dumps(job)


async def test_terminal_outbox_survives_restart_until_ack_without_credentials(tmp_path):
    async def engine(spec, key):
        return {'score': 100, 'verdict': 'passed', 'engine_version': 'fixture', 'checks': []}
    app = create_app(service_token='test-service-token', engine=engine, report_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        assert (await poll(c))['state'] == 'done'
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert 'synthetic-secret-key' not in files[0].read_text()
    restarted = create_app(service_token='test-service-token', engine=engine, report_dir=tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(restarted), base_url='http://worker') as c:
        assert (await c.get('/jobs/test-job-1', headers=HEADERS)).json()['state'] == 'done'
        assert (await c.post('/jobs/test-job-1/ack', headers=HEADERS)).status_code == 200
        assert not list(tmp_path.glob('*.json'))


async def test_scope_idempotency_and_retention(tmp_path):
    async def engine(spec, key):
        return {'score': 100, 'verdict': 'passed', 'engine_version': 'fixture', 'checks': []}
    app = create_app(service_token='test-service-token', engine=engine,
                     retention=.001, report_dir=tmp_path, max_records=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        await poll(c)
        await asyncio.sleep(.005)
        # Unacknowledged paid reports cannot be pruned to make capacity.
        assert (await c.get('/jobs/test-job-1', headers=HEADERS)).status_code == 200
        assert (await c.post('/jobs', json=dict(PAYLOAD, long_context=True), headers=HEADERS)).status_code == 409
        assert (await c.post('/jobs', json=dict(PAYLOAD, id='other'), headers=HEADERS)).status_code == 429
        await c.post('/jobs/test-job-1/ack', headers=HEADERS)
        app.state.jobs.expiry['test-job-1'] = 0
        assert (await c.get('/jobs/test-job-1', headers=HEADERS)).status_code == 404


async def test_ready_recovers_full_acknowledged_capacity(tmp_path):
    async def engine(spec, key):
        return {'score': 100, 'verdict': 'passed', 'engine_version': 'fixture', 'checks': []}
    app = create_app(service_token='test-service-token', engine=engine,
                     report_dir=tmp_path, max_records=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://worker') as c:
        assert (await c.get('/readyz')).status_code == 401
        await c.post('/jobs', json=PAYLOAD, headers=HEADERS)
        await poll(c)
        assert not (await c.get('/readyz', headers=HEADERS)).json()['ready']
        await c.post('/jobs/test-job-1/ack', headers=HEADERS)
        assert (await c.get('/readyz', headers=HEADERS)).json()['ready']
        assert (await c.post('/jobs', json=dict(PAYLOAD, id='replacement'), headers=HEADERS)).status_code == 202
        await poll(c, 'replacement')
