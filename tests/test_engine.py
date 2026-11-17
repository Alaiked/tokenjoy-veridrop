import json
import httpx
import pytest
from types import SimpleNamespace
from detector_worker.engine import run_detection, safe_report
from detector_worker.security import SafeTransport


def test_report_drops_all_raw_echo_and_signature_claims():
    result = SimpleNamespace(name='thinking_signature', status='pass', score=100, weight=25,
        display_name='secret', error='secret', details={'signature_length': 123,
        'thinking_block_seen': True, 'signature_received': True, 'signature_prefix': 'secret'})
    report = safe_report([result])
    assert 'secret' not in json.dumps(report)
    check = report['checks'][0]
    assert check['metrics']['signature_length'] == 123
    assert 'cryptographic' in check['summary']
    assert check['label'] == 'Thinking signature fields'


@pytest.mark.parametrize('protocol,model', [('anthropic','claude-sonnet-4-6'),
    ('openai','gpt-4o-mini'), ('gemini','gemini-2.5-pro')])
async def test_real_upstream_runner_uses_injected_transport_no_retries(protocol, model):
    requests = []
    def handler(req):
        requests.append(req)
        return httpx.Response(429, json={'error': {'message': 'synthetic-secret-key'}})
    transport = SafeTransport('https://relay.example', inner=httpx.MockTransport(handler))
    report = await run_detection(dict(protocol=protocol, model=model,
        base_url='https://relay.example', long_context=False), 'synthetic-secret-key', transport=transport)
    assert report['verdict'] == 'inconclusive'
    assert len(requests) == 1
    assert 'synthetic-secret-key' not in json.dumps(report)


def test_errors_and_no_checks_are_inconclusive():
    assert safe_report([])['verdict'] == 'inconclusive'
    r = SimpleNamespace(name='protocol', status='error', score=0, weight=5, details={}, error=None)
    assert safe_report([r])['verdict'] == 'inconclusive'


@pytest.mark.parametrize('protocol', ['anthropic', 'openai'])
async def test_long_context_disabled_sends_nothing_and_error_stops_tiers(protocol):
    import importlib
    from relay_detector.core.models import ExecutionConfig
    detector = importlib.import_module(f'relay_detector.protocols.{protocol}.detectors.long_context').LongContextDetector()
    detector.config = ExecutionConfig(include_long_context=False)
    client = SimpleNamespace(base_url='https://relay.example')
    assert (await detector.run(client, 'gpt-4o-mini')).status == 'skip'
    calls = []
    async def probe(client, model, tokens, seed, limit):
        calls.append(tokens)
        return dict(status='error', target_tokens=tokens, estimated_cost_usd=0,
                    needles_found=0, needles_total=3)
    detector._probe_tier = probe
    detector.config = ExecutionConfig(include_long_context=True)
    await detector.run(client, 'claude-sonnet-4-6' if protocol == 'anthropic' else 'gpt-4o-mini')
    assert calls == [32000]


@pytest.mark.parametrize('protocol', ['anthropic', 'openai'])
async def test_long_context_rate_limit_does_not_retry(protocol, monkeypatch):
    import importlib
    detector = importlib.import_module(f'relay_detector.protocols.{protocol}.detectors.long_context').LongContextDetector()
    async def probe(*args):
        return {'status': 'rate_limited'}
    async def sleep(*args):
        pytest.fail('automatic retry sleep')
    detector._probe_tier = probe
    monkeypatch.setattr('asyncio.sleep', sleep)
    assert await detector._probe_tier_with_tpm_retry(None, 'model', 32000, 'seed', 200000) == {'status': 'rate_limited'}


@pytest.mark.parametrize('protocol,model', [('anthropic','claude-sonnet-4-6'),
    ('openai','gpt-4o-mini'), ('gemini','gemini-2.5-pro')])
async def test_real_runner_success_wire_and_echo_scrubbing(protocol, model):
    requests = []
    class Stream(httpx.AsyncByteStream):
        def __init__(self, data):
            self.data = data
        async def __aiter__(self):
            yield self.data
    def handler(req):
        body = json.loads(req.content)
        requests.append(body)
        usage = dict(input_tokens=30, output_tokens=2) if protocol == 'anthropic' else dict(prompt_tokens=30, completion_tokens=2, total_tokens=32)
        if protocol == 'anthropic':
            response = dict(id='msg_fixture12345678901234567890', type='message', role='assistant', model=model,
                content=[{'type':'thinking','thinking':'test','signature':'s'*100},
                         {'type':'text','text':'synthetic-secret-key'}], usage=usage, stop_reason='end_turn')
            data = ('event: message_start\ndata: '+json.dumps({'type':'message_start','message':response})+'\n\n'
                    'event: content_block_delta\ndata: '+json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'synthetic-secret-key'}})+'\n\n'
                    'event: message_delta\ndata: '+json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':2}})+'\n\n'
                    'event: message_stop\ndata: {"type":"message_stop"}\n\n').encode()
        else:
            response = dict(id='chatcmpl-fixture', object='chat.completion', created=1700000000,
                model=model, choices=[{'index':0,'message':{'role':'assistant','content':'synthetic-secret-key'},'finish_reason':'stop'}], usage=usage)
            chunk = dict(response, object='chat.completion.chunk', choices=[{'index':0,'delta':{'role':'assistant','content':'synthetic-secret-key'},'finish_reason':'stop'}])
            data = ('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()
        if body.get('stream'):
            return httpx.Response(200, headers={'content-type':'text/event-stream'}, stream=Stream(data))
        return httpx.Response(200, json=response)
    transport = SafeTransport('https://relay.example', inner=httpx.MockTransport(handler))
    report = await run_detection(dict(protocol=protocol, model=model,
        base_url='https://relay.example', long_context=False), 'synthetic-secret-key', transport=transport)
    assert len(requests) >= 3
    assert report['checks']
    assert any(c['status'] in ('pass', 'fail') for c in report['checks'])
    assert 'synthetic-secret-key' not in json.dumps(report)
    assert all(len(json.dumps(r)) < 150000 for r in requests)


def test_upstream_scoring_threshold_and_critical_veto():
    r = SimpleNamespace(name='protocol', status='pass', score=75, weight=5, details={}, error=None)
    assert safe_report([r])['verdict'] == 'passed'
    r.details['critical_issue_count'] = 1
    assert safe_report([r])['verdict'] == 'marginal'
