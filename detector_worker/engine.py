"""Run pinned Veridrop probes without upstream report persistence or web code."""
import importlib
import math

from relay_detector.core.models import ExecutionConfig, Mode
from relay_detector.core.scorer import effective_verdict, fatal_run_error
from .security import SafeTransport

ENGINE_VERSION = 'veridrop-e87ed92ae37250f87a6503b321c0bc7109618008-tokenjoy.1'
LABELS = {
    'identity': 'Identity consistency', 'behavioral_signature': 'Behavioral probes',
    'thinking_signature': 'Thinking signature fields', 'consistency': 'Model consistency',
    'knowledge': 'Knowledge probes', 'pdf': 'PDF processing',
    'structured_output': 'Structured output', 'protocol': 'Protocol fields',
    'integrity': 'Response integrity', 'message_id': 'Message identifier fields',
    'token_usage': 'Token usage fields', 'long_context': 'Long context recall',
    'basic_request': 'Basic request', 'model_consistency': 'Model consistency',
    'function_calling': 'Function calling', 'token_billing': 'Token billing fields',
    'token_parity': 'Token field parity', 'model_info': 'Model response fields',
}
SUMMARIES = dict(pass_='Probe conditions satisfied.', fail='Probe conditions not satisfied.',
                 skip='Not tested for this model or scope.', error='Unable to evaluate this probe.')
NUMERIC = {'signature_length', 'request_count', 'input_tokens', 'output_tokens', 'critical_issue_count',
           'ttft_ms', 'total_latency_ms'}
BOOLEAN = {'thinking_block_seen', 'signature_received'}


def safe_report(results, failed=False):
    checks, weighted, weights = [], 0.0, 0.0
    for result in results:
        if result.name not in LABELS:
            continue
        status = result.status if result.status in ('pass', 'fail', 'skip', 'error') else 'error'
        score = float(result.score) if isinstance(result.score, (int, float)) else 0.0
        if not math.isfinite(score):
            score, status = 0.0, 'error'
        score = min(100.0, max(0.0, score))
        summary = SUMMARIES['pass_' if status == 'pass' else status]
        if result.name == 'thinking_signature':
            summary += ' Presence and length only; no cryptographic verification.'
        check = dict(name=result.name, label=LABELS[result.name], status=status,
                     score=score, summary=summary)
        metrics = {}
        details = result.details if isinstance(result.details, dict) else {}
        for key in NUMERIC:
            value = details.get(key)
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1_000_000_000:
                metrics[key] = value
        for key in BOOLEAN:
            if type(details.get(key)) is bool:
                metrics[key] = details[key]
        if metrics:
            check['metrics'] = metrics
        checks.append(check)
        if status != 'skip':
            weight = result.weight if type(result.weight) in (int, float) else 0
            if math.isfinite(weight) and weight > 0:
                weighted += score * weight
                weights += weight
    score = round(weighted / weights, 2) if weights else 0.0
    verdict = effective_verdict(score, results)
    if failed or fatal_run_error(results) or not weights or any(c['status'] == 'error' for c in checks):
        verdict = 'inconclusive'
    return dict(score=score, verdict=verdict, engine_version=ENGINE_VERSION, checks=checks)


async def run_detection(spec, key, *, transport=None):
    protocol = spec['protocol']
    if protocol not in ('anthropic', 'openai', 'gemini'):
        raise ValueError('unsupported_protocol')
    module = f'relay_detector.protocols.{protocol}'
    clients = importlib.import_module(module + '.client')
    runner_type = importlib.import_module(module + '.runner').Runner
    build_all = importlib.import_module(module + '.detectors').build_all
    client_type = getattr(clients, {'anthropic': 'AnthropicClient', 'openai': 'OpenAIChatClient',
                                   'gemini': 'GeminiClient'}[protocol])
    transport = transport or SafeTransport(spec['base_url'], long_context=spec['long_context'])
    config = ExecutionConfig(mode=Mode.STANDARD, max_concurrent=1,
        request_timeout_s=30, overall_timeout_s=210, use_cache=False, persist_cache=False,
        include_long_context=spec['long_context'] and protocol != 'gemini',
        include_long_context_extreme=False)
    client = client_type(base_url=spec['base_url'], api_key=key, timeout=30, transport=transport)
    try:
        runner = runner_type(client, build_all(), config)
        if config.include_long_context:
            runner._mode_detectors = {mode: names | {'long_context'}
                                      for mode, names in runner._mode_detectors.items()}
        outcome = await runner.run(spec['model'])
        return safe_report(outcome.results, failed=transport.failed)
    finally:
        await client.aclose()
        client.api_key = ''
        client._client.headers.clear()
