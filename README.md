# TokenJoy Veridrop worker

Private, self-hosted model protocol and behavioral probes based on
[Veridrop](https://github.com/canarybyte/veridrop), pinned to
`e87ed92ae37250f87a6503b321c0bc7109618008`. Copyright and AGPL-3.0-or-later
terms are preserved in [LICENSE](LICENSE) and [NOTICE](NOTICE).

Corresponding modified source: https://github.com/Alaiked/tokenjoy-veridrop.
Publish the exact source revision used to build your image, including further
modifications, and make it available to network users under AGPL section 13.
This repository contains the worker and vendored engine, not the upstream web
service. Its license does not by itself resolve obligations for other integrated
applications; evaluate the actual integration and license terms.

## Build and test

Python 3.10+; production Docker uses Python 3.12.12. Dependencies, including the
audited httpx/httpcore integration versions, are pinned in requirements files.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests -q
docker build -t tokenjoy-veridrop:local .
```

Windows: activate `.venv/Scripts/Activate.ps1`. The suite uses synthetic secrets,
fake DNS/socket streams and mock HTTP transports, with the actual upstream
Runners. It never calls paid APIs. On a Windows restricted execution sandbox,
asyncio's own local socketpair may need execution outside that sandbox.

## Run

Set a randomly generated `DETECTOR_SERVICE_TOKEN` (at least 16 characters).
Production must set `DETECTOR_REPORT_DIR=/data` and mount a persistent Docker
volume at `/data`; the image initializes its owner as UID/GID 10001. The directory
stores only sanitized terminal job/report JSON. No credential or running-job
state is written. Start one process only:

```sh
PYTHONPATH=src:. uvicorn detector_worker.app:create_app --factory --host 0.0.0.0 --port 8080 --workers 1 --no-access-log --log-level critical
```

Expose port 8080 only to the authenticated website backend on a dedicated
network. Use a read-only root filesystem, non-root UID 10001, `/tmp` tmpfs,
`cap_drop: [ALL]`, `no-new-privileges`, a PID limit, memory/CPU limits, bounded
container logs, and no Docker socket or host-network access. Bind no public
host port. Pin the deployed image digest after building and testing it.

Network policy must deny access to internal databases, Redis, control planes,
host services and metadata endpoints. The application permits outbound public
HTTPS on port 443 only; it checks every DNS answer at each actual TCP connection,
rejects special/private/mapped/transition addresses, and dials the validated
numeric IP. httpcore preserves the original HTTP Host and TLS hostname and
uses normal certificate verification. No proxy environment, redirects or
connection retries are used. All probe clients receive the same hardened
transport. Bounded streams preserve actual first-byte timings.

## Internal API

`GET /healthz` is unauthenticated and returns only `{ "status": "ok" }`.
All remaining endpoints require `X-Detector-Token`:

- `GET /readyz`: `{ready,engine_version}`; false on outbox failure/fullness.
- `POST /jobs`: website-generated `id`, `protocol`, `base_url`, `api_key`,
  `model`, `long_context` (default false); HTTP 202.
- `GET /jobs/{id}`: queued/running/terminal job including sanitized report.
- `DELETE /jobs/{id}`: cancel queued/running job and close in-flight requests.
- `POST /jobs/{id}/ack`: website calls only after its database durably stores
  the terminal record; removes its durable outbox entry. Safe to repeat while
  the in-memory terminal job remains available.

Job shape: `{id,protocol,base_url,model,long_context,state,created_at,
finished_at?,report?,error_code?}`. Dates are ISO UTC. Reports contain
`score,verdict,engine_version,checks`. Each check contains fixed known
`name,label,status,score,summary` and optional numeric/boolean `metrics` only.
Metrics currently permit signature presence/length, critical-issue count,
request/token counts and latency fields. No response text, signature prefix,
exception message, auth header, raw payload or arbitrary upstream label is
returned or stored. Errors use generic codes. Idempotent IDs return the
existing job without another probe; changed target/model/protocol/scope is 409.
Use new random IDs only after explicit user resubmission, never automatic retry.

`protocol` supports Anthropic Messages, OpenAI Chat Completions, and **Gemini
through its OpenAI-compatible Chat Completions endpoint**. Gemini-native
`generateContent` is not supported by this pinned engine.

Basic probes use upstream STANDARD mode with cache/persistence disabled.
Long context is separately opt-in for Anthropic/OpenAI, never Gemini. Only
standard approximate 32k / 100k / 200k input-token tiers within the known model
limit are tested; token counts vary by tokenizer. Extreme tiers are disabled.
Probing stops at the first failed, partial or error tier. No automatic billed
request retry occurs, including upstream long-context TPM and Gemini streaming
fallback retries. Reference OpenAI environment credentials/cache calls are
disabled. Each request can incur provider charges; cancellation cannot undo
charges already incurred.

Thinking signatures are checked for field existence and length only. No
cryptographic verification is performed. Protocol agreement or a high score
cannot prove exact model identity, upstream account or procurement channel.
Weighted score and 70/50 verdict thresholds follow upstream; critical findings
cap a passing verdict at marginal. Any observed transport, auth, quota, rate,
server or detector error makes the overall verdict inconclusive. Upstream raw
exception details are discarded.

## Bounds, privacy and recovery

Defaults: two simultaneous jobs; twelve active/queued jobs; 2,048 retained
records; 240-second job deadline; 30-second read/write and 10-second connection
timeouts; 48 requests/job; 16,384 requested output tokens/request and 65,536
cumulative requested output tokens/job. Standard request bodies are at most
150 KB each / 1 MB cumulative, opt-in long context 1.5 MB each / 4 MB cumulative.
Response streams are capped at 2 MB each; compression is refused. Inbound job
JSON is capped at 16 KB. These are worker safety ceilings, not fee estimates.

Keys remain only in request/queued/execution memory, never hashes, masked
prefixes, logs or files. References are released at completion/cancellation/
failure/timeout. Python does not guarantee cryptographic erasure of immutable
strings. Disable request/body tracing and crash/core dumps at deployment.
Only sanitized terminal reports enter the atomic, fsynced disk outbox. They
survive restart and never expire before website acknowledgement; capacity
backpressure stops new paid work if delivery falls behind. Acknowledged memory
records expire after 24 hours and can be evicted earlier to make capacity.
Outbox write failures stop readiness/submissions rather than silently accepting
more work. Polling can still retrieve an in-memory report for database delivery.

Restarting abandons running/queued jobs and discards their keys; the website
marks missing unfinished jobs failed. Users must re-enter a key to retry.
Restart does not rerun probes. Persist the `/data` volume across updates and
rollbacks. The website should background-poll unfinished jobs and ack terminal
ones after committing, independent of browser activity. Do not run two worker
processes or share the outbox between concurrent worker instances.

To update: publish corresponding source, run tests, build and smoke-test image,
record digest, drain active jobs, replace the single worker retaining `/data`,
then verify authenticated readiness. Roll back to the previous verified digest
with the same volume. Website release alone does not open this service.
