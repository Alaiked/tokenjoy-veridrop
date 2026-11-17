"""Authenticated single-process in-memory jobs. Never run multiple ASGI workers."""
import asyncio
import hmac
import json
import os
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from .security import UnsafeTarget, validate_url


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    protocol: Literal['anthropic', 'openai', 'gemini']
    base_url: str = Field(max_length=2048)
    api_key: SecretStr
    model: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$')
    long_context: bool = False


class Jobs:
    def __init__(self, engine, concurrency, capacity, timeout, retention, max_records, report_dir):
        self.engine, self.capacity, self.timeout = engine, capacity, timeout
        self.retention, self.max_records = retention, max_records
        self.records, self.tasks, self.secrets, self.expiry, self.scopes = {}, {}, {}, {}, {}
        self.semaphore = asyncio.Semaphore(concurrency)
        self.report_dir = Path(report_dir) if report_dir else None
        self.unacked = set()
        self.storage_failed = False
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            for path in self.report_dir.glob('*.json'):
                if len(self.records) >= max_records or path.stat().st_size > 1_000_000:
                    raise RuntimeError('outbox_capacity_or_corruption')
                record = json.loads(path.read_text(encoding='utf-8'))
                ident = record['id']
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', ident) or path.stem != ident or record['state'] not in ('done', 'error', 'cancelled'):
                    raise RuntimeError('outbox_corruption')
                self.records[ident] = record
                self.scopes[ident] = record['long_context']
                self.unacked.add(ident)

    def persist_terminal(self, ident):
        if self.report_dir:
            temporary = self.report_dir / (ident + '.tmp')
            target = self.report_dir / (ident + '.json')
            encoded = json.dumps(self.records[ident], allow_nan=False).encode('utf-8')
            if len(encoded) > 1_000_000:
                raise RuntimeError('outbox_record_limit')
            with temporary.open('wb') as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
            # Persist directory entry as well on the production Linux runtime.
            if os.name == 'posix':
                descriptor = os.open(self.report_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            self.unacked.add(ident)

    def prune(self, make_space=False):
        for ident, expires in list(self.expiry.items()):
            if expires <= time.monotonic() and ident not in self.unacked:
                self.expiry.pop(ident, None)
                self.records.pop(ident, None)
                self.scopes.pop(ident, None)
        if make_space:
            for ident in list(self.records):
                if len(self.records) < self.max_records:
                    break
                if ident not in self.unacked and self.records[ident]['state'] in ('done', 'error', 'cancelled'):
                    self.records.pop(ident, None)
                    self.scopes.pop(ident, None)
                    self.expiry.pop(ident, None)

    async def run(self, ident, spec):
        key = None
        try:
            async with self.semaphore:
                self.records[ident]['state'] = 'running'
                key = self.secrets.pop(ident)
                report = await asyncio.wait_for(self.engine(spec, key), self.timeout)
                self.records[ident].update(state='done', report=report)
        except asyncio.CancelledError:
            self.records[ident]['state'] = 'cancelled'
        except asyncio.TimeoutError:
            self.records[ident].update(state='error', error_code='timeout')
        except Exception:
            self.records[ident].update(state='error', error_code='upstream_error')
        finally:
            key = None
            self.secrets.pop(ident, None)
            self.records[ident]['finished_at'] = now()
            try:
                self.persist_terminal(ident)
            except Exception:
                # Stop accepting billable work if durable delivery cannot be assured.
                self.storage_failed = True
            self.expiry[ident] = time.monotonic() + self.retention
            self.tasks.pop(ident, None)


def create_app(*, service_token=None, engine=None, concurrency=2, capacity=12,
               job_timeout=240, retention=86400, max_records=2048, report_dir=None):
    token = service_token if service_token is not None else os.environ.get('DETECTOR_SERVICE_TOKEN', '')
    if len(token) < 16:
        raise RuntimeError('DETECTOR_SERVICE_TOKEN must contain at least 16 characters')
    if engine is None:
        from .engine import run_detection
        engine = run_detection
    jobs = Jobs(engine, concurrency, capacity, job_timeout, retention, max_records,
                report_dir if report_dir is not None else os.environ.get('DETECTOR_REPORT_DIR'))

    @asynccontextmanager
    async def lifespan(app):
        yield
        tasks = list(jobs.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        jobs.secrets.clear()

    app = FastAPI(title='TokenJoy Veridrop worker', docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({'detail': 'invalid_request'}, status_code=422)

    @app.middleware('http')
    async def authentication(request, call_next):
        if request.url.path != '/healthz':
            supplied = request.headers.get('X-Detector-Token', '')
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return JSONResponse({'detail': 'unauthorized'}, status_code=401)
        return await call_next(request)

    @app.get('/healthz')
    async def health():
        return {'status': 'ok'}

    @app.get('/readyz')
    async def ready():
        from .engine import ENGINE_VERSION
        jobs.prune(make_space=True)
        return {'ready': not jobs.storage_failed and len(jobs.records) < jobs.max_records,
                'engine_version': ENGINE_VERSION}

    @app.post('/jobs', status_code=202)
    async def submit(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 16384:
                raise HTTPException(413, 'request_too_large')
        try:
            data = Submission.model_validate_json(raw)
            key = data.api_key.get_secret_value()
            if not 1 <= len(key) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
                raise ValueError()
            target = validate_url(data.base_url)
            if key in target or key in data.model or (data.protocol == 'gemini' and data.long_context):
                raise ValueError()
        except (ValidationError, UnsafeTarget, ValueError):
            raise HTTPException(422, 'invalid_request') from None
        spec = data.model_dump(exclude={'api_key'})
        spec['base_url'] = target
        jobs.prune()
        if data.id in jobs.records:
            previous = jobs.records[data.id]
            if any(previous[k] != spec[k] for k in ('protocol', 'model', 'base_url')) or jobs.scopes[data.id] != data.long_context:
                raise HTTPException(409, 'id_conflict')
            return previous
        if jobs.storage_failed:
            raise HTTPException(503, 'storage_unavailable')
        jobs.prune(make_space=True)
        if len(jobs.tasks) >= jobs.capacity or len(jobs.records) >= jobs.max_records:
            raise HTTPException(429, 'capacity_exceeded')
        jobs.records[data.id] = {k: spec[k] for k in ('id', 'protocol', 'base_url', 'model', 'long_context')}
        jobs.records[data.id].update(state='queued', created_at=now())
        jobs.scopes[data.id] = data.long_context
        jobs.secrets[data.id] = key
        jobs.tasks[data.id] = asyncio.create_task(jobs.run(data.id, spec))
        return jobs.records[data.id]

    @app.get('/jobs/{ident}')
    async def get_job(ident: str):
        jobs.prune()
        if ident not in jobs.records:
            raise HTTPException(404, 'job_not_found')
        return jobs.records[ident]

    @app.delete('/jobs/{ident}')
    async def cancel_job(ident: str):
        job = await get_job(ident)
        task = jobs.tasks.get(ident)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            jobs.secrets.pop(ident, None)
            jobs.tasks.pop(ident, None)
            job.update(state='cancelled', finished_at=now())
            try:
                jobs.persist_terminal(ident)
            except Exception:
                jobs.storage_failed = True
            jobs.expiry[ident] = time.monotonic() + jobs.retention
        return job

    @app.post('/jobs/{ident}/ack')
    async def acknowledge(ident: str):
        job = await get_job(ident)
        if job['state'] not in ('done', 'error', 'cancelled'):
            raise HTTPException(409, 'job_not_terminal')
        if jobs.report_dir:
            (jobs.report_dir / (ident + '.json')).unlink(missing_ok=True)
        jobs.unacked.discard(ident)
        jobs.expiry[ident] = time.monotonic() + jobs.retention
        return {'acknowledged': True}

    return app
