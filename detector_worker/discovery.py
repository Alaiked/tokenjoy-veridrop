"""Transient, non-probing catalog fetches through the same outbound policy."""
import asyncio
import re
from urllib.parse import unquote

import httpx

from .security import SafeTransport, UnsafeTarget, validate_url

MODEL_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}\Z')
FAMILY_PATTERNS = (
    ('anthropic', re.compile(r'(?:anthropic\.)?claude(?:[-.:]|$)', re.I)),
    ('openai', re.compile(r'(?:gpt|chatgpt|codex)(?:[-.]|$)|o[134](?:[-.]|$)', re.I)),
    ('gemini', re.compile(r'gemini(?:[-.]|$)', re.I)),
)


class CatalogError(Exception):
    def __init__(self, code='model_catalog_unavailable', status=502):
        self.code, self.status = code, status
        super().__init__(code)


def validate_credentials(base_url, key):
    if not 1 <= len(key) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError('invalid_request')
    target = validate_url(base_url)
    decoded = target
    for _ in range(4):
        if key.casefold() in decoded.casefold():
            raise ValueError('invalid_request')
        decoded = unquote(decoded)
    return target


def model_family(ident):
    leaf = ident.rsplit('/', 1)[-1]
    for family, pattern in FAMILY_PATTERNS:
        if pattern.match(leaf):
            return family
    return None


class ModelCatalog:
    def __init__(self, *, transport_factory=None, concurrency=4, timeout=12):
        self.transport_factory = transport_factory or SafeTransport
        self.concurrency, self.timeout, self.active = concurrency, timeout, 0

    async def discover(self, base_url, key):
        try:
            target = validate_credentials(base_url, key)
        except (ValueError, UnsafeTarget):
            raise CatalogError('invalid_request', 422) from None
        if self.active >= self.concurrency:
            raise CatalogError('model_catalog_busy', 429)
        self.active += 1
        try:
            return await asyncio.wait_for(self._fetch(target, key), timeout=self.timeout)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise CatalogError('model_catalog_timeout', 504) from None
        except CatalogError:
            raise
        except Exception:
            raise CatalogError() from None
        finally:
            self.active -= 1

    async def _fetch(self, target, key):
        suffix = '/models' if target.endswith(('/v1', '/v1beta/openai')) else '/v1/models'
        endpoint = target + suffix
        transport = self.transport_factory(target, max_requests=8, max_response=1_000_000,
                                           catalog_path=httpx.URL(endpoint).path)
        async with httpx.AsyncClient(transport=transport, trust_env=False,
                follow_redirects=False, timeout=self.timeout,
                headers={'accept': 'application/json'}) as client:
            models, seen = [], set()
            cursor, cursors, native, total_rows, total_bytes = None, set(), False, 0, 0
            for attempt in range(8):
                remaining = 1_000_000 - total_bytes
                if remaining <= 0:
                    raise CatalogError()
                transport.max_response = remaining
                headers = ({'x-api-key': key, 'anthropic-version': '2023-06-01'} if native else
                           {'authorization': 'Bearer ' + key})
                params = {'limit': '1000'} if native or cursor is not None else None
                if cursor is not None:
                    params['after_id'] = cursor
                response = await client.get(endpoint, headers=headers, params=params)
                total_bytes += len(response.content)
                if total_bytes > 1_000_000:
                    raise CatalogError()
                if attempt == 0 and response.status_code in (401, 403):
                    native = True
                    continue
                if response.status_code != 200:
                    raise CatalogError()
                payload = response.json()
                rows = payload.get('data') if isinstance(payload, dict) else None
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise CatalogError()
                total_rows += len(rows)
                if total_rows > 2048 or ('has_more' in payload and type(payload['has_more']) is not bool):
                    raise CatalogError()
                for row in rows:
                    ident = row.get('id')
                    if not isinstance(ident, str) or not MODEL_PATTERN.fullmatch(ident) or key.casefold() in ident.casefold() or ident in seen:
                        continue
                    family = model_family(ident)
                    if family is None:
                        continue
                    seen.add(ident)
                    models.append({'id': ident, 'family': family})
                    if len(models) > 512:
                        raise CatalogError()
                if not payload.get('has_more', False):
                    return models
                cursor = payload.get('last_id')
                if (not rows or not isinstance(cursor, str) or not MODEL_PATTERN.fullmatch(cursor) or
                        key.casefold() in cursor.casefold() or cursor in cursors or rows[-1].get('id') != cursor):
                    raise CatalogError()
                cursors.add(cursor)
            # Never publish a partial catalog when pagination exceeds the budget.
            raise CatalogError()
