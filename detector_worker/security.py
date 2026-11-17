"""Connection-time DNS validation; httpcore retains the original TLS SNI.

Only the TCP backend substitutes the validated numeric address. The request
origin, Host header and httpcore TLS server_hostname remain the original host.
No connection retry, proxy, redirect or Unix socket is used. A system resolver
returning only benchmark-range fake IPs can use the fixed public DoH resolver;
neither fake nor private addresses are ever dialed.
"""
import asyncio
import ipaddress
import json
import re
import socket
from urllib.parse import urlsplit, urlunsplit

import httpcore
import httpx
from httpcore._backends.anyio import AnyIOBackend

SPECIAL_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    '192.0.0.0/24', '192.88.99.0/24',
    '64:ff9b::/96', '64:ff9b:1::/48', '2001::/23', '2002::/16',
))
FAKE_IP_NETWORK = ipaddress.ip_network('198.18.0.0/15')
DOH_TIMEOUT = 5


class UnsafeTarget(Exception):
    def __init__(self):
        super().__init__('outbound_policy')


def public_ip(value):
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        raise UnsafeTarget() from None
    if (not ip.is_global or ip.is_multicast or ip.is_reserved or
            any(ip.version == net.version and ip in net for net in SPECIAL_NETWORKS) or
            getattr(ip, 'ipv4_mapped', None) is not None or
            (ip.version == 6 and (ip.sixtofour is not None or ip.teredo is not None))):
        raise UnsafeTarget()
    return str(ip)


def validate_url(value):
    try:
        if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 33 for c in value) or '\\' in value:
            raise UnsafeTarget()
        p = urlsplit(value)
        if (p.scheme != 'https' or not p.hostname or p.username is not None or
                p.password is not None or '?' in value or '#' in value or
                p.port not in (None, 443) or '%' in p.netloc):
            raise UnsafeTarget()
        host = p.hostname.encode('idna').decode('ascii').lower()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if '.' not in host or host.endswith('.') or any(not (c.isalnum() or c in '.-') for c in host):
                raise UnsafeTarget()
        else:
            public_ip(host)
        authority = f'[{host}]' if ':' in host else host
        return urlunsplit(('https', authority, p.path.rstrip('/'), '', ''))
    except (ValueError, UnicodeError):
        raise UnsafeTarget() from None


async def resolve_public(host, port):
    rows = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(row[4][0] for row in rows))
    if not addresses:
        raise UnsafeTarget()
    # Narrow fake-IP mode recovery only. Mixed answers, local names and other
    # non-public ranges retain the ordinary fail-closed policy.
    try:
        fake_only = all(ipaddress.ip_address(address) in FAKE_IP_NETWORK for address in addresses)
    except ValueError:
        raise UnsafeTarget() from None
    if fake_only:
        addresses = await resolve_doh(host)
        if not addresses:
            raise UnsafeTarget()
    return [public_ip(address) for address in addresses]


async def resolve_doh(host, *, transport=None):
    """One bounded A lookup to a fixed numeric, certificate-verified resolver.

    Only the target hostname leaves this function; no target authentication,
    path or request body is forwarded. This transport deliberately does not use
    PublicBackend, preventing recursive fake-IP fallback for the resolver.
    """
    async def query():
        outbound = transport if transport is not None else httpx.AsyncHTTPTransport(
            retries=0, trust_env=False, verify=True,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0))
        async with httpx.AsyncClient(transport=outbound, trust_env=False,
                follow_redirects=False, timeout=DOH_TIMEOUT,
                headers={'accept': 'application/dns-json', 'accept-encoding': 'identity'}) as client:
            async with client.stream('GET', 'https://1.1.1.1/dns-query',
                    params={'name': host, 'type': 'A'}) as response:
                if response.status_code != 200 or response.headers.get('content-encoding', 'identity') != 'identity':
                    raise UnsafeTarget()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 16384:
                        raise UnsafeTarget()
                payload = json.loads(raw)
                if not isinstance(payload, dict) or type(payload.get('Status')) is not int or payload['Status'] != 0 or payload.get('TC'):
                    raise UnsafeTarget()
                answers = payload.get('Answer')
                if not isinstance(answers, list) or not 1 <= len(answers) <= 64:
                    raise UnsafeTarget()
                addresses = []
                for answer in answers:
                    if not isinstance(answer, dict) or type(answer.get('type')) is not int or not isinstance(answer.get('data'), str):
                        raise UnsafeTarget()
                    if answer['type'] == 5:  # CNAME; only returned A records are dialed
                        continue
                    if answer['type'] != 1:
                        raise UnsafeTarget()
                    address = public_ip(answer['data'])
                    if ipaddress.ip_address(address).version != 4:
                        raise UnsafeTarget()
                    addresses.append(address)
                if not addresses:
                    raise UnsafeTarget()
                return list(dict.fromkeys(addresses))
    try:
        return await asyncio.wait_for(query(), timeout=DOH_TIMEOUT)
    except Exception:
        raise UnsafeTarget() from None


class PublicBackend(AnyIOBackend):
    def __init__(self, resolver=resolve_public, delegate=None):
        self.resolver = resolver
        self.delegate = delegate or AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if port != 443 or local_address is not None:
            raise UnsafeTarget()
        addresses = await asyncio.wait_for(self.resolver(host, port), timeout=min(timeout or 10, 10))
        if not addresses:
            raise UnsafeTarget()
        pinned = [public_ip(a) for a in addresses]
        return await self.delegate.connect_tcp(pinned[0], port, timeout=timeout,
            local_address=None, socket_options=socket_options)

    async def connect_unix_socket(self, *args, **kwargs):
        raise UnsafeTarget()


class SafeTransport(httpx.AsyncBaseTransport):
    def __init__(self, base_url, *, inner=None, max_response=2_000_000,
                 max_requests=48, long_context=False, catalog_path=None):
        self.origin = httpx.URL(validate_url(base_url))
        self.max_response = max_response
        self.max_requests = max_requests
        self.long_context = long_context
        self.catalog_path = catalog_path
        self.requests = 0
        self.output_tokens = 0
        self.input_bytes = 0
        self.failed = False
        if inner is None:
            inner = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
            # Pinned httpx/httpcore versions: this is the one audited integration seam.
            inner._pool = httpcore.AsyncConnectionPool(
                network_backend=PublicBackend(), retries=0, max_connections=2,
                max_keepalive_connections=0, http1=True, http2=False)
        self.inner = inner

    async def handle_async_request(self, request):
        response = None
        try:
            if self.catalog_path is not None and (request.method != 'GET' or request.url.path != self.catalog_path):
                raise UnsafeTarget()
            if request.url.query and self.catalog_path is not None:
                pairs = request.url.params.multi_items()
                params = dict(pairs)
                if (len(pairs) != len(params) or set(params) not in ({'limit'}, {'limit', 'after_id'}) or
                        params.get('limit') != '1000' or
                        ('after_id' in params and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}', params['after_id']))):
                    raise UnsafeTarget()
                validated = httpx.URL(validate_url(str(request.url.copy_with(query=None))))
            else:
                validated = httpx.URL(validate_url(str(request.url)))
            if (validated.host, validated.port) != (self.origin.host, self.origin.port):
                raise UnsafeTarget()
            if request.headers.get('host') != request.url.netloc.decode():
                raise UnsafeTarget()
            self.requests += 1
            if self.failed or self.requests > self.max_requests:
                raise UnsafeTarget()
            raw = await request.aread()
            if self.catalog_path is not None and raw:
                raise UnsafeTarget()
            self.input_bytes += len(raw)
            if len(raw) > (1_500_000 if self.long_context else 150_000) or self.input_bytes > (4_000_000 if self.long_context else 1_000_000):
                raise UnsafeTarget()
            if raw:
                body = json.loads(raw)
                tokens = body.get('max_tokens', body.get('max_completion_tokens', 0))
                if not isinstance(tokens, int) or tokens < 0 or tokens > 16384:
                    raise UnsafeTarget()
                self.output_tokens += tokens
                if self.output_tokens > 65536:
                    raise UnsafeTarget()
            request.headers['accept-encoding'] = 'identity'
            # Upstream long-context request overrides cannot lift the per-request cap.
            request.extensions['timeout'] = dict(connect=10, read=30, write=30, pool=10)
            response = await self.inner.handle_async_request(request)
            if 300 <= response.status_code < 400:
                raise UnsafeTarget()
            allowed_catalog_auth = self.catalog_path is not None and response.status_code in (401, 403)
            if not allowed_catalog_auth and (response.status_code in (401, 402, 403, 408, 429) or response.status_code >= 500):
                self.failed = True
            if response.headers.get('content-encoding', 'identity') != 'identity':
                raise UnsafeTarget()
            result = httpx.Response(response.status_code, headers=response.headers,
                                    stream=BoundedStream(response, self), extensions=response.extensions)
            response = None  # bounded stream now owns closing the response
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            if response is not None:
                await response.aclose()

    async def aclose(self):
        await self.inner.aclose()


class BoundedStream(httpx.AsyncByteStream):
    def __init__(self, response, owner):
        self.response, self.owner = response, owner

    async def __aiter__(self):
        size = 0
        try:
            # Mock/custom transports may return an already buffered response.
            if self.response.is_stream_consumed:
                if len(self.response.content) > self.owner.max_response:
                    raise UnsafeTarget()
                yield self.response.content
                return
            async for chunk in self.response.aiter_raw():
                size += len(chunk)
                if size > self.owner.max_response:
                    raise UnsafeTarget()
                yield chunk
        except BaseException:
            self.owner.failed = True
            raise
        finally:
            await self.response.aclose()

    async def aclose(self):
        await self.response.aclose()
