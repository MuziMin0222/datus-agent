# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Opt-in bearer-token gate for the HTTP API.

The desktop client connects to a remote ``datus-api`` over plain HTTP+SSE.
Setting ``DATUS_API_TOKEN`` requires every request (except health probes and
CORS preflight) to carry ``Authorization: Bearer <token>``. When the variable
is unset or blank the middleware is not installed and the API keeps its
historical open behaviour for loopback-only deployments.
"""

import hmac
import os
from typing import FrozenSet, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

TOKEN_ENV = "DATUS_API_TOKEN"

#: Probes and preflight requests must stay reachable without credentials.
EXEMPT_PATHS: FrozenSet[str] = frozenset({"/health", "/health/live"})


def configured_token() -> str:
    """Return the configured token, or an empty string when auth is disabled."""
    return (os.getenv(TOKEN_ENV) or "").strip()


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject unauthenticated requests with 401 when a token is configured."""

    def __init__(self, app, token: str, exempt_paths: Iterable[str] = EXEMPT_PATHS):
        super().__init__(app)
        if not token:
            raise ValueError("BearerTokenMiddleware requires a non-empty token")
        self._token = token
        self._exempt_paths = frozenset(exempt_paths)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS" or request.url.path in self._exempt_paths:
            return await call_next(request)

        scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(credentials.strip(), self._token):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def install_token_auth(app: FastAPI) -> bool:
    """Install :class:`BearerTokenMiddleware` when ``DATUS_API_TOKEN`` is set."""
    token = configured_token()
    if not token:
        return False
    app.add_middleware(BearerTokenMiddleware, token=token)
    return True
