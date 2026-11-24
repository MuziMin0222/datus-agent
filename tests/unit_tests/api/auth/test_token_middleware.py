# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the opt-in bearer-token gate."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datus.api.auth.token_middleware import (
    TOKEN_ENV,
    BearerTokenMiddleware,
    configured_token,
    install_token_auth,
)


def _app_with_auth(monkeypatch, token: str | None) -> FastAPI:
    if token is None:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(TOKEN_ENV, token)
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"ok": True}

    @app.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    install_token_auth(app)
    return app


class TestConfiguredToken:
    def test_missing_env_disables_auth(self, monkeypatch):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        assert configured_token() == ""

    def test_blank_env_disables_auth(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "   ")
        assert configured_token() == ""

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "  secret  ")
        assert configured_token() == "secret"


class TestInstallTokenAuth:
    def test_noop_when_unset(self, monkeypatch):
        app = _app_with_auth(monkeypatch, None)
        client = TestClient(app)
        assert client.get("/").status_code == 200

    def test_rejects_missing_header(self, monkeypatch):
        app = _app_with_auth(monkeypatch, "secret")
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_rejects_wrong_token(self, monkeypatch):
        app = _app_with_auth(monkeypatch, "secret")
        client = TestClient(app)
        response = client.get("/", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_rejects_non_bearer_scheme(self, monkeypatch):
        app = _app_with_auth(monkeypatch, "secret")
        client = TestClient(app)
        response = client.get("/", headers={"Authorization": "Basic secret"})
        assert response.status_code == 401

    def test_accepts_matching_token(self, monkeypatch):
        app = _app_with_auth(monkeypatch, "secret")
        client = TestClient(app)
        response = client.get("/", headers={"Authorization": "bearer secret"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    @pytest.mark.parametrize("path", ["/health", "/health/live"])
    def test_health_probes_are_exempt(self, monkeypatch, path):
        app = FastAPI()

        @app.get(path)
        async def probe():
            return {"status": "ok"}

        monkeypatch.setenv(TOKEN_ENV, "secret")
        install_token_auth(app)
        client = TestClient(app)
        assert client.get(path).status_code == 200

    def test_cors_preflight_is_exempt(self, monkeypatch):
        app = _app_with_auth(monkeypatch, "secret")
        client = TestClient(app)
        response = client.options("/", headers={"Origin": "app://datus"})
        assert response.status_code == 405  # no OPTIONS handler, but not 401


class TestBearerTokenMiddleware:
    def test_empty_token_is_rejected(self):
        with pytest.raises(ValueError):
            BearerTokenMiddleware(FastAPI(), token="")

    def test_accepts_punctuation_heavy_token(self, monkeypatch):
        token = "s3cr3t.token-value_1"
        app = _app_with_auth(monkeypatch, token)
        client = TestClient(app)
        assert client.get("/", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert client.get("/", headers={"Authorization": "Bearer s3cr3t.other"}).status_code == 401
