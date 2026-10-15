from __future__ import annotations

import asyncio
from uuid import UUID

import jwt
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings

_bearer = HTTPBearer(auto_error=False)


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jwks: jwt.PyJWKClient | None = None
        if settings.supabase_url:
            self._jwks = jwt.PyJWKClient(
                f"{settings.supabase_url}/auth/v1/.well-known/jwks.json",
                cache_keys=True,
            )

    async def authenticate(self, token: str) -> UUID:
        if self.settings.dev_auth and token.startswith("dev:"):
            try:
                user_id = UUID(token.removeprefix("dev:"))
            except ValueError as exc:
                raise self._unauthorized() from exc
            if user_id != self.settings.dev_user_id:
                raise self._unauthorized()
            return user_id

        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")
            if algorithm == "HS256":
                if not self.settings.supabase_jwt_secret:
                    raise self._unauthorized()
                key: object = self.settings.supabase_jwt_secret
            elif algorithm in {"RS256", "ES256"}:
                if not self._jwks:
                    raise self._unauthorized()
                signing_key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
                key = signing_key.key
            else:
                raise self._unauthorized()

            options = {"require": ["exp", "sub", "aud"]}
            payload = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=self.settings.jwt_audience,
                issuer=self.settings.jwt_issuer,
                options=options,
            )
            return UUID(payload["sub"])
        except HTTPException:
            raise
        except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
            raise self._unauthorized() from exc

    @staticmethod
    def _unauthorized() -> HTTPException:
        return HTTPException(
            status_code=401,
            detail={
                "code": "authentication_required",
                "message": "A valid Supabase access token is required",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


async def current_user(request: Request) -> UUID:
    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if not credentials or credentials.scheme.lower() != "bearer":
        raise Authenticator._unauthorized()
    authenticator: Authenticator = request.app.state.authenticator
    return await authenticator.authenticate(credentials.credentials)
