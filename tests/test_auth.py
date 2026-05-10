"""Tests for Kerberos/SPNEGO authentication."""

import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_zuul.auth import _extract_oidc_params, _follow_redirect


class TestFollowRedirect:
    def test_returns_location_for_302(self):
        resp = httpx.Response(302, headers={"location": "https://sso.example.com/login"})
        assert _follow_redirect(resp) == "https://sso.example.com/login"

    def test_returns_location_for_301(self):
        resp = httpx.Response(301, headers={"location": "https://new.example.com/"})
        assert _follow_redirect(resp) == "https://new.example.com/"

    def test_returns_location_for_307(self):
        resp = httpx.Response(307, headers={"location": "https://temp.example.com/"})
        assert _follow_redirect(resp) == "https://temp.example.com/"

    def test_returns_location_for_308(self):
        resp = httpx.Response(308, headers={"location": "https://perm.example.com/"})
        assert _follow_redirect(resp) == "https://perm.example.com/"

    def test_returns_none_for_200(self):
        resp = httpx.Response(200)
        assert _follow_redirect(resp) is None

    def test_returns_none_for_404(self):
        resp = httpx.Response(404)
        assert _follow_redirect(resp) is None

    def test_returns_none_for_401(self):
        resp = httpx.Response(401)
        assert _follow_redirect(resp) is None

    def test_raises_when_no_location_header(self):
        resp = httpx.Response(302, headers={})
        with pytest.raises(RuntimeError, match="no Location header"):
            _follow_redirect(resp)


@pytest.fixture
def mock_gssapi():
    """Inject a mock gssapi module into sys.modules."""
    mock_mod = MagicMock()
    mock_mod.NameType.hostbased_service = "hostbased"
    mock_mod.exceptions.GSSError = type("GSSError", (Exception,), {})
    original = sys.modules.get("gssapi")
    sys.modules["gssapi"] = mock_mod
    yield mock_mod
    if original is not None:
        sys.modules["gssapi"] = original
    else:
        del sys.modules["gssapi"]


class TestKerberosAuth:
    async def test_successful_auth(self, mock_gssapi):
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"spnego-token-bytes"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": "https://sso.example.com/auth"}),
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(
                    302,
                    headers={"location": "https://zuul.example.com/callback?code=abc"},
                ),
                httpx.Response(200),
                httpx.Response(200),  # verification GET
            ]
        )

        await kerberos_auth(client, "https://zuul.example.com")

        # Verify SPNEGO token was sent in the auth request
        calls = client.get.call_args_list
        auth_call = calls[2]
        auth_header = auth_call.kwargs.get("headers", {}).get("Authorization", "")
        expected_token = base64.b64encode(b"spnego-token-bytes").decode()
        assert auth_header == f"Negotiate {expected_token}"

    @patch("mcp_zuul.auth.secrets.token_urlsafe", return_value="fixed-state")
    async def test_phase1_then_phase2_jwt(self, _mock_secrets, mock_gssapi):
        """Full flow: OIDC redirect → SPNEGO → session → JWT acquisition."""
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"spnego-token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        oidc_url = (
            "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
            "?client_id=zuul&redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
        )

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(302, headers={"location": oidc_url})
            if call_count == 2:
                return httpx.Response(401, headers={"www-authenticate": "Negotiate"})
            if call_count == 3:
                return httpx.Response(
                    302,
                    headers={"location": "https://zuul.example.com/callback?code=session"},
                )
            if call_count == 4:
                return httpx.Response(200)
            # Phase 2: JWT authorize URL → redirect with code (state must match)
            if call_count == 5:
                return httpx.Response(
                    302,
                    headers={
                        "location": (
                            "https://zuul.example.com/callback?code=jwt-code&state=fixed-state"
                        )
                    },
                )
            # Verification GET after auth
            return httpx.Response(200)

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(side_effect=mock_get)
        client.post = AsyncMock(
            return_value=httpx.Response(200, json={"access_token": "the-jwt", "expires_in": 300})
        )

        await kerberos_auth(client, "https://zuul.example.com")

        assert client.headers["authorization"] == "Bearer the-jwt"
        assert client.post.call_count == 1
        token_call = client.post.call_args
        assert "openid-connect/token" in token_call.args[0]
        assert token_call.kwargs["data"]["code"] == "jwt-code"

    async def test_200_treated_as_already_authed(self, mock_gssapi):
        """If server returns 200 (session valid after cookie clear), accept it."""
        from mcp_zuul.auth import kerberos_auth

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(200),  # initial GET: session valid
                httpx.Response(200),  # verification GET
            ]
        )

        await kerberos_auth(client, "https://zuul.example.com")

    async def test_200_early_return_jwt_crash_still_verifies(self, mock_gssapi):
        """JWT crash on the 200 early-return path doesn't kill auth."""
        from mcp_zuul.auth import kerberos_auth

        oidc_url = (
            "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
            "?client_id=zuul&redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": oidc_url}),
                httpx.Response(200),  # SSO 200 → early-return path with oidc_params
                # _acquire_admin_jwt: malformed 302 → crash
                httpx.Response(302),
                httpx.Response(200),  # verification GET: session valid
            ]
        )

        await kerberos_auth(client, "https://zuul.example.com")
        assert "authorization" not in client.headers

    async def test_200_from_intermediate_redirect_caught_by_verification(self, mock_gssapi):
        """If redirect chain ends at 200 (not from Zuul), verification catches it."""
        from mcp_zuul.auth import kerberos_auth

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": "https://sso.example.com/login"}),
                httpx.Response(200),  # SSO returned 200 (cached session, HTML page)
                httpx.Response(302),  # verification GET: Zuul redirects (no valid session)
            ]
        )

        with pytest.raises(RuntimeError, match=r"session verification failed.*302"):
            await kerberos_auth(client, "https://zuul.example.com")

    async def test_unexpected_status_raises(self, mock_gssapi):
        """Non-200, non-401 status raises RuntimeError."""
        from mcp_zuul.auth import kerberos_auth

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=httpx.Response(403))

        with pytest.raises(RuntimeError, match=r"expected 401 Negotiate.*got 403"):
            await kerberos_auth(client, "https://zuul.example.com")

    async def test_wrong_auth_scheme(self, mock_gssapi):
        from mcp_zuul.auth import kerberos_auth

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(
            return_value=httpx.Response(401, headers={"www-authenticate": "Basic realm=test"})
        )

        with pytest.raises(RuntimeError, match="did not offer Negotiate"):
            await kerberos_auth(client, "https://zuul.example.com")

    async def test_spnego_failure(self, mock_gssapi):
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.side_effect = mock_gssapi.exceptions.GSSError("no ticket")
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(
            return_value=httpx.Response(401, headers={"www-authenticate": "Negotiate"})
        )

        with pytest.raises(RuntimeError, match="SPNEGO token generation failed"):
            await kerberos_auth(client, "https://zuul.example.com")

    async def test_final_response_not_200(self, mock_gssapi):
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(403),
            ]
        )

        with pytest.raises(RuntimeError, match="final response was 403"):
            await kerberos_auth(client, "https://zuul.example.com")

    async def test_clears_cookies_before_auth(self, mock_gssapi):
        """Stale session cookies are cleared so the OIDC chain starts fresh."""
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.cookies = MagicMock()
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(200),
                httpx.Response(200),  # verification GET
            ]
        )

        await kerberos_auth(client, "https://zuul.example.com")
        client.cookies.clear.assert_called_once()

    async def test_verification_get_failure_raises(self, mock_gssapi):
        """Session verification after auth detects stale sessions."""
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.cookies = MagicMock()
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(200),  # SPNEGO accepted
                httpx.Response(302),  # verification GET: session not valid
            ]
        )

        with pytest.raises(RuntimeError, match=r"session verification failed.*302"):
            await kerberos_auth(client, "https://zuul.example.com")

    @patch("mcp_zuul.auth.secrets.token_urlsafe", return_value="fixed-state")
    async def test_jwt_crash_does_not_kill_session_auth(self, _mock_secrets, mock_gssapi):
        """JWT acquisition failure should not prevent session-based auth."""
        from mcp_zuul.auth import kerberos_auth

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        oidc_url = (
            "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
            "?client_id=zuul&redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
        )
        client = AsyncMock(spec=httpx.AsyncClient)
        client.cookies = MagicMock()
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": oidc_url}),
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(200),  # SPNEGO accepted
                # _acquire_admin_jwt: SSO returns malformed 302 (no Location)
                httpx.Response(302),
                httpx.Response(200),  # verification GET: session is valid
            ]
        )

        await kerberos_auth(client, "https://zuul.example.com")
        assert "authorization" not in client.headers


class TestExtractOidcParams:
    """Tests for _extract_oidc_params URL parsing."""

    def test_extracts_from_valid_oidc_url(self):
        url = (
            "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
            "?client_id=zuul&redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
            "&response_type=code&scope=openid"
        )
        result = _extract_oidc_params(url)
        assert result is not None
        client_id, redirect_uri, token_url, authorize_url = result
        assert client_id == "zuul"
        assert redirect_uri == "https://zuul.example.com/callback"
        assert token_url == "https://sso.example.com/realms/zuul/protocol/openid-connect/token"
        assert authorize_url == "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"

    def test_returns_none_for_non_oidc_url(self):
        assert _extract_oidc_params("https://zuul.example.com/api/tenants") is None

    def test_returns_none_without_client_id(self):
        url = (
            "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
            "?redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
        )
        assert _extract_oidc_params(url) is None

    def test_returns_none_without_redirect_uri(self):
        url = "https://sso.example.com/realms/zuul/protocol/openid-connect/auth?client_id=zuul"
        assert _extract_oidc_params(url) is None


class TestAcquireAdminJwt:
    """Tests for _acquire_admin_jwt JWT acquisition flow."""

    _OIDC_URL = (
        "https://sso.example.com/realms/zuul/protocol/openid-connect/auth"
        "?client_id=zuul&redirect_uri=https%3A%2F%2Fzuul.example.com%2Fcallback"
    )
    _OIDC_PARAMS = _extract_oidc_params(_OIDC_URL)
    assert _OIDC_PARAMS is not None
    _CLIENT_ID, _REDIRECT_URI, _TOKEN_URL, _AUTHORIZE_URL = _OIDC_PARAMS

    _FIXED_STATE = "fixed-state"
    _SECRETS_PATCH = patch("mcp_zuul.auth.secrets.token_urlsafe", return_value=_FIXED_STATE)

    @_SECRETS_PATCH
    async def test_acquires_jwt_and_sets_header(self, _mock_secrets, mock_gssapi):
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        callback_url = (
            f"https://zuul.example.com/callback?code=auth-code-123&state={self._FIXED_STATE}"
        )
        client.get = AsyncMock(return_value=httpx.Response(302, headers={"location": callback_url}))
        client.post = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"access_token": "jwt-token-abc", "expires_in": 300},
            )
        )

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert client.headers["authorization"] == "Bearer jwt-token-abc"
        post_call = client.post.call_args
        assert post_call.args[0] == self._TOKEN_URL
        assert post_call.kwargs["data"]["code"] == "auth-code-123"
        assert post_call.kwargs["data"]["grant_type"] == "authorization_code"

    async def test_warns_when_no_code_captured(self, mock_gssapi):
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(return_value=httpx.Response(200))

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers

    @_SECRETS_PATCH
    async def test_warns_on_token_endpoint_error(self, _mock_secrets, mock_gssapi):
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            return_value=httpx.Response(
                302,
                headers={
                    "location": (
                        f"https://zuul.example.com/callback?code=c1&state={self._FIXED_STATE}"
                    )
                },
            )
        )
        client.post = AsyncMock(return_value=httpx.Response(500, text="Internal Server Error"))

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers

    @_SECRETS_PATCH
    async def test_warns_on_non_json_token_response(self, _mock_secrets, mock_gssapi):
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            return_value=httpx.Response(
                302,
                headers={
                    "location": (
                        f"https://zuul.example.com/callback?code=c1&state={self._FIXED_STATE}"
                    )
                },
            )
        )
        client.post = AsyncMock(return_value=httpx.Response(200, text="not json"))

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers

    @_SECRETS_PATCH
    async def test_warns_on_missing_access_token(self, _mock_secrets, mock_gssapi):
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            return_value=httpx.Response(
                302,
                headers={
                    "location": (
                        f"https://zuul.example.com/callback?code=c1&state={self._FIXED_STATE}"
                    )
                },
            )
        )
        client.post = AsyncMock(return_value=httpx.Response(200, json={"token_type": "Bearer"}))

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers

    @_SECRETS_PATCH
    async def test_handles_spnego_renegotiate_in_jwt_flow(self, _mock_secrets, mock_gssapi):
        """SSO requests Kerberos re-negotiate during JWT acquisition."""
        from mcp_zuul.auth import _acquire_admin_jwt

        mock_ctx = MagicMock()
        mock_ctx.step.return_value = b"spnego-token"
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": "https://sso.example.com/negotiate"}),
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
                httpx.Response(
                    302,
                    headers={
                        "location": (
                            "https://zuul.example.com/callback"
                            f"?code=renegotiated&state={self._FIXED_STATE}"
                        )
                    },
                ),
            ]
        )
        client.post = AsyncMock(
            return_value=httpx.Response(
                200, json={"access_token": "jwt-after-renego", "expires_in": 600}
            )
        )

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert client.headers["authorization"] == "Bearer jwt-after-renego"

    async def test_spnego_failure_in_jwt_flow_degrades_gracefully(self, mock_gssapi):
        """GSSError during JWT re-negotiate logs warning and returns."""
        from mcp_zuul.auth import _acquire_admin_jwt

        mock_ctx = MagicMock()
        mock_ctx.step.side_effect = mock_gssapi.exceptions.GSSError("expired")
        mock_gssapi.SecurityContext.return_value = mock_ctx

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            side_effect=[
                httpx.Response(302, headers={"location": "https://sso.example.com/negotiate"}),
                httpx.Response(401, headers={"www-authenticate": "Negotiate"}),
            ]
        )

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers
        client.post.assert_not_called()

    @_SECRETS_PATCH
    async def test_state_mismatch_rejects_code(self, _mock_secrets, mock_gssapi):
        """OIDC state mismatch aborts JWT acquisition (CSRF protection)."""
        from mcp_zuul.auth import _acquire_admin_jwt

        client = AsyncMock(spec=httpx.AsyncClient)
        client.headers = {}
        client.get = AsyncMock(
            return_value=httpx.Response(
                302,
                headers={
                    "location": "https://zuul.example.com/callback?code=stolen&state=wrong-state"
                },
            )
        )

        await _acquire_admin_jwt(
            client, self._CLIENT_ID, self._REDIRECT_URI, self._TOKEN_URL, self._AUTHORIZE_URL
        )

        assert "authorization" not in client.headers
        client.post.assert_not_called()
