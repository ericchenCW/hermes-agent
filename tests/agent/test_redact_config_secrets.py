"""Redaction backstop for config-shaped secrets in file content and replies.

``redact_sensitive_text(file_read=True)`` implies ``code_file=True``, which
skips the ENV/JSON/YAML assignment passes to protect source code. A config
file returned verbatim by ``read_file`` therefore kept ``api_key: …`` /
``bot_secret: …`` values in the clear whenever the value carried no known
vendor prefix. These tests pin the narrow key/value fallback that closes it,
plus the standalone ``Bearer`` and opaque-blob rules.
"""

import pytest

from agent.redact import redact_sensitive_text


class TestKeyValueSecrets:
    @pytest.mark.parametrize("text,expected", [
        ("bot_secret: supersecretvalue", "bot_secret: ***"),
        ("botSecret: supersecretvalue", "botSecret: ***"),
        ("api_key=abc123def456ghi", "api_key=***"),
        ("apikey: abc123def456ghi", "apikey: ***"),
        ("password: hunter2", "password: ***"),
        ("passwd = hunter2", "passwd = ***"),
        ("credential: abc123def456", "credential: ***"),
        ("authorization: abc123def456", "authorization: ***"),
        ('  "token": "abc123def456"', '  "token": "***"'),
        ("secret: abc123def456", "secret: ***"),
    ])
    def test_masked_in_file_content(self, text, expected):
        assert redact_sensitive_text(text, file_read=True) == expected

    def test_authorization_bearer_header(self):
        out = redact_sensitive_text(
            "Authorization: Bearer abcdefghijklmnopqrstuvwx", file_read=True
        )
        assert "abcdefghijklmnopqrstuvwx" not in out
        assert out == "Authorization: Bearer ***"

    def test_standalone_bearer_token(self):
        out = redact_sensitive_text("use Bearer abcdefghijklmnopqrst to call it")
        assert "abcdefghijklmnopqrst" not in out
        assert out == "use Bearer *** to call it"

    def test_realistic_config_block(self):
        raw = (
            "model_endpoint: http://10.10.24.11:3000/v1\n"
            "api_key: hermes-local-abcdefghijklmnop\n"
            "wecom:\n"
            "  corpid: ww1234567890abcdef\n"
            "  bot_secret: Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MA\n"
        )
        out = redact_sensitive_text(raw, file_read=True)
        assert "hermes-local-abcdefghijklmnop" not in out
        assert "Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MA" not in out
        # WeCom corp ids are public identifiers needed for support answers.
        assert "ww1234567890abcdef" in out


class TestNoCollateralDamage:
    """Ordinary knowledge-base prose must survive untouched."""

    @pytest.mark.parametrize("text", [
        "密码策略：请每 90 天更换一次密码。",
        "## Token 管理\n本节介绍令牌的申请流程。",
        "Secretary: J. Smith",
        "tokenizer: cl100k_base",
        "错误提示 error: token expired，请重新登录。",
        "登录地址 http://portal.weops.proc/login",
        "sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "commit 0123456789abcdef0123456789abcdef01234567",
    ])
    def test_unchanged(self, text):
        assert redact_sensitive_text(text, file_read=True) == text

    def test_source_code_unaffected_by_new_pass(self):
        # code_file=True (without file_read) must not gain the new pass.
        code = 'MAX_TOKENS = 4096\nAPI_KEY_ENV = "OPENAI_API_KEY"\n'
        assert redact_sensitive_text(code, code_file=True) == code


class TestOpaqueTokens:
    def test_long_hex_masked_in_file_content(self):
        raw = "runtime_ref: 0123456789abcdef0123456789abcdef0123456789abcdef"
        assert redact_sensitive_text(raw, file_read=True).endswith("***")

    def test_wecom_id_not_masked(self):
        raw = "id: wo0123456789abcdef0123456789abcdef0123456789"
        assert redact_sensitive_text(raw, file_read=True) == raw


class TestFinalReplyIsRedacted:
    """The final user-visible reply passes through the same function."""

    def test_turn_finalizer_masks_quoted_secret(self, monkeypatch):
        import agent.turn_finalizer as tf

        secret = "sk-abcdefghijklmnopqrstuvwxyz012345"
        assert secret not in redact_sensitive_text(secret)

    def test_finalizer_calls_redactor(self):
        import inspect

        import agent.turn_finalizer as tf

        source = inspect.getsource(tf.finalize_turn)
        assert "redact_sensitive_text" in source
