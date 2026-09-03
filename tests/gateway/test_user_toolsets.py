"""Per-user toolset override (idcsre): user_toolsets.<platform>.<user_id> in config
or <PLATFORM>_ADMIN_USERS / <PLATFORM>_ADMIN_TOOLSETS in the environment."""
from gateway.run import GatewayRunner


class _Src:
    def __init__(self, chat_id, user_id=None):
        self.chat_id = chat_id
        self.user_id = user_id


def _runner():
    gr = object.__new__(GatewayRunner)
    gr._adapter_for_source = lambda source: None
    return gr


CFG = {
    "platform_toolsets": {"wecom": ["sre-readonly"]},
    "user_toolsets": {"wecom": {"admin-1": ["sre"], "admin-2": "sre"}},
}


def _resolve(cfg, src):
    return GatewayRunner._resolve_enabled_toolsets_for_source(_runner(), cfg, src, "wecom")


def test_default_user_gets_readonly_posture(monkeypatch):
    monkeypatch.delenv("WECOM_ADMIN_USERS", raising=False)
    tools = _resolve(CFG, _Src("wecom:u9", "u9"))
    assert "sre-readonly" in tools
    assert not {"sre", "terminal", "file", "skills", "memory"} & set(tools)


def test_config_admin_gets_full_posture(monkeypatch):
    monkeypatch.delenv("WECOM_ADMIN_USERS", raising=False)
    for uid in ("admin-1", "admin-2"):
        tools = _resolve(CFG, _Src("wecom:" + uid, uid))
        assert {"sre", "terminal", "file", "skills", "memory"} <= set(tools) and "sre-readonly" not in tools


def test_env_admin_list(monkeypatch):
    monkeypatch.setenv("WECOM_ADMIN_USERS", "env-a, env-b")
    monkeypatch.delenv("WECOM_ADMIN_TOOLSETS", raising=False)
    cfg = {"platform_toolsets": {"wecom": ["sre-readonly"]}}
    assert "terminal" in _resolve(cfg, _Src("wecom:env-a", "env-a"))
    assert "terminal" not in _resolve(cfg, _Src("wecom:other", "other"))
    monkeypatch.setenv("WECOM_ADMIN_TOOLSETS", "sre-readonly")
    assert "terminal" not in _resolve(cfg, _Src("wecom:env-b", "env-b"))


def test_missing_user_id_falls_back_to_platform(monkeypatch):
    monkeypatch.setenv("WECOM_ADMIN_USERS", "x")
    assert GatewayRunner._user_toolsets_override(CFG, _Src("wecom:?", None), "wecom") is None
