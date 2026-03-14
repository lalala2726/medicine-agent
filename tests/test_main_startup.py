from __future__ import annotations

import asyncio

import app.main as main_module


def test_prepare_runtime_before_serving_initializes_config_before_speech_probes(
        monkeypatch,
) -> None:
    """测试目的：启动阶段必须先加载 Redis 快照，再执行语音探活；预期结果：调用顺序固定。"""

    call_order: list[str] = []

    def _fake_initialize_snapshot() -> None:
        call_order.append("snapshot")

    async def _fake_verify_stt() -> None:
        call_order.append("stt")

    async def _fake_verify_tts() -> None:
        call_order.append("tts")

    monkeypatch.setattr(main_module, "initialize_agent_config_snapshot", _fake_initialize_snapshot)
    monkeypatch.setattr(main_module, "verify_volcengine_stt_connection_on_startup", _fake_verify_stt)
    monkeypatch.setattr(main_module, "verify_volcengine_tts_connection_on_startup", _fake_verify_tts)

    asyncio.run(main_module._prepare_runtime_before_serving())

    assert call_order[0] == "snapshot"
    assert set(call_order[1:]) == {"stt", "tts"}
