import json

from fastapi.responses import StreamingResponse

from app.services import admin_assisant_service as service_module
from app.services.assistant_stream_service import AssistantStreamConfig


class _DummyGraph:
    def __init__(self, final_state: dict | None = None):
        self.final_state = final_state or {}
        self.captured_config = None

    def invoke(self, _state: dict, config: dict | None = None) -> dict:
        self.captured_config = config
        return self.final_state


def test_invoke_admin_workflow_passes_langsmith_config(monkeypatch):
    graph = _DummyGraph(final_state={"results": {"chat": {"content": "ok"}}})
    monkeypatch.setattr(service_module, "ADMIN_WORKFLOW", graph)
    monkeypatch.setattr(
        service_module,
        "build_langsmith_runnable_config",
        lambda **kwargs: {
            "run_name": "admin_assistant_graph",
            "tags": ["admin-assistant", "langgraph"],
            "metadata": {"entrypoint": "api.admin_assistant.chat"},
        },
    )

    state = {"user_input": "hello"}
    result = service_module._invoke_admin_workflow(state)

    assert result["results"]["chat"]["content"] == "ok"
    assert graph.captured_config is not None
    assert graph.captured_config["run_name"] == "admin_assistant_graph"


def test_assistant_chat_delegates_to_stream_service(monkeypatch):
    captured: dict = {}

    def _fake_create_streaming_response(question: str, config: AssistantStreamConfig):
        captured["question"] = question
        captured["config"] = config

        async def _stream():
            yield (
                "data: "
                + json.dumps(
                    {
                        "content": {"text": "delegated"},
                        "type": "answer",
                        "is_end": False,
                        "timestamp": 1,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "content": {"text": ""},
                        "type": "answer",
                        "is_end": True,
                        "timestamp": 2,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        return StreamingResponse(_stream(), media_type="text/event-stream")

    monkeypatch.setattr(
        service_module,
        "create_streaming_response",
        _fake_create_streaming_response,
    )

    response = service_module.assistant_chat(
        question="代理测试",
        conversation_uuid="conv-1",
    )

    assert isinstance(response, StreamingResponse)
    stream_config = captured["config"]
    assert captured["question"] == "代理测试"
    assert isinstance(stream_config, AssistantStreamConfig)
    assert stream_config.workflow is service_module.ADMIN_WORKFLOW
    assert stream_config.build_initial_state("x")["user_input"] == "x"
    assert stream_config.build_initial_state("x")["history_messages"] == []
    assert stream_config.extract_final_content({"results": {"chat": {"content": "ok"}}}) == ""
    assert stream_config.should_stream_token("chat_agent", {"routing": {}, "plan": []}) is True
    assert stream_config.should_stream_token("router", {"routing": {}, "plan": []}) is False
    assert stream_config.map_exception(RuntimeError("boom")) == "服务暂时不可用，请稍后重试。"
