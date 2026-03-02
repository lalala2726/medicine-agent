import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from app.services.admin_assistant_service import generate_title

def test_generate_title_with_valid_input(monkeypatch):
    """测试正常生成标题的情况"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="感冒灵销量查询")
    
    monkeypatch.setattr("app.services.admin_assistant_service.create_chat_model", lambda **kwargs: mock_llm)
    monkeypatch.setattr("app.services.admin_assistant_service.load_prompt", lambda path: "# 模拟提示词")
    
    title = generate_title("帮我查查感冒灵卖了多少")
    
    assert title == "感冒灵销量查询"
    # 验证模型参数
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "帮我查查感冒灵卖了多少"

def test_generate_title_empty_input():
    """测试空输入返回兜底标题"""
    assert generate_title("") == "未知标题"
    assert generate_title(None) == "未知标题"

def test_generate_title_llm_returns_empty(monkeypatch):
    """测试模型返回为空的情况"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="")
    
    monkeypatch.setattr("app.services.admin_assistant_service.create_chat_model", lambda **kwargs: mock_llm)
    monkeypatch.setattr("app.services.admin_assistant_service.load_prompt", lambda path: "# 模拟提示词")
    
    title = generate_title("测试输入")
    assert title == "未知标题"

def test_generate_title_strips_whitespace(monkeypatch):
    """测试自动去除首尾空格"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="  有空格的标题  ")
    
    monkeypatch.setattr("app.services.admin_assistant_service.create_chat_model", lambda **kwargs: mock_llm)
    monkeypatch.setattr("app.services.admin_assistant_service.load_prompt", lambda path: "# 模拟提示词")
    
    title = generate_title("测试输入")
    assert title == "有空格的标题"
