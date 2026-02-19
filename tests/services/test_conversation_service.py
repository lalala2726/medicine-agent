from bson.int64 import Int64

from app.services import conversation_service as service_module


class _DummyInsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class _DummyCursor:
    def __init__(self, documents: list[dict]):
        self._documents = documents
        self.sort_args: tuple[str, int] | None = None
        self.skip_value = 0
        self.limit_value: int | None = None

    def sort(self, field_name: str, direction: int):
        self.sort_args = (field_name, direction)
        return self

    def skip(self, value: int):
        self.skip_value = value
        return self

    def limit(self, value: int):
        self.limit_value = value
        return self

    def __iter__(self):
        items = list(self._documents)
        if self.skip_value:
            items = items[self.skip_value:]
        if self.limit_value is not None:
            items = items[:self.limit_value]
        return iter(items)


class _DummyCollection:
    def __init__(self):
        self.last_inserted: dict | None = None
        self.last_query: dict | None = None
        self.find_one_result: dict | None = None
        self.last_update_query: dict | None = None
        self.last_update_doc: dict | None = None
        self.find_result: list[dict] = []
        self.last_find_query: dict | None = None
        self.last_find_projection: dict | None = None
        self.last_count_query: dict | None = None
        self.last_cursor: _DummyCursor | None = None

    def insert_one(self, document: dict) -> _DummyInsertResult:
        self.last_inserted = document
        return _DummyInsertResult("507f1f77bcf86cd799439011")

    def find_one(self, query: dict) -> dict | None:
        self.last_query = query
        return self.find_one_result

    def update_one(self, query: dict, update_doc: dict):
        self.last_update_query = query
        self.last_update_doc = update_doc

    def count_documents(self, query: dict) -> int:
        self.last_count_query = query
        return len(self.find_result)

    def find(self, query: dict, projection: dict):
        self.last_find_query = query
        self.last_find_projection = projection
        self.last_cursor = _DummyCursor(self.find_result)
        return self.last_cursor


def test_add_admin_conversation_uses_int64_user_id(monkeypatch):
    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"conversations": collection})

    result = service_module.add_admin_conversation(
        conversation_uuid="conv-1",
        user_id=1,
    )

    assert result == "507f1f77bcf86cd799439011"
    assert collection.last_inserted is not None
    assert collection.last_inserted["conversation_type"] == "admin"
    assert collection.last_inserted["user_id"] == Int64(1)
    assert isinstance(collection.last_inserted["user_id"], Int64)


def test_get_admin_conversation_uses_int64_user_id_in_query(monkeypatch):
    collection = _DummyCollection()
    collection.find_one_result = {"uuid": "conv-1"}
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"conversations": collection})

    result = service_module.get_admin_conversation(
        conversation_uuid="conv-1",
        user_id=1,
    )

    assert result == {"uuid": "conv-1"}
    assert collection.last_query == {
        "uuid": "conv-1",
        "conversation_type": "admin",
        "user_id": Int64(1),
    }


def test_add_client_conversation_uses_client_type(monkeypatch):
    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"conversations": collection})

    service_module.add_client_conversation(
        conversation_uuid="conv-client-1",
        user_id=2,
    )

    assert collection.last_inserted is not None
    assert collection.last_inserted["conversation_type"] == "client"


def test_save_conversation_title_updates_title(monkeypatch):
    collection = _DummyCollection()
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"conversations": collection})

    service_module.save_conversation_title(
        conversation_uuid="conv-1",
        title="新标题",
    )

    assert collection.last_update_query == {"uuid": "conv-1"}
    assert collection.last_update_doc is not None
    assert collection.last_update_doc["$set"]["title"] == "新标题"
    assert "update_time" in collection.last_update_doc["$set"]


def test_list_admin_conversations_returns_uuid_and_title(monkeypatch):
    collection = _DummyCollection()
    collection.find_result = [
        {"uuid": "conv-1", "title": "会话一"},
        {"uuid": "conv-2", "title": ""},
        {"uuid": "", "title": "应忽略"},
    ]
    monkeypatch.setattr(service_module, "get_mongo_database", lambda: {"conversations": collection})

    rows, total = service_module.list_admin_conversations(
        user_id=1,
        page_num=1,
        page_size=2,
    )

    assert collection.last_count_query == {
        "conversation_type": "admin",
        "user_id": Int64(1),
    }
    assert collection.last_find_query == {
        "conversation_type": "admin",
        "user_id": Int64(1),
    }
    assert collection.last_find_projection == {
        "_id": 0,
        "uuid": 1,
        "title": 1,
    }
    assert collection.last_cursor is not None
    assert collection.last_cursor.sort_args == ("update_time", -1)
    assert collection.last_cursor.skip_value == 0
    assert collection.last_cursor.limit_value == 2
    assert total == 3
    assert rows == [
        {"conversation_uuid": "conv-1", "title": "会话一"},
        {"conversation_uuid": "conv-2", "title": "新聊天"},
    ]
