from bson.int64 import Int64

from app.services import conversation_service as service_module


class _DummyInsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class _DummyCollection:
    def __init__(self):
        self.last_inserted: dict | None = None
        self.last_query: dict | None = None
        self.find_one_result: dict | None = None
        self.last_update_query: dict | None = None
        self.last_update_doc: dict | None = None

    def insert_one(self, document: dict) -> _DummyInsertResult:
        self.last_inserted = document
        return _DummyInsertResult("507f1f77bcf86cd799439011")

    def find_one(self, query: dict) -> dict | None:
        self.last_query = query
        return self.find_one_result

    def update_one(self, query: dict, update_doc: dict):
        self.last_update_query = query
        self.last_update_doc = update_doc


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
