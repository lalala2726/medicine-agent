import pytest
from pymilvus import DataType

from app.core.exception.exceptions import ServiceException
from app.repositories import vector_repository as repository_module


class _FakeIndexParams:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def add_index(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _FakeMilvusClient:
    def __init__(self, has_collection_result: bool) -> None:
        self._has_collection_result = has_collection_result
        self.index_params = _FakeIndexParams()
        self.create_collection_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.rows_result: list[dict] = []
        self.count_result: list[dict] = [{"count(*)": 0}]

    def has_collection(self, _name: str) -> bool:
        return self._has_collection_result

    def prepare_index_params(self) -> _FakeIndexParams:
        return self.index_params

    def create_collection(self, **kwargs) -> None:
        self.create_collection_calls.append(kwargs)

    def drop_collection(self, _name: str) -> None:
        return None

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        output_fields = kwargs.get("output_fields")
        if output_fields == ["count(*)"]:
            return self.count_result
        if output_fields == ["id"]:
            return []
        return self.rows_result

    def delete(self, **kwargs) -> None:
        self.delete_calls.append(kwargs)


def test_build_collection_schema_contains_standard_11_fields() -> None:
    """
    测试目的：验证 Milvus repository 使用标准版 11 字段 schema。
    预期结果：schema 字段顺序、字段类型、向量维度与字符串长度约束均符合约定。
    """
    schema = repository_module._build_collection_schema(
        embedding_dim=1024,
        description="demo",
    )
    field_names = [field.name for field in schema.fields]
    assert field_names == [
        "id",
        "document_id",
        "chunk_no",
        "content",
        "char_count",
        "embedding",
        "chunk_strategy",
        "chunk_size",
        "token_size",
        "source_hash",
        "created_at_ts",
    ]

    fields = {field.name: field for field in schema.fields}
    assert fields["id"].dtype == DataType.INT64
    assert fields["id"].is_primary is True
    assert fields["id"].auto_id is True
    assert fields["document_id"].dtype == DataType.INT64
    assert fields["chunk_no"].dtype == DataType.INT64
    assert fields["content"].dtype == DataType.VARCHAR
    assert (
        fields["content"].params["max_length"]
        == repository_module.DEFAULT_CONTENT_MAX_LENGTH
    )
    assert fields["char_count"].dtype == DataType.INT32
    assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
    assert fields["embedding"].params["dim"] == 1024
    assert fields["chunk_strategy"].dtype == DataType.VARCHAR
    assert (
        fields["chunk_strategy"].params["max_length"]
        == repository_module.DEFAULT_CHUNK_STRATEGY_MAX_LENGTH
    )
    assert fields["chunk_size"].dtype == DataType.INT32
    assert fields["token_size"].dtype == DataType.INT32
    assert fields["source_hash"].dtype == DataType.VARCHAR
    assert (
        fields["source_hash"].params["max_length"]
        == repository_module.DEFAULT_SOURCE_HASH_MAX_LENGTH
    )
    assert fields["created_at_ts"].dtype == DataType.INT64


def test_build_index_params_adds_document_and_embedding_indexes() -> None:
    """
    测试目的：验证索引参数组装包含 document_id 标量索引与 embedding 向量索引。
    预期结果：add_index 被调用两次且参数值符合预期。
    """
    client = _FakeMilvusClient(has_collection_result=False)

    index_params = repository_module._build_index_params(client)

    assert index_params is client.index_params
    assert client.index_params.calls == [
        {"field_name": "document_id", "index_type": "STL_SORT"},
        {
            "field_name": "embedding",
            "index_type": repository_module.DEFAULT_VECTOR_INDEX_TYPE,
            "metric_type": repository_module.DEFAULT_VECTOR_METRIC_TYPE,
        },
    ]


def test_create_collection_raises_when_collection_exists(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证集合已存在时 repository 会抛出统一业务异常。
    预期结果：抛出 ServiceException，错误文案包含 knowledge 已存在。
    """
    client = _FakeMilvusClient(has_collection_result=True)
    monkeypatch.setattr(repository_module, "get_milvus_client", lambda: client)

    with pytest.raises(ServiceException, match="knowledge 已存在"):
        repository_module.create_collection(
            knowledge_name="demo_kb",
            embedding_dim=1024,
            description="demo",
        )


def test_list_document_chunks_queries_with_expected_filter_and_fields(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证分页查询按 document_id 过滤并返回约定输出字段。
    预期结果：返回 rows 与 total 正确，且 query 参数包含正确 filter/limit/offset/output_fields。
    """
    client = _FakeMilvusClient(has_collection_result=True)
    client.count_result = [{"count(*)": 3}]
    client.rows_result = [
        {
            "id": 1,
            "document_id": 10,
            "chunk_no": 1,
            "content": "demo",
        }
    ]
    monkeypatch.setattr(repository_module, "get_milvus_client", lambda: client)

    rows, total = repository_module.list_document_chunks(
        knowledge_name="demo_kb",
        document_id=10,
        page_num=2,
        page_size=5,
    )

    assert total == 3
    assert rows == client.rows_result
    assert len(client.query_calls) == 2
    count_query = client.query_calls[0]
    rows_query = client.query_calls[1]
    assert count_query["filter"] == "document_id == 10"
    assert rows_query["filter"] == "document_id == 10"
    assert rows_query["limit"] == 5
    assert rows_query["offset"] == 5
    assert rows_query["output_fields"] == [
        "id",
        "document_id",
        "chunk_no",
        "content",
        "char_count",
        "chunk_strategy",
        "chunk_size",
        "token_size",
        "source_hash",
        "created_at_ts",
    ]


def test_delete_document_chunks_calls_milvus_delete_with_document_filter(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试目的：验证删除文档切片时会按 document_id 过滤调用 Milvus delete。
    预期结果：delete 被调用一次且过滤表达式为 document_id == 指定值。
    """
    client = _FakeMilvusClient(has_collection_result=True)
    monkeypatch.setattr(repository_module, "get_milvus_client", lambda: client)

    repository_module.delete_document_chunks(knowledge_name="demo_kb", document_id=42)

    assert len(client.delete_calls) == 1
    payload = client.delete_calls[0]
    assert payload["collection_name"] == "demo_kb"
    assert payload["filter"] == "document_id == 42"
