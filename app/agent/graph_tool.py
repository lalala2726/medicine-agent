from __future__ import annotations

from typing import Any, LiteralString

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.database.neo4j.client import get_neo4j_client
from app.utils.list_utils import TextListUtils

# 医学图谱工具固定查询的业务数据库名称。
MEDICAL_GRAPH_DATABASE = "medicine"
# 图谱工具默认返回条数。
DEFAULT_GRAPH_QUERY_LIMIT = 10
# 图谱工具允许的最大返回条数。
MAX_GRAPH_QUERY_LIMIT = 50

# 症状候选检索的固定 Cypher 查询语句。
SEARCH_SYMPTOM_CANDIDATES_CYPHER: LiteralString = """
    MATCH (s:Symptom)
    WHERE s.name CONTAINS $keyword
    RETURN s.name AS symptom
    ORDER BY
      CASE WHEN s.name = $keyword THEN 0 ELSE 1 END ASC,
      size(s.name) ASC,
      s.name ASC
    LIMIT toInteger($limit)
"""

# 标准症状召回候选疾病的固定 Cypher 查询语句。
QUERY_DISEASE_CANDIDATES_BY_SYMPTOMS_CYPHER: LiteralString = """
    MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
    WHERE s.name IN $symptoms
    RETURN
      d.name AS disease,
      collect(DISTINCT s.name) AS matched_symptoms,
      count(DISTINCT s) AS score
    ORDER BY score DESC, disease ASC
    LIMIT toInteger($limit)
"""

# 疾病详情快照查询的固定 Cypher 查询语句。
QUERY_DISEASE_DETAIL_CYPHER: LiteralString = """
    MATCH (d:Disease {name: $disease_name})
    OPTIONAL MATCH (d)-[:has_symptom]->(s:Symptom)
    OPTIONAL MATCH (d)-[:need_check]->(c:Check)
    OPTIONAL MATCH (d)-[:common_drug]->(cd:Drug)
    OPTIONAL MATCH (d)-[:recommand_drug]->(rd:Drug)
    OPTIONAL MATCH (d)-[:do_eat]->(food_ok:Food)
    OPTIONAL MATCH (d)-[:no_eat]->(food_bad:Food)
    OPTIONAL MATCH (d)-[:recommand_eat]->(recipe:Food)
    OPTIONAL MATCH (d)-[:belongs_to]->(dep:Department)
    OPTIONAL MATCH (d)-[:acompany_with]->(comp:Disease)
    RETURN
      d.name AS disease,
      d.desc AS desc,
      d.cause AS cause,
      d.prevent AS prevent,
      d.easy_get AS easy_get,
      d.cure_way AS cure_way,
      d.cure_lasttime AS cure_lasttime,
      d.cured_prob AS cured_prob,
      d.get_prob AS get_prob,
      d.get_way AS get_way,
      d.cost_money AS cost_money,
      d.yibao_status AS yibao_status,
      d.category AS category,
      d.cure_department AS cure_department,
      collect(DISTINCT s.name) AS symptoms,
      collect(DISTINCT c.name) AS checks,
      collect(DISTINCT cd.name) AS common_drugs,
      collect(DISTINCT rd.name) AS recommended_drugs,
      collect(DISTINCT food_ok.name) AS should_eat,
      collect(DISTINCT food_bad.name) AS avoid_eat,
      collect(DISTINCT recipe.name) AS recipes,
      collect(DISTINCT dep.name) AS departments,
      collect(DISTINCT comp.name) AS complications
"""

# 候选疾病差异症状查询的固定 Cypher 查询语句。
QUERY_FOLLOWUP_SYMPTOM_CANDIDATES_CYPHER: LiteralString = """
    MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
    WHERE d.name IN $candidate_diseases
      AND NOT (s.name IN $known_symptoms)
    WITH
      s.name AS symptom,
      collect(DISTINCT d.name) AS candidate_diseases,
      count(DISTINCT d) AS disease_count,
      size($candidate_diseases) AS total_candidate_count
    WHERE disease_count < total_candidate_count
    WITH
      symptom,
      candidate_diseases,
      disease_count,
      abs((disease_count * 2) - total_candidate_count) AS balance_distance
    RETURN
      symptom,
      candidate_diseases,
      disease_count
    ORDER BY balance_distance ASC, disease_count DESC, symptom ASC
    LIMIT toInteger($limit)
"""


def _normalize_required_text(value: str, *, field_name: str) -> str:
    """规范化必填字符串字段。

    Args:
        value: 原始字符串值。
        field_name: 字段名称。

    Returns:
        str: 去除首尾空白后的非空字符串。

    Raises:
        ValueError: 归一化后为空时抛出。
    """

    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError(f"{field_name} 不能为空")
    return normalized_value


class SearchSymptomCandidatesRequest(BaseModel):
    """症状候选检索工具入参。"""

    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(
        ...,
        min_length=1,
        description="用户口语症状关键词，例如 '喉咙疼' 或 '咽痛'。",
    )
    limit: int = Field(
        default=DEFAULT_GRAPH_QUERY_LIMIT,
        ge=1,
        le=MAX_GRAPH_QUERY_LIMIT,
        description="最多返回的候选症状数量。",
    )

    @field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, value: str) -> str:
        """规范化症状关键词。

        Args:
            value: 原始关键词。

        Returns:
            str: 归一化后的关键词。
        """

        return _normalize_required_text(value, field_name="keyword")


class QueryDiseaseCandidatesBySymptomsRequest(BaseModel):
    """标准症状召回候选疾病工具入参。"""

    model_config = ConfigDict(extra="forbid")

    symptoms: list[str] = Field(
        ...,
        min_length=1,
        description="标准症状列表，例如 ['喉咙痛', '咽痛', '咽喉疼痛']。",
    )
    limit: int = Field(
        default=DEFAULT_GRAPH_QUERY_LIMIT,
        ge=1,
        le=MAX_GRAPH_QUERY_LIMIT,
        description="最多返回的候选疾病数量。",
    )

    @field_validator("symptoms")
    @classmethod
    def _validate_symptoms(cls, value: list[str]) -> list[str]:
        """规范化标准症状列表。

        Args:
            value: 原始症状列表。

        Returns:
            list[str]: 归一化后的症状列表。
        """

        return TextListUtils.normalize_required(value, field_name="symptoms")


class QueryDiseaseDetailRequest(BaseModel):
    """疾病详情查询工具入参。"""

    model_config = ConfigDict(extra="forbid")

    disease_name: str = Field(
        ...,
        min_length=1,
        description="疾病名称，例如 '上呼吸道感染'。",
    )

    @field_validator("disease_name")
    @classmethod
    def _validate_disease_name(cls, value: str) -> str:
        """规范化疾病名称。

        Args:
            value: 原始疾病名称。

        Returns:
            str: 归一化后的疾病名称。
        """

        return _normalize_required_text(value, field_name="disease_name")


class QueryFollowupSymptomCandidatesRequest(BaseModel):
    """追问症状候选查询工具入参。"""

    model_config = ConfigDict(extra="forbid")

    candidate_diseases: list[str] = Field(
        ...,
        min_length=1,
        description="当前候选疾病列表。",
    )
    known_symptoms: list[str] = Field(
        default_factory=list,
        description="当前已知症状列表，用于过滤不需要重复追问的症状。",
    )
    limit: int = Field(
        default=DEFAULT_GRAPH_QUERY_LIMIT,
        ge=1,
        le=MAX_GRAPH_QUERY_LIMIT,
        description="最多返回的追问症状数量。",
    )

    @field_validator("candidate_diseases")
    @classmethod
    def _validate_candidate_diseases(cls, value: list[str]) -> list[str]:
        """规范化候选疾病列表。

        Args:
            value: 原始候选疾病列表。

        Returns:
            list[str]: 归一化后的候选疾病列表。
        """

        return TextListUtils.normalize_required(value, field_name="candidate_diseases")

    @field_validator("known_symptoms")
    @classmethod
    def _validate_known_symptoms(cls, value: list[str]) -> list[str]:
        """规范化已知症状列表。

        Args:
            value: 原始已知症状列表。

        Returns:
            list[str]: 归一化后的已知症状列表。
        """

        return TextListUtils.normalize(value)


@tool(
    args_schema=SearchSymptomCandidatesRequest,
    description=(
            "检索医学图谱中的标准症状候选。"
            "调用时机：用户先说口语化症状时，用于把自然语言症状映射成图谱标准症状。"
    ),
)
def search_symptom_candidates(keyword: str, limit: int = DEFAULT_GRAPH_QUERY_LIMIT) -> list[dict[str, Any]]:
    """检索标准症状候选。

    Args:
        keyword: 用户口语症状关键词。
        limit: 最多返回的候选症状数量。

    Returns:
        list[dict[str, Any]]: 标准症状候选列表。
    """

    normalized_keyword = _normalize_required_text(keyword, field_name="keyword")
    return get_neo4j_client().query_all(
        SEARCH_SYMPTOM_CANDIDATES_CYPHER,
        parameters={
            "keyword": normalized_keyword,
            "limit": limit,
        },
        database=MEDICAL_GRAPH_DATABASE,
    )


@tool(
    args_schema=QueryDiseaseCandidatesBySymptomsRequest,
    description=(
            "按标准症状列表召回候选疾病。"
            "调用时机：已经把用户口语症状映射成标准症状后，用于生成候选疾病池。"
    ),
)
def query_disease_candidates_by_symptoms(
        symptoms: list[str],
        limit: int = DEFAULT_GRAPH_QUERY_LIMIT,
) -> list[dict[str, Any]]:
    """按标准症状查询候选疾病。

    Args:
        symptoms: 需要参与查询的标准症状列表。
        limit: 最多返回的候选疾病数量。

    Returns:
        list[dict[str, Any]]: 候选疾病列表，包含疾病名、命中症状和得分。
    """

    normalized_symptoms = TextListUtils.normalize_required(
        symptoms,
        field_name="symptoms",
    )
    return get_neo4j_client().query_all(
        QUERY_DISEASE_CANDIDATES_BY_SYMPTOMS_CYPHER,
        parameters={
            "symptoms": normalized_symptoms,
            "limit": limit,
        },
        database=MEDICAL_GRAPH_DATABASE,
    )


@tool(
    args_schema=QueryDiseaseDetailRequest,
    description=(
            "查询疾病详情快照。"
            "调用时机：候选疾病已经收敛或用户明确点名某个疾病时，用于一次性获取图谱详情。"
    ),
)
def query_disease_detail(disease_name: str) -> dict[str, Any] | None:
    """查询疾病详情快照。

    Args:
        disease_name: 需要查询的疾病名称。

    Returns:
        dict[str, Any] | None: 疾病详情字典；未命中时返回 `None`。
    """

    normalized_disease_name = _normalize_required_text(
        disease_name,
        field_name="disease_name",
    )
    return get_neo4j_client().query_one(
        QUERY_DISEASE_DETAIL_CYPHER,
        parameters={"disease_name": normalized_disease_name},
        database=MEDICAL_GRAPH_DATABASE,
    )


@tool(
    args_schema=QueryFollowupSymptomCandidatesRequest,
    description=(
            "基于候选疾病列表生成下一轮追问症状候选。"
            "调用时机：已有候选疾病池，且需要继续追问时，用它找最能区分候选疾病的症状。"
    ),
)
def query_followup_symptom_candidates(
        candidate_diseases: list[str],
        known_symptoms: list[str] | None = None,
        limit: int = DEFAULT_GRAPH_QUERY_LIMIT,
) -> list[dict[str, Any]]:
    """查询下一轮追问症状候选。

    Args:
        candidate_diseases: 当前候选疾病列表。
        known_symptoms: 当前已知症状列表。
        limit: 最多返回的追问症状数量。

    Returns:
        list[dict[str, Any]]: 追问症状候选列表，包含症状名、关联候选疾病和疾病覆盖数。
    """

    normalized_candidate_diseases = TextListUtils.normalize_required(
        candidate_diseases,
        field_name="candidate_diseases",
    )
    normalized_known_symptoms = TextListUtils.normalize(known_symptoms)
    return get_neo4j_client().query_all(
        QUERY_FOLLOWUP_SYMPTOM_CANDIDATES_CYPHER,
        parameters={
            "candidate_diseases": normalized_candidate_diseases,
            "known_symptoms": normalized_known_symptoms,
            "limit": limit,
        },
        database=MEDICAL_GRAPH_DATABASE,
    )


__all__ = [
    "DEFAULT_GRAPH_QUERY_LIMIT",
    "MAX_GRAPH_QUERY_LIMIT",
    "MEDICAL_GRAPH_DATABASE",
    "QUERY_DISEASE_CANDIDATES_BY_SYMPTOMS_CYPHER",
    "QUERY_DISEASE_DETAIL_CYPHER",
    "QUERY_FOLLOWUP_SYMPTOM_CANDIDATES_CYPHER",
    "QueryDiseaseCandidatesBySymptomsRequest",
    "QueryDiseaseDetailRequest",
    "QueryFollowupSymptomCandidatesRequest",
    "SEARCH_SYMPTOM_CANDIDATES_CYPHER",
    "SearchSymptomCandidatesRequest",
    "search_symptom_candidates",
    "query_disease_candidates_by_symptoms",
    "query_disease_detail",
    "query_followup_symptom_candidates",
]
