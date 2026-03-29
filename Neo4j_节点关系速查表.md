# Neo4j 节点关系速查表

## 1. 使用说明

本表用于快速查看 `medicine` 数据库中的：

- 节点标签
- 节点主要属性
- 关系类型
- 关系方向
- 常见查询入口

适合后续写 Cypher、做 AI 问答接入、排查图谱结构时直接查阅。

当前数据库：`medicine`

## 2. 节点表

| 节点标签         |    数量 | 主键属性   | 主要属性                                                                                                                                                      | 说明          | 常见查询入口                    |
|--------------|------:|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------------------|
| `Disease`    |  8807 | `name` | `desc` `prevent` `cause` `easy_get` `cure_department` `cure_way` `cure_lasttime` `cured_prob` `get_prob` `get_way` `cost_money` `yibao_status` `category` | 疾病核心节点      | 按疾病名查简介、病因、症状、检查、药物、饮食、科室 |
| `Symptom`    |  5998 | `name` | `name`                                                                                                                                                    | 症状节点        | 按症状反查疾病                   |
| `Drug`       |  3832 | `name` | `name`                                                                                                                                                    | 药物通用名       | 按药物反查疾病                   |
| `Producer`   | 17234 | `name` | `name`                                                                                                                                                    | 在售药品名或生产方名称 | 查某个商品药对应什么药物              |
| `Food`       |  4870 | `name` | `name`                                                                                                                                                    | 食物或食谱       | 查疾病宜吃、忌吃、推荐食谱             |
| `Check`      |  3353 | `name` | `name`                                                                                                                                                    | 检查项目        | 按检查项目反查疾病                 |
| `Department` |    54 | `name` | `name`                                                                                                                                                    | 科室节点        | 查疾病挂什么科、科室层级              |

## 3. 疾病节点属性表

`Disease` 是最核心的节点，后续大部分查询都围绕它展开。

| 属性名               | 类型             | 含义       | 常见用途         |
|-------------------|----------------|----------|--------------|
| `name`            | `string`       | 疾病名称     | 精确查询主键       |
| `desc`            | `string`       | 疾病简介     | 用户问“这是什么病”   |
| `prevent`         | `string`       | 预防措施     | 用户问“怎么预防”    |
| `cause`           | `string`       | 病因       | 用户问“为什么会得”   |
| `easy_get`        | `string`       | 易感人群     | 用户问“哪些人容易得”  |
| `cure_department` | `list[string]` | 原始治疗科室列表 | 补充科室信息       |
| `cure_way`        | `list[string]` | 治疗方式     | 用户问“怎么治疗”    |
| `cure_lasttime`   | `string`       | 治疗周期     | 用户问“多久能好”    |
| `cured_prob`      | `string`       | 治愈概率     | 用户问“治愈率多少”   |
| `get_prob`        | `string`       | 发病概率     | 用户问“发病率多少”   |
| `get_way`         | `string`       | 传播/传染方式  | 用户问“传不传染”    |
| `cost_money`      | `string`       | 费用说明     | 用户问“治疗大概多少钱” |
| `yibao_status`    | `string`       | 医保状态     | 用户问“是否医保”    |
| `category`        | `list[string]` | 疾病分类     | 用于分类展示       |

## 4. 关系表

| 起点节点         | 关系类型             | 终点节点         |      数量 | 中文含义        | 典型问题           |
|--------------|------------------|--------------|--------:|-------------|----------------|
| `Disease`    | `has_symptom`    | `Symptom`    |   54710 | 疾病有什么症状     | 上呼吸道感染有哪些症状    |
| `Disease`    | `acompany_with`  | `Disease`    |   12024 | 疾病并发症       | 上呼吸道感染会引起什么并发症 |
| `Disease`    | `common_drug`    | `Drug`       |   14647 | 疾病常用药       | 上呼吸道感染常用什么药    |
| `Disease`    | `recommand_drug` | `Drug`       |   59465 | 疾病推荐药       | 上呼吸道感染推荐什么药    |
| `Disease`    | `need_check`     | `Check`      |   39418 | 疾病需要做什么检查   | 上呼吸道感染做什么检查    |
| `Disease`    | `do_eat`         | `Food`       |   22230 | 疾病宜吃什么      | 上呼吸道感染适合吃什么    |
| `Disease`    | `no_eat`         | `Food`       |   22239 | 疾病忌吃什么      | 上呼吸道感染不能吃什么    |
| `Disease`    | `recommand_eat`  | `Food`       |   40221 | 疾病推荐食谱      | 上呼吸道感染推荐食谱     |
| `Disease`    | `belongs_to`     | `Department` | 8807 左右 | 疾病所属科室      | 上呼吸道感染挂什么科     |
| `Department` | `belongs_to`     | `Department` |   36 左右 | 科室上下级关系     | 内科下面有哪些子科室     |
| `Producer`   | `drugs_of`       | `Drug`       |   17318 | 商品药/生产方对应药物 | 某商品药对应什么通用药    |

说明：

- `belongs_to` 总数是 `8843`，其中包含 `Disease -> Department` 和 `Department -> Department` 两种关系。
- `Disease -> Department` 约 `8807` 条。
- `Department -> Department` 约 `36` 条。

## 5. 关系方向图

```text
Disease --has_symptom--> Symptom
Disease --acompany_with--> Disease
Disease --common_drug--> Drug
Disease --recommand_drug--> Drug
Disease --need_check--> Check
Disease --do_eat--> Food
Disease --no_eat--> Food
Disease --recommand_eat--> Food
Disease --belongs_to--> Department
Department --belongs_to--> Department
Producer --drugs_of--> Drug
```

## 6. 按查询目的查什么节点和关系

| 查询目的     | 起始节点         | 关系/属性                              | 返回内容     |
|----------|--------------|------------------------------------|----------|
| 查疾病简介    | `Disease`    | `desc`                             | 疾病介绍     |
| 查疾病病因    | `Disease`    | `cause`                            | 病因说明     |
| 查疾病预防    | `Disease`    | `prevent`                          | 预防建议     |
| 查疾病症状    | `Disease`    | `has_symptom`                      | 症状列表     |
| 由症状反查疾病  | `Symptom`    | `has_symptom` 反向查                  | 疾病候选     |
| 查疾病检查项目  | `Disease`    | `need_check`                       | 检查项目     |
| 由检查反查疾病  | `Check`      | `need_check` 反向查                   | 疾病候选     |
| 查疾病药物    | `Disease`    | `common_drug` `recommand_drug`     | 药物列表     |
| 由药物反查疾病  | `Drug`       | `common_drug` `recommand_drug` 反向查 | 疾病候选     |
| 查疾病饮食建议  | `Disease`    | `do_eat` `no_eat` `recommand_eat`  | 宜吃、忌吃、食谱 |
| 查疾病科室    | `Disease`    | `belongs_to`                       | 科室       |
| 查科室层级    | `Department` | `belongs_to`                       | 上级/下级科室  |
| 查疾病并发症   | `Disease`    | `acompany_with`                    | 并发症列表    |
| 查商品药对应药物 | `Producer`   | `drugs_of`                         | 通用药名称    |

## 7. 常用查询模板

### 7.1 按疾病名查基础信息

```cypher
MATCH (d:Disease {name: $disease_name})
RETURN d
```

### 7.2 查疾病症状

```cypher
MATCH (d:Disease {name: $disease_name})-[:has_symptom]->(s:Symptom)
RETURN d.name AS disease, collect(DISTINCT s.name) AS symptoms
```

### 7.3 由症状反查疾病

```cypher
MATCH (d:Disease)-[:has_symptom]->(s:Symptom {name: $symptom_name})
RETURN s.name AS symptom, collect(DISTINCT d.name)[0..20] AS diseases
```

### 7.4 查疾病检查项目

```cypher
MATCH (d:Disease {name: $disease_name})-[:need_check]->(c:Check)
RETURN d.name AS disease, collect(DISTINCT c.name) AS checks
```

### 7.5 查疾病药物

```cypher
MATCH (d:Disease {name: $disease_name})
OPTIONAL MATCH (d)-[:common_drug]->(cd:Drug)
OPTIONAL MATCH (d)-[:recommand_drug]->(rd:Drug)
RETURN
  d.name AS disease,
  collect(DISTINCT cd.name) AS common_drugs,
  collect(DISTINCT rd.name) AS recommended_drugs
```

### 7.6 查疾病饮食建议

```cypher
MATCH (d:Disease {name: $disease_name})
OPTIONAL MATCH (d)-[:do_eat]->(food_ok:Food)
OPTIONAL MATCH (d)-[:no_eat]->(food_bad:Food)
OPTIONAL MATCH (d)-[:recommand_eat]->(recipe:Food)
RETURN
  d.name AS disease,
  collect(DISTINCT food_ok.name) AS should_eat,
  collect(DISTINCT food_bad.name) AS avoid_eat,
  collect(DISTINCT recipe.name) AS recipes
```

### 7.7 查疾病所属科室

```cypher
MATCH (d:Disease {name: $disease_name})-[:belongs_to]->(dep:Department)
RETURN d.name AS disease, collect(DISTINCT dep.name) AS departments
```

### 7.8 批量查多个候选疾病详情

```cypher
UNWIND range(0, size($disease_names) - 1) AS idx
WITH idx, $disease_names[idx] AS disease_name
MATCH (d:Disease {name: disease_name})
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
  idx AS order_index,
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
ORDER BY order_index ASC
```

适用场景：

- 已经把候选疾病收敛到 2 到 5 个。
- 需要一次性比较多个候选疾病的症状、检查、药物、饮食、科室、并发症。
- 不希望按疾病一个一个重复查询详情。

### 7.9 查商品药对应通用药

```cypher
MATCH (p:Producer {name: $producer_name})-[:drugs_of]->(d:Drug)
RETURN p.name AS producer_name, collect(DISTINCT d.name) AS drugs
```

## 8. 后续查询建议

如果用户信息不完整，推荐按下面顺序查：

1. 先把用户口语映射到标准 `Symptom`
2. 用 `Symptom -> Disease` 先召回前 5 个候选疾病
3. 基于这 5 个候选疾病的差异症状继续追问用户
4. 根据新增症状反复收敛候选疾病，直到缩小到前 2 到 3 个
5. 最后一次性批量查询这 2 到 3 个疾病的详情，不要逐个重复查询

比如用户只说 `喉咙疼`：

- 先映射症状：`喉咙痛`、`咽痛`、`咽喉疼痛`
- 再反查前 5 个候选疾病
- 然后追问：`是否鼻塞、打喷嚏、低热、咳嗽、声音嘶哑`
- 收敛到前 2 到 3 个后，再批量查询这些疾病详情并给出可能病情说明

## 9. 文档定位

如果你要看完整说明、AI 提示词模板、详细 Cypher 方案，可以再看：

- [Neo4j_AI_医疗图谱查询说明.md](document/Neo4j_AI_医疗图谱查询说明.md)
