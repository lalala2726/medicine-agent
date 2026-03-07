# Repository Guidelines

## Project Structure & Module Organization

- `app/` is the FastAPI package. `app/main.py` defines the app and entry routes.
- `app/api/routes/` holds feature routers (ex: `app/api/routes/image.py`).
- `app/schemas/` contains Pydantic response/request models (ex: `app/schemas/response.py`).
- `app/core/` keeps shared constants and enums (ex: `app/core/codes.py`).
- `tests/` is reserved for test modules.

## Build, Test, and Development Commands

- This project uses conda; run commands in env `medicine-ai-agent`.
- Prefer `conda run -n medicine-ai-agent <command>` for reproducible execution.
- `uvicorn app.main:app --reload` runs the API locally with hot reload.
- `python -m pytest` runs the test suite (currently minimal; add tests as you implement features).
- No build step is required; this is a pure Python service.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation and type hints on public functions.
- Use `snake_case` for modules/functions, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants.
- Keep route handlers in `app/api/routes/` and shared response types in `app/schemas/`.

## Testing Guidelines

- Use `pytest`; name files `tests/test_*.py` and functions `test_*`.
- Mirror package paths for clarity (ex: `tests/api/test_image.py` for `app/api/routes/image.py`).
- Add FastAPI `TestClient` coverage for new endpoints and error paths.

## Commit & Pull Request Guidelines

- The repo has no commit history yet, so use short, imperative subjects (ex: "Add image parse stub").
- PRs should include a concise description, test command output, and any API changes (request/response examples).

## Security and Configuration

- Do not commit secrets. The app expects `DASHSCOPE_API_KEY` in the environment for model calls.
- LLM provider selection (optional): `LLM_PROVIDER` (defaults to `openai` when unset).
  Allowed values: `openai`, `aliyun`, `volcengine` (also accepts `LlmProvider.<NAME>` style strings).
- LLM config priority: function args > environment values after `python-dotenv` (`load_dotenv`) > defaults.
- OpenAI chat provider configuration (optional): `OPENAI_API_KEY` (required when provider is `openai`),
  `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`), `OPENAI_CHAT_MODEL` (required when chat model name is not
  passed explicitly),
  `OPENAI_GATEWAY_ROUTER_MODEL` (optional gateway 路由专用模型，未配置时回退 `OPENAI_CHAT_MODEL`),
  `OPENAI_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `OPENAI_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- Aliyun LLM provider configuration (optional): `DASHSCOPE_API_KEY` (required when provider is `aliyun`),
  `DASHSCOPE_BASE_URL` (defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`),
  `DASHSCOPE_CHAT_MODEL` (required when chat model name is not passed explicitly),
  `DASHSCOPE_GATEWAY_ROUTER_MODEL` (optional gateway 路由专用模型，未配置时回退 `DASHSCOPE_CHAT_MODEL`),
  `DASHSCOPE_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `DASHSCOPE_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- Volcengine LLM provider configuration (optional): `VOLCENGINE_LLM_API_KEY` (required when provider is `volcengine`),
  `VOLCENGINE_LLM_BASE_URL` (defaults to `https://ark.cn-beijing.volces.com/api/v3`),
  `VOLCENGINE_LLM_CHAT_MODEL` (required when chat model name is not passed explicitly),
  `VOLCENGINE_LLM_GATEWAY_ROUTER_MODEL` (optional gateway 路由专用模型，未配置时回退 `VOLCENGINE_LLM_CHAT_MODEL`),
  `VOLCENGINE_LLM_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `VOLCENGINE_LLM_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- LLM thinking switch semantics: `think=true` enables provider-specific deep thinking payload;
  default is `false`. For `volcengine`, framework explicitly sends `thinking.type=disabled`
  when `think=false` to avoid provider default reasoning output.
- 管理助手对话接口（`POST /admin/assistant/chat`）支持请求参数 `enable_thinking`（默认 `false`），
  仅在显式传 `true` 时透传深度思考流式事件（`type=thinking`）。
- 管理助手记忆配置（optional）：
  `ASSISTANT_MEMORY_MODE`（`window|summary`，默认 `window`），
  `ASSISTANT_MEMORY_WINDOW_LIMIT`（默认 `50`，仅 `window` 模式生效），
  `ASSISTANT_SUMMARY_TRIGGER_WINDOW`（默认 `100`，可总结消息触发阈值），
  `ASSISTANT_SUMMARY_TAIL_WINDOW`（默认 `20`，summary 模式原文尾部窗口），
  `ASSISTANT_SUMMARY_MAX_TOKENS`（默认 `2000`，summary 文本 token 硬上限），
  `ASSISTANT_SUMMARY_MODEL`（可选全局摘要模型兜底），
  `OPENAI_SUMMARY_MODEL` / `DASHSCOPE_SUMMARY_MODEL` / `VOLCENGINE_LLM_SUMMARY_MODEL`
  （可选厂商专属摘要模型，优先于 `ASSISTANT_SUMMARY_MODEL`）。
- Embedding provider configuration:
  `OPENAI_EMBEDDING_MODEL` / `DASHSCOPE_EMBEDDING_MODEL` / `VOLCENGINE_LLM_EMBEDDING_MODEL`
  （根据 provider 选择，向量模型名称必填；可由函数参数或环境变量提供）。
- Milvus configuration (optional): `MILVUS_URI`, `MILVUS_USER`, `MILVUS_PASSWORD`, `MILVUS_TOKEN`, `MILVUS_DB_NAME`,
  `MILVUS_TIMEOUT`.
- MongoDB configuration (optional): `MONGODB_URI` (defaults to `mongodb://localhost:27017`),
  `MONGODB_DB_NAME` (defaults to `medicine_ai_agent`), `MONGODB_TIMEOUT_MS` (defaults to `3000`),
  `MONGODB_CONVERSATIONS_COLLECTION` (defaults to `conversations`), `MONGODB_MESSAGES_COLLECTION`
  (defaults to `messages`), `MONGODB_MESSAGE_TRACES_COLLECTION` (defaults to `message_traces`),
  `MONGODB_MESSAGE_TTS_USAGES_COLLECTION` (defaults to `message_tts_usages`),
  `MONGODB_CONVERSATION_SUMMARIES_COLLECTION` (defaults to `conversation_summaries`),
  `MONGODB_STARTUP_PING_ENABLED` (default false, set true to fail fast
  on startup when MongoDB is unreachable/unauthorized).
- Redis configuration (optional): `REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_SSL`.
  Redis connection/config entry is `app/core/database/redis/config.py`.
- Rate limit configuration (optional): `RATE_LIMIT_KEY_PREFIX` (defaults to `rate_limit`),
  `RATE_LIMIT_TRUST_X_FORWARDED_FOR` (default false; when true, fallback to `X-Forwarded-For`
  only if `request.client.host` is unavailable).
- System auth configuration (optional):
  `SYSTEM_AUTH_ENABLED` (default true),
  `SYSTEM_AUTH_MAX_SKEW_SECONDS` (default `300`),
  `SYSTEM_AUTH_NONCE_TTL_SECONDS` (default `600`),
  `SYSTEM_AUTH_NONCE_KEY_PREFIX` (default `system_auth:nonce`),
  `SYSTEM_AUTH_CLIENTS_JSON` (JSON array, element keys: `app_id`, `secret`, `enabled`),
  `SYSTEM_AUTH_DEFAULT_SIGN_VERSION` (default `v1`),
  `X_AGENT_KEY` (AI 侧出站系统调用使用的 app_id),
  `SYSTEM_AUTH_LOCAL_SECRET` (AI 侧出站系统调用使用的本地签名密钥).
  Header contract: `X-Agent-Key`, `X-Agent-Timestamp`, `X-Agent-Nonce`,
  `X-Agent-Signature`, `X-Agent-Sign-Version`.
- RabbitMQ configuration (optional): `RABBITMQ_URL`, `RABBITMQ_EXCHANGE`,
  `RABBITMQ_COMMAND_QUEUE`, `RABBITMQ_COMMAND_ROUTING_KEY`, `RABBITMQ_RESULT_ROUTING_KEY`,
  `RABBITMQ_PREFETCH_COUNT`, `MQ_CONSUMER_ENABLED`,
  `KNOWLEDGE_LATEST_VERSION_KEY_PREFIX`, `KNOWLEDGE_VECTOR_BATCH_SIZE`.
- Chunk rebuild MQ configuration (optional):
  `RABBITMQ_CHUNK_REBUILD_EXCHANGE` (defaults to `knowledge.chunk_rebuild`),
  `RABBITMQ_CHUNK_REBUILD_COMMAND_QUEUE` (defaults to `knowledge.chunk_rebuild.command.q`),
  `RABBITMQ_CHUNK_REBUILD_COMMAND_ROUTING_KEY` (defaults to `knowledge.chunk_rebuild.command`),
  `RABBITMQ_CHUNK_REBUILD_RESULT_ROUTING_KEY` (defaults to `knowledge.chunk_rebuild.result`),
  `MQ_CHUNK_REBUILD_CONSUMER_ENABLED` (default true; controls in-process chunk rebuild consumer startup),
  `KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX` (defaults to `kb:chunk_edit:latest_version`).
- Knowledge import MQ protocol:
  business service publishes command messages (`routing_key=knowledge.import.command`);
  AI service publishes result messages (`routing_key=knowledge.import.result`).
  Result stages: `STARTED`, `PROCESSING`, `COMPLETED`, `FAILED`.
  `PROCESSING` includes `stage_detail` values: `downloading`, `parsing`,
  `chunking`, `embedding`, `inserting`.
  AI consumer enforces latest-version semantics by reading Redis key
  `kb:latest:{biz_key}` (prefix configurable) and dropping stale messages (`version < latest`).
- Knowledge chunk rebuild MQ protocol:
  business service publishes command messages (`routing_key=knowledge.chunk_rebuild.command`);
  AI service publishes result messages (`routing_key=knowledge.chunk_rebuild.result`).
  Command payload carries `task_uuid`, `knowledge_name`, `document_id`, `vector_id`, `version`,
  `content`, `embedding_model`, `created_at`.
  Result stages only include `STARTED`, `COMPLETED`, `FAILED`.
  This protocol only supports single-chunk rebuild. It uses shared Redis latest-version gating by
  `vector_id`, with key format `{KNOWLEDGE_CHUNK_EDIT_LATEST_VERSION_KEY_PREFIX}:{vector_id}`; stale
  messages are dropped with reason logging and do not update Milvus.
- Knowledge import structured logging (`app/core/mq/observability/document/import_logger.py`):
  `ImportStage` enum identifies each pipeline step. Use `import_log(stage, task_uuid, **metrics)`
  for one-line structured log output; log level is auto-selected (error / warning / info).
- HTTP client configuration (optional): `HTTP_BASE_URL` (defaults to `http://localhost:8080`).
- HTTP client logging (optional): `HTTP_CLIENT_LOG_ENABLED` (default false, set true to log request/response details).
- Download file storage configuration (required for URL import):
  `FILE_DOWNLOAD_ROOT_DIR`（固定下载根目录，必填；URL 下载文件会按 `yyyy/mm/dd` 分层落盘并长期保留，当前阶段不启用自动清理）。
- File type detection dependency (required for URL import parsing):
  `filetype`（纯 Python 依赖；用于下载后真实类型识别与解析器分发，无需系统 `libmagic`）。
- Agent tool logging (optional): `AGENT_TOOL_LOG_ENABLED` (default false, set true to log tool invocations and results).
- CORS configuration (optional): `CORS_ALLOW_ORIGINS` (comma-separated origins, takes precedence over regex when set).
- CORS configuration (optional): `CORS_ALLOW_ORIGIN_REGEX` (defaults to allowing `localhost`/`127.0.0.1` on any port).
- CORS configuration (optional): `CORS_ALLOW_METHODS` (comma-separated methods, defaults to `*`).
- CORS configuration (optional): `CORS_ALLOW_HEADERS` (comma-separated headers, defaults to `*`).
- CORS configuration (optional): `CORS_ALLOW_CREDENTIALS` (default true).
- CORS defaults are intended for local localhost debugging; tighten allowed origins/methods/headers explicitly in
  production.
- LangSmith tracing (optional): `LANGSMITH_TRACING` (or legacy `LANGCHAIN_TRACING_V2`), `LANGSMITH_API_KEY`,
  `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`.
- Volcengine shared speech auth configuration: `VOLCENGINE_APP_ID` (required),
  `VOLCENGINE_ACCESS_TOKEN` (required) for both STT and TTS.
- Volcengine bidirectional TTS configuration: `VOLCENGINE_TTS_ENDPOINT` (defaults to
  `wss://openspeech.bytedance.com/api/v3/tts/bidirection`), `VOLCENGINE_TTS_RESOURCE_ID` (optional override),
  `VOLCENGINE_TTS_VOICE_TYPE` (optional default voice), `VOLCENGINE_TTS_ENCODING` (defaults to `mp3`),
  `VOLCENGINE_TTS_SAMPLE_RATE` (defaults to `24000`), `VOLCENGINE_TTS_MAX_TEXT_CHARS` (defaults to `300`,
  max chars after sanitizer), `VOLCENGINE_TTS_STARTUP_CONNECT_ENABLED`
  (default true, run startup pre-connect check), `VOLCENGINE_TTS_STARTUP_FAIL_FAST` (default false,
  fail service startup when pre-connect check fails).
- Volcengine streaming STT configuration: `VOLCENGINE_STT_RESOURCE_ID` (required),
  `VOLCENGINE_STT_ENDPOINT` (defaults to `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async`),
  `VOLCENGINE_STT_MAX_DURATION_SECONDS` (defaults to `600`, server-side allowed max duration cap;
  business code may pass shorter `session_duration_seconds`, default session duration is 60s).
- Document new config values in this file when you introduce them.
