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
  `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`), `OPENAI_CHAT_MODEL` (required when chat model name is not passed explicitly),
  `OPENAI_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `OPENAI_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- Aliyun LLM provider configuration (optional): `DASHSCOPE_API_KEY` (required when provider is `aliyun`),
  `DASHSCOPE_BASE_URL` (defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`),
  `DASHSCOPE_CHAT_MODEL` (required when chat model name is not passed explicitly),
  `DASHSCOPE_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `DASHSCOPE_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- Volcengine LLM provider configuration (optional): `VOLCENGINE_LLM_API_KEY` (required when provider is `volcengine`),
  `VOLCENGINE_LLM_BASE_URL` (defaults to `https://ark.cn-beijing.volces.com/api/v3`),
  `VOLCENGINE_LLM_CHAT_MODEL` (required when chat model name is not passed explicitly),
  `VOLCENGINE_LLM_IMAGE_MODEL` (required when image model name is not passed explicitly),
  `VOLCENGINE_LLM_EMBEDDING_MODEL` (required when embedding model name is not passed explicitly).
- LLM thinking switch semantics: `think=true` enables provider-specific deep thinking payload;
  default is `false`. For `volcengine`, framework explicitly sends `thinking.type=disabled`
  when `think=false` to avoid provider default reasoning output.
- 管理助手对话接口（`POST /admin/assistant/chat`）支持请求参数 `enable_thinking`（默认 `false`），
  仅在显式传 `true` 时透传深度思考流式事件（`type=thinking`）。
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
- Rate limit configuration (optional): `RATE_LIMIT_KEY_PREFIX` (defaults to `rate_limit`),
  `RATE_LIMIT_TRUST_X_FORWARDED_FOR` (default false; when true, fallback to `X-Forwarded-For`
  only if `request.client.host` is unavailable).
- RQ configuration (optional): `RQ_QUEUE_NAME`, `RQ_DEFAULT_TIMEOUT`.
- HTTP client configuration (optional): `HTTP_BASE_URL` (defaults to `http://localhost:8080`).
- HTTP client logging (optional): `HTTP_CLIENT_LOG_ENABLED` (default false, set true to log request/response details).
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
