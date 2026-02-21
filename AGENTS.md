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
- Embedding 配置：`DASHSCOPE_EMBEDDING_MODEL`（向量模型名称，必填）。
- Milvus configuration (optional): `MILVUS_URI`, `MILVUS_USER`, `MILVUS_PASSWORD`, `MILVUS_TOKEN`, `MILVUS_DB_NAME`,
  `MILVUS_TIMEOUT`.
- MongoDB configuration (optional): `MONGODB_URI` (defaults to `mongodb://localhost:27017`),
  `MONGODB_DB_NAME` (defaults to `medicine_ai_agent`), `MONGODB_TIMEOUT_MS` (defaults to `3000`),
  `MONGODB_CONVERSATIONS_COLLECTION` (defaults to `conversations`), `MONGODB_ADMIN_MESSAGES_COLLECTION`
  (defaults to `admin_messages`), `MONGODB_STARTUP_PING_ENABLED` (default false, set true to fail fast
  on startup when MongoDB is unreachable/unauthorized).
- Redis configuration (optional): `REDIS_URL`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_SSL`.
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
- Document new config values in this file when you introduce them.
