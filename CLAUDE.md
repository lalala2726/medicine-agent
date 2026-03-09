# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **FastAPI-based medicine AI service** that analyzes drug packaging images using Large Language Models (LLMs).
It receives base64-encoded images, processes them through Alibaba's Qwen VL models, and returns structured drug
information.

**Tech Stack:** FastAPI, LangChain, Qwen VL (via DashScope API), Pydantic

## Build, Test, and Development Commands

- This project uses conda; run commands in env `medicine-agent`.
- Prefer `conda run -n medicine-agent <command>` for reproducible execution.
- `uvicorn app.main:app --reload` runs the API locally with hot reload.
- `python -m pytest` runs the test suite (currently minimal; add tests as you implement features).
- No build step is required; this is a pure Python service.

**Required Environment Variable:**

- `DASHSCOPE_API_KEY` - Alibaba DashScope API key for Qwen models

## Architecture

The project follows a **layered architecture** with unified exception handling:

```
app/main.py              # FastAPI app bootstrap & global exception handler registration
├── app/api/main.py      # API router aggregator (prefix: /api)
│   └── app/api/routes/  # Feature-specific route handlers
├── app/core/
│   ├── llm.py           # LLM client factory (create_chat_model)
│   ├── prompts.py       # AI prompt templates (DRUG_PARSER_PROMPT)
│   ├── codes.py         # ResponseCode IntEnum
│   ├── exceptions.py    # ServiceException class
│   └── exception_handlers.py  # Global exception handlers (static methods)
└── app/schemas/
    └── response.py      # ApiResponse[T] generic model with .success() and .error() class methods
```

## Request Flow

```
Client Request
  → POST /api/image/parse/drug
  → Pydantic schema validation (ImageParseRequest)
  → create_chat_model("qwen3-vl-plus", response_format={"type": "json_object"})
  → LangChain messages (SystemMessage + HumanMessage with image parts)
  → Qwen model invocation → JSON parsing
  → ApiResponse.success(data) wrapper
  → JSONResponse to client
```

## Exception Handling Strategy

The application implements a **three-tier global exception handling** system registered in `app/main.py`:

1. **ServiceException** (`app/core/exceptions.py`) - Custom business exceptions with optional code/message/data
2. **StarletteHTTPException** - HTTP exceptions (404, 422, etc.)
3. **Exception** - Catch-all for unhandled errors

All exceptions are converted to `ApiResponse` format via `ExceptionHandlers` static methods in
`app/core/exception_handlers.py`. The `ResponseCode` IntEnum (`app/core/codes.py`) defines standard HTTP status codes
with messages.

**Raising exceptions:**

```python
raise ServiceException(message="Custom error", code=ResponseCode.BAD_REQUEST)
```

**Response pattern:**

```python
return ApiResponse.success(data)  # 200 with data
return ApiResponse.error(ResponseCode.NOT_FOUND)  # 404 with default message
```

## LLM Integration

- **Factory function:** `create_chat_model()` in `app/core/llm.py`
- **Default model:** `qwen-flash` (configurable)
- **Base URL:** Alibaba DashScope API (`https://dashscope.aliyuncs.com/compatible-mode/v1`)
- **Structured output:** Use `response_format={"type": "json_object"}` for JSON responses

The `ChatOpenAI` adapter from LangChain is used for Qwen model compatibility.

## Adding New Endpoints

1. Create route file in `app/api/routes/`
2. Define Pydantic request/response schemas
3. Use `ApiResponse[T]` as return type for consistency
4. Raise `ServiceException` for business errors
5. Include router in `app/api/main.py`

Example:

```python
from app.schemas.response import ApiResponse
from app.core.exceptions import ServiceException
from app.core.codes import ResponseCode


@router.post("/endpoint")
async def endpoint(request: RequestSchema) -> ApiResponse[ResponseSchema]:
    try:
        result = await some_service(request.data)
        return ApiResponse.success(result)
    except ValueError as e:
        raise ServiceException(message=str(e), code=ResponseCode.BAD_REQUEST)
```

## Response Schema

All API responses follow this structure via `ApiResponse[T]`:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    ...
  },
  "timestamp": 1737888000
}
```

The `.model_dump()` method automatically excludes `None` values. Use type generics for type safety:
`ApiResponse[DrugData]`.

最后！你在回答时可以直接使用中文了！如果你觉得有必要的话！
