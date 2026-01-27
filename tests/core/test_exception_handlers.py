import json

import anyio
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.codes import ResponseCode
from app.core.exception_handlers import ExceptionHandlers
from app.core.exceptions import ServiceException


def test_service_exception_handler_known_code():
    exc = ServiceException(message="bad", code=ResponseCode.BAD_REQUEST, data={"a": 1})

    response = anyio.run(ExceptionHandlers.service_exception_handler, None, exc)

    assert response.status_code == ResponseCode.BAD_REQUEST
    body = json.loads(response.body)
    assert body["code"] == ResponseCode.BAD_REQUEST
    assert body["message"] == "bad"
    assert body["data"] == {"a": 1}


def test_service_exception_handler_unknown_code():
    exc = ServiceException(message="custom", code=422)

    response = anyio.run(ExceptionHandlers.service_exception_handler, None, exc)

    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["code"] == 422
    assert body["message"] == "custom"


def test_http_exception_handler_known_code():
    exc = StarletteHTTPException(status_code=404, detail="not found")

    response = anyio.run(ExceptionHandlers.http_exception_handler, None, exc)

    assert response.status_code == 404
    body = json.loads(response.body)
    assert body["code"] == 404
    assert body["message"] == ResponseCode.NOT_FOUND.message


def test_http_exception_handler_unknown_code():
    exc = StarletteHTTPException(status_code=418, detail="teapot")

    response = anyio.run(ExceptionHandlers.http_exception_handler, None, exc)

    assert response.status_code == 418
    body = json.loads(response.body)
    assert body["code"] == 418
    assert body["message"] == "teapot"


def test_unhandled_exception_handler():
    response = anyio.run(ExceptionHandlers.unhandled_exception_handler, None, Exception("boom"))

    assert response.status_code == ResponseCode.INTERNAL_ERROR
    body = json.loads(response.body)
    assert body["code"] == ResponseCode.INTERNAL_ERROR
    assert body["message"] == ResponseCode.INTERNAL_ERROR.message
