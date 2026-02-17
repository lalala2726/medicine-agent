import pytest

from app.core.codes import ResponseCode
from app.core.exceptions import ServiceException
from app.core.request_context import reset_current_user, set_current_user
from app.schemas.auth import AuthUser
from app.services import user_service


def test_get_current_user_from_context():
    token = set_current_user(AuthUser(id=7, username="alice"))
    try:
        user = user_service.get_current_user()
        assert user.id == 7
        assert user.username == "alice"
    finally:
        reset_current_user(token)


def test_get_current_user_id_from_context():
    token = set_current_user(AuthUser(id=9, username="bob"))
    try:
        user_id = user_service.get_current_user_id()
        assert user_id == 9
    finally:
        reset_current_user(token)


def test_get_user_info_is_backward_compatible_alias():
    token = set_current_user(AuthUser(id=12, username="carol"))
    try:
        user = user_service.get_user_info()
        assert user.id == 12
    finally:
        reset_current_user(token)


def test_get_current_user_raises_when_context_missing():
    with pytest.raises(ServiceException) as exc_info:
        user_service.get_current_user()
    assert exc_info.value.code == ResponseCode.UNAUTHORIZED.code
