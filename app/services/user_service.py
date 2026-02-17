import httpx

from app.utils.http_client import HttpClient


async def get_user_info() -> httpx.Response:
    """
    获取当前用户信息

    从远程服务获取当前登录用户的详细信息

    Returns:
        httpx.Response: HTTP 响应对象，包含用户信息

    Raises:
        httpx.HTTPError: 当请求失败时抛出异常
    """
    async with HttpClient() as client:
        response = await client.get("/agent/user/info")
        print(f"[DEBUG] get_user_info response: status={response.status_code}")
        print(f"[DEBUG] get_user_info response body: {response.text}")
        return response
