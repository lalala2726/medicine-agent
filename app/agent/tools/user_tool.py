from langchain_core.tools import tool

from app.utils.http_client import HttpClient
from app.schemas.http_response import HttpResponse


@tool(description="获取当前用户信息，在用户要求获取当前用户信息的时候你需要调用本工具")
async def get_user_info() -> dict:
    """
    调用用户服务接口获取当前用户信息。
    默认会携带当前请求上下文中的 Authorization。
    """
    async with HttpClient() as client:
        response = await client.get(url="/agent/tools/current_user")
        return HttpResponse.parse_data(response)
