"""隔离按钮回调的回复凭据，不修改原消息事件或共享 SDK 客户端。"""

import copy
import hashlib
from typing import Any


class CallbackHTTP:
    def __init__(self, http: Any, event_id: str) -> None:
        self.http = http
        self.event_id = event_id
        self.sequence = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.http, name)

    async def request(self, route: Any, **kwargs) -> Any:
        # 普通 SDK 发送与 AstrBot 的 C2C 发送都在该处换成当前点击的回复凭据。
        if route.method == "POST" and route.path.endswith("/messages") and isinstance(kwargs.get("json"), dict):
            payload = dict(kwargs["json"])
            payload.pop("msg_id", None)
            payload["event_id"] = self.event_id
            if route.path.startswith(("/v2/groups/", "/v2/users/")):
                self.sequence += 1
                payload["msg_seq"] = self.sequence
            kwargs["json"] = payload
        return await self.http.request(route, **kwargs)


def callback_event(original: Any, interaction: Any, admin: bool) -> Any:
    """浅复制事件及 SDK 外壳，连接、平台与图片上传仍使用原有实现。"""
    event = copy.copy(original)
    event.bot = copy.copy(original.bot)
    event.bot.api = copy.copy(original.bot.api)
    event.bot.api._http = CallbackHTTP(original.bot.api._http, interaction.event_id)
    event._extras = dict(getattr(original, "_extras", {}))
    event._result = None
    event._force_stopped = False
    event.send_buffer = None
    event.role = "admin" if admin else "member"
    event._stealer_original_event = original
    return event


def identity_tag(value: Any) -> str:
    """记录可比对的标识摘要，日志不暴露完整账号或群标识。"""
    text = str(value or "")
    return f"{len(text)}:{hashlib.sha256(text.encode()).hexdigest()[:10]}" if text else "empty"


def source_sender(event: Any) -> tuple[str, str]:
    """QQ 按钮使用原始 OpenID，框架发送者标识仍用于 AstrBot 权限。"""
    source = event.message_obj.raw_message
    author = getattr(source, "author", None)
    for field in ("member_openid", "user_openid", "id"):
        value = getattr(author, field, None)
        if value:
            return str(value), f"author.{field}"
    return str(event.get_sender_id()), "event.sender_id"


def interaction_sender(interaction: Any) -> str:
    resolved = interaction.data.resolved
    return str(
        getattr(interaction, "group_member_openid", None)
        or getattr(interaction, "user_openid", None)
        or getattr(resolved, "user_id", None)
        or ""
    )


def context_mismatch(interaction: Any, event: Any, client: Any) -> str:
    """返回服务端拒绝的具体原因，空字符串表示身份及会话匹配。"""
    if event.bot is not client:
        return "client"
    expected, _ = source_sender(event)
    if interaction_sender(interaction) != expected:
        return "sender"
    source = event.message_obj.raw_message
    if getattr(source, "group_openid", None):
        return "" if source.group_openid == getattr(interaction, "group_openid", None) else "group"
    if getattr(getattr(source, "author", None), "user_openid", None):
        return "scene" if getattr(interaction, "group_openid", None) or getattr(interaction, "channel_id", None) else ""
    if event.is_private_chat() and getattr(source, "guild_id", None):
        return "" if source.guild_id == getattr(interaction, "guild_id", None) else "guild"
    return "" if source.channel_id == getattr(interaction, "channel_id", None) else "channel"
