"""QQ 原生功能面板的布局和发送接口。"""

import json
import secrets
from dataclasses import dataclass
from typing import Any

from .qq_panel_interactions import BUTTON_PREFIX


@dataclass(frozen=True)
class PanelButton:
    label: str
    action: str
    primary: bool = False


def panel_rows(section: str, admin: bool) -> list[list[PanelButton]]:
    """按钮使用固定动作标识，由服务端按动作权限执行。"""
    if section == "settings":
        if not admin:
            return []
        return [
            [PanelButton("开启收集", "on", True), PanelButton("关闭收集", "off")],
            [PanelButton("开启自动发送", "auto_on", True), PanelButton("关闭自动发送", "auto_off")],
            [PanelButton("智能检索", "natural_on", True), PanelButton("原文检索", "natural_off")],
            [PanelButton("强制收录", "capture", True), PanelButton("目标名单", "group")],
            [PanelButton("‹ 查看首页", "overview"), PanelButton("刷新设置", "settings"), PanelButton("维护工具 ›", "maintenance")],
        ]
    if section == "maintenance":
        if not admin:
            return []
        return [
            [PanelButton("查看收录", "list", True), PanelButton("标签统计", "tag_stats", True)],
            [PanelButton("清空情绪缓存", "clear_cache"), PanelButton("重建索引", "rebuild_index")],
            [PanelButton("清理暂存", "clean"), PanelButton("容量清理", "capacity")],
            [PanelButton("‹ 查看首页", "overview"), PanelButton("运行设置", "settings"), PanelButton("收起面板", "close")],
        ]
    rows = [
        [PanelButton("运行状态", "status", True), PanelButton("随机表情", "random", True)],
    ]
    if admin:
        rows.append([PanelButton("运行设置", "settings"), PanelButton("维护工具", "maintenance")])
    rows.append([PanelButton("刷新面板", "overview"), PanelButton("收起面板", "close")])
    return rows


def panel_content(section: str, admin: bool, config: Any) -> str:
    """正文仅展示当前状态，功能由按钮名称表达。"""
    title = {"overview": "功能面板", "settings": "运行设置", "maintenance": "维护工具"}[section]
    steal = "开" if config.steal_meme else "关"
    auto = "开" if config.auto_send_meme else "关"
    mode = "智能检索" if config.enable_natural_emotion_analysis else "原文检索"
    role = "管理员" if admin else "普通用户"
    return "\n".join([
        f"### 表情包小偷 · {title}",
        f"收集：{steal} · 自动发送：{auto} · {mode}",
        f"{role} · 仅本人 · 5 分钟有效",
    ])


def build_keyboard(rows: list[list[PanelButton]], sender: str, token: str) -> dict:
    """主功能使用蓝色按钮，设置和导航使用灰色按钮。"""
    return {"content": {"rows": [
        {"buttons": [{
            "id": BUTTON_PREFIX + button.action,
            "render_data": {"label": button.label, "visited_label": button.label, "style": int(button.primary)},
            "action": {
                "type": 1,
                "permission": {"type": 0, "specify_user_ids": [sender]},
                "data": json.dumps({"panel": token, "action": button.action}, separators=(",", ":")),
                "unsupport_tips": "请更新 QQ 或使用面板中的文字指令",
            },
        } for button in row]}
        for row in rows
    ]}}


async def send_keyboard(event: Any, content: str, keyboard: dict) -> None:
    """沿用当前消息的被动回复凭据，兼容群、单聊及频道。"""
    source = event.message_obj.raw_message
    payload = {"msg_id": source.id, "markdown": {"content": content}, "keyboard": keyboard}
    if getattr(source, "group_openid", None):
        result = await event.bot.api.post_group_message(
            group_openid=source.group_openid, msg_type=2,
            msg_seq=secrets.randbelow(10000) + 1, **payload,
        )
    elif getattr(getattr(source, "author", None), "user_openid", None):
        result = await event.post_c2c_message(
            openid=source.author.user_openid, msg_type=2,
            msg_seq=secrets.randbelow(10000) + 1, **payload,
        )
    elif event.is_private_chat() and getattr(source, "guild_id", None):
        result = await event.bot.api.post_dms(guild_id=source.guild_id, **payload)
    else:
        result = await event.bot.api.post_message(channel_id=source.channel_id, **payload)
    if result is None:
        raise RuntimeError("QQ 面板发送未返回消息")
