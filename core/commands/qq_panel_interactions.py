"""QQ 面板回调的注册、鉴权、应答和任务回收。"""

import asyncio
import json
import weakref
from typing import Any

from astrbot.api import logger

from .panel_actions import PANEL_ACTIONS
from .qq_callback_event import callback_event, context_mismatch, identity_tag, interaction_sender, source_sender


BUTTON_PREFIX = "stealer_panel:"


class QQPanelInteractions:
    def __init__(self, menu: Any) -> None:
        self.menu = menu
        self.bindings: dict[int, tuple] = {}
        self.tasks: set[asyncio.Task] = set()
        self.active = True

    def register(self) -> None:
        for platform in self.menu.plugin.context.platform_manager.get_insts():
            if platform.meta().name in {"qq_official", "qq_official_webhook"}:
                self.bind(platform.get_client())
                intents = getattr(platform, "intents", None)
                if intents is not None:
                    intents.interaction = True

    def bind(self, client: Any) -> None:
        if not self.active or id(client) in self.bindings:
            return
        if isinstance(getattr(client, "intents", None), int):
            client.intents |= 1 << 26
        elif getattr(client, "intents", None) is not None:
            client.intents.interaction = True
        previous = getattr(client, "on_interaction_create", None)
        reference = weakref.ref(self)

        async def handler(interaction):
            router = reference()
            if router is not None and router.active and await router.handle(interaction, client):
                return
            if previous is not None:
                await previous(interaction)

        client.on_interaction_create = handler
        self.bindings[id(client)] = (client, handler, previous)

    def close(self) -> None:
        self.active = False
        for client, handler, previous in self.bindings.values():
            if getattr(client, "on_interaction_create", None) is handler:
                if previous is None:
                    delattr(client, "on_interaction_create")
                else:
                    client.on_interaction_create = previous
        self.bindings.clear()
        for task in list(self.tasks):
            task.cancel()

    async def handle(self, interaction: Any, client: Any) -> bool:
        resolved = getattr(getattr(interaction, "data", None), "resolved", None)
        button_id = str(getattr(resolved, "button_id", "") or "")
        if not button_id.startswith(BUTTON_PREFIX):
            return False

        async def respond(code):
            if code:
                logger.warning(f"[MemeMenu] 回调拒绝 code={code}")
            await interaction._api.on_interaction_result(interaction.id, code)

        try:
            payload = json.loads(resolved.button_data)
            token, action = payload["panel"], payload["action"]
            if not isinstance(token, str) or not isinstance(action, str):
                raise ValueError("invalid panel payload")
            if button_id != BUTTON_PREFIX + action or action not in PANEL_ACTIONS:
                raise ValueError("unknown panel action")
        except (ValueError, TypeError, KeyError):
            logger.warning("[MemeMenu] 回调数据无效 reason=payload")
            await respond(1)
            return True
        logger.info(
            f"[MemeMenu] 收到按钮回调 panel={identity_tag(token)} action={action} "
            f"qq_user={identity_tag(interaction_sender(interaction))} "
            f"group={identity_tag(getattr(interaction, 'group_openid', None))} "
            f"scene={getattr(interaction, 'scene', None)}"
        )
        session = self.menu._sessions.get(token)
        if session is None or session.expires_at <= asyncio.get_running_loop().time():
            await respond(3)
            return True
        mismatch = context_mismatch(interaction, session.event, client)
        if mismatch:
            expected, sender_field = source_sender(session.event)
            logger.warning(
                f"[MemeMenu] 回调身份不匹配 reason={mismatch} panel={identity_tag(token)} "
                f"expected_user={identity_tag(expected)} user_field={sender_field} "
                f"actual_user={identity_tag(interaction_sender(interaction))} "
                f"expected_group={identity_tag(getattr(session.event.message_obj.raw_message, 'group_openid', None))} "
                f"actual_group={identity_tag(getattr(interaction, 'group_openid', None))}"
            )
            await respond(4)
            return True
        if interaction.id in session.handled:
            await respond(3)
            return True
        admin = self.menu.current_admin(session.event)
        if PANEL_ACTIONS[action].admin and not admin:
            await respond(5)
            return True
        if session.busy:
            await respond(2)
            return True
        if not getattr(interaction, "event_id", None):
            await respond(1)
            return True
        session.busy = True
        try:
            event = callback_event(session.event, interaction, admin)
            await respond(0)
        except (Exception, asyncio.CancelledError):
            session.busy = False
            raise
        # 应答期间可能发生卸载、收起或到期，不能再启动新的执行任务。
        if (
            not self.active
            or self.menu._sessions.get(token) is not session
            or session.expires_at <= asyncio.get_running_loop().time()
        ):
            session.busy = False
            return True
        session.handled.add(interaction.id)
        task = asyncio.create_task(self._execute(event, token, action, session))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return True

    async def _execute(self, event, token, action, session) -> None:
        try:
            await self.menu.execute_action(event, token, action, session)
        except Exception as exc:
            logger.error(f"[MemeMenu] 面板回调执行失败: {exc}", exc_info=True)
            try:
                await event.send(event.plain_result("操作失败，请重试或查看插件日志。"))
            except Exception as reply_error:
                logger.error(f"[MemeMenu] 回调失败提示发送失败: {reply_error}")
        finally:
            session.busy = False
