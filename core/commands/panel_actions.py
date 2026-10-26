"""面板回调动作与现有命令服务的对应关系。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelAction:
    command: str
    service: str = ""
    method: str = ""
    args: tuple = ()
    admin: bool = False


PANEL_ACTIONS = {
    "overview": PanelAction("meme menu"),
    "settings": PanelAction("meme menu settings", admin=True),
    "maintenance": PanelAction("meme menu maintenance", admin=True),
    "random": PanelAction("meme random"),
    "close": PanelAction("meme close"),
    "status": PanelAction("meme status", "command_handler", "status"),
    "on": PanelAction("meme on", "command_handler", "meme_on", admin=True),
    "off": PanelAction("meme off", "command_handler", "meme_off", admin=True),
    "auto_on": PanelAction("meme auto_on", "command_handler", "auto_on", admin=True),
    "auto_off": PanelAction("meme auto_off", "command_handler", "auto_off", admin=True),
    "natural_on": PanelAction("meme natural_analysis on", "command_handler", "toggle_natural_analysis", ("on",), True),
    "natural_off": PanelAction("meme natural_analysis off", "command_handler", "toggle_natural_analysis", ("off",), True),
    "capture": PanelAction("meme 偷", "command_handler", "capture", admin=True),
    "group": PanelAction("meme group show", "target_commands", "group_filter", ("show",), True),
    "list": PanelAction("meme list", "image_commands", "list_images", admin=True),
    "tag_stats": PanelAction("meme tag_stats", "command_handler", "tag_stats", admin=True),
    "clear_cache": PanelAction("meme clear_emotion_cache", "command_handler", "clear_emotion_cache", admin=True),
    "rebuild_index": PanelAction("meme rebuild_index", "index_commands", "rebuild_index", admin=True),
    "clean": PanelAction("meme clean", "command_handler", "clean", admin=True),
    "capacity": PanelAction("meme capacity", "command_handler", "enforce_capacity", admin=True),
}
