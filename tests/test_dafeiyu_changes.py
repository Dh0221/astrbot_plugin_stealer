"""大肥鱼分支改动：未收录原因、非表情包判定、普通图片 GIF、自动配图挑图与去重。"""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from core.events.event_handler import EventHandler
from core.processing.classification_parser import ClassificationParser
from core.processing.image_processor_service import ImageProcessorService
from core.processing.natural_emotion_analyzer import NaturalEmotionAnalyzer
from core.processing.phash_dedup_service import PHashDedupService
from core.search.meme_selector import MemeSelector
from core.search.meme_smart_select_service import MemeSmartSelectService
from core.tools.meme_search_tool import _RECENT_CAPTURE_PATTERN, MemeSearchToolWorkflow


def _analyzer(llm_text: str = ""):
    plugin = SimpleNamespace(
        plugin_config=SimpleNamespace(
            get_categories=lambda: ["happy", "troll"],
            category_info={},
            emotion_analysis_provider_id="flash",
            enable_natural_emotion_analysis=True,
        ),
        context=SimpleNamespace(
            llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text=llm_text))
        ),
    )
    return NaturalEmotionAnalyzer(plugin)


def test_not_emoji_payload_is_parsed_as_not_emoji():
    parser = ClassificationParser(None)
    parser.validate_response('{"category":"not_emoji"}')
    result = parser._parse_classification_response('{"category":"not_emoji"}', "/x.png")
    assert result[0] == parser.CATEGORY_NOT_EMOJI
    assert result[3] == parser.CATEGORY_NOT_EMOJI


@pytest.mark.asyncio
async def test_duplicate_reasons_are_distinguished(tmp_path):
    png = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "white").save(png)
    service = PHashDedupService()
    phash = await service.compute_phash(str(png))
    other = tmp_path / "b.png"
    Image.new("RGB", (32, 32), "white").save(other)
    db = SimpleNamespace(
        hash_exists=lambda h: h == "dup",
        blacklisted_hashes=lambda: {"black"},
        get_phash_map=lambda: {str(other): phash},
    )
    processor = ImageProcessorService.__new__(ImageProcessorService)
    processor.plugin, processor._phash_service = SimpleNamespace(db_service=db), service
    assert await processor._is_duplicate_or_blacklisted("dup", {}, str(png), False) == "duplicate"
    assert await processor._is_duplicate_or_blacklisted("black", {}, str(png), False) == "blacklisted"
    assert await processor._is_duplicate_or_blacklisted("new", {}, str(png), False, phash) == "similar"
    assert await processor._is_duplicate_or_blacklisted(
        "new", {}, str(png), False, phash, check_similar=False
    ) == ""


def test_plain_image_gif_is_a_candidate():
    image = SimpleNamespace(file="ABCDEF.gif", url="https://example.com/x", path="")
    assert EventHandler._looks_like_gif(image, 0, [])
    image = SimpleNamespace(file="ABCDEF.jpg", url="https://example.com/x", path="")
    segments = [{"type": "image", "data": {"file": "ABCDEF.jpg", "sub_type": 0}}]
    assert not EventHandler._looks_like_gif(image, 0, segments)


def test_send_false_abstains():
    analyzer = _analyzer()
    from core.processing.natural_emotion_analyzer import _EMOTION_ABSTAIN

    assert analyzer._parse_emotion_query('{"send":false}', fallback_query="x") is _EMOTION_ABSTAIN


@pytest.mark.asyncio
async def test_pick_candidate_parses_choice_and_zero():
    candidates = [{"path": "/a", "desc": "a"}, {"path": "/b", "desc": "b"}]
    assert await _analyzer('{"pick":2}').pick_candidate(None, "好耶", candidates) == 1
    assert await _analyzer('{"pick":0}').pick_candidate(None, "好耶", candidates) is None
    assert await _analyzer("乱说").pick_candidate(None, "好耶", candidates) == -1


@pytest.mark.asyncio
async def test_rank_excludes_recently_used(tmp_path):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.png"
        path.touch()
        paths.append(str(path))
    index = {
        paths[0]: {"category": "happy", "overlay_text": "好耶", "last_used_at": int(time.time())},
        paths[1]: {"category": "happy", "overlay_text": "好耶"},
    }
    search = SimpleNamespace(get_index=lambda: index, _bm25_dirty=False,
                             _bm25_index=SimpleNamespace(get_top_k=lambda *a, **k: []))
    scope = SimpleNamespace(_is_entry_allowed_for_event=lambda data, event: True)
    strategy = SimpleNamespace(_calculate_recent_penalty=lambda *a: 0)
    service = MemeSmartSelectService(None, search, strategy, scope)
    ranked = await service.rank_for_pick("happy", "好耶", exclude_used_within=3600)
    assert [item["path"] for item in ranked] == [paths[1]]
    assert ranked[0]["overlay_text"] == "好耶"


@pytest.mark.asyncio
async def test_picked_send_respects_model_refusal_and_session_history():
    selector = MemeSelector.__new__(MemeSelector)
    turn = SimpleNamespace(is_active_sent=lambda: False)
    analyzer = SimpleNamespace(pick_candidate=AsyncMock(return_value=None))
    selector.plugin = SimpleNamespace(
        _emoji_sender_engine=SimpleNamespace(emoji_turn_state=lambda e: turn),
        emotion_analyzer=analyzer,
    )
    selector._check_group_allowed = lambda e: True
    selector.normalize_category = lambda c: c
    selector._smart_select_service = SimpleNamespace(
        rank_for_pick=AsyncMock(return_value=[{"path": "/a.png"}])
    )
    selector.send_emoji_with_text = AsyncMock(return_value=True)
    event = SimpleNamespace(unified_msg_origin="g1")

    assert not await selector.try_send_emoji_picked(event, ["happy"], "q", reply_text="好耶")
    selector.send_emoji_with_text.assert_not_awaited()

    analyzer.pick_candidate.return_value = 0
    assert await selector.try_send_emoji_picked(event, ["happy"], "q", reply_text="好耶")
    assert "/a.png" in selector._session_recent_paths(event)
    kwargs = selector._smart_select_service.rank_for_pick.await_args.kwargs
    assert kwargs["exclude_used_within"] == MemeSelector.AUTO_RECENT_GLOBAL_SECONDS


def test_recent_capture_query_returns_newest_first():
    assert _RECENT_CAPTURE_PATTERN.search("刚偷的那张")
    assert not _RECENT_CAPTURE_PATTERN.search("开心")
    workflow = MemeSearchToolWorkflow()
    workflow.meme_selector = SimpleNamespace(is_path_allowed_for_event=lambda p, e: p != "/hidden")
    idx = {
        "/old": {"created_at": 1, "desc": "旧", "category": "happy"},
        "/new": {"created_at": 3, "desc": "新", "category": "cry", "tags": ["哭"]},
        "/hidden": {"created_at": 9, "desc": "藏", "category": "happy"},
    }
    results = workflow._recent_capture_candidates(None, idx, 5)
    assert [item[0] for item in results] == ["/new", "/old"]
    assert results[0][1] == "新"
