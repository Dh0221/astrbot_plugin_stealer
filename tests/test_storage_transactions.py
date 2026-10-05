"""File/SQLite failure injection and cancellation regressions."""
import asyncio
import errno
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from core.db.database_service import DatabaseService
from core.processing.image_processor_service import ImageProcessorService
from core.util.safe_io import commit_image_file
from astrbot_plugin_stealer.plugin_api import PluginAPI


@pytest.mark.asyncio
async def test_publication_rolls_back_and_never_overwrites(tmp_path):
    source, target = tmp_path / "raw.png", tmp_path / "library.png"
    source.write_bytes(b"original")
    async def commit(published):
        assert source.read_bytes() == published.read_bytes()
        raise RuntimeError("database failed")
    with pytest.raises(RuntimeError):
        await commit_image_file(source, target, commit)
    assert source.read_bytes() == b"original"
    assert not target.exists()
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        await commit_image_file(source, target, commit)
    assert target.read_bytes() == b"existing"
    assert source.read_bytes() == b"original"


@pytest.mark.asyncio
async def test_cross_volume_fallback_and_borrowed_source(tmp_path, monkeypatch):
    import core.util.safe_io as module
    monkeypatch.setattr(module.os, "link", Mock(side_effect=OSError(errno.EXDEV, "cross device")))
    source, target = tmp_path / "original", tmp_path / "stored"
    source.write_bytes(b"bytes")
    async def commit(published):
        assert source.exists() and published.read_bytes() == b"bytes"
    await commit_image_file(source, target, commit, consume=False)
    assert source.exists()
    second = tmp_path / "moved"
    await commit_image_file(source, second, commit)
    assert not source.exists()
    assert second.read_bytes() == b"bytes"
    assert not list(tmp_path.glob(".meme-*"))


@pytest.mark.asyncio
async def test_cancelled_publication_waits_for_filesystem_thread(tmp_path, monkeypatch):
    import core.util.safe_io as module
    source, target = tmp_path / "raw", tmp_path / "target"
    source.write_bytes(b"bytes")
    entered, release = threading.Event(), threading.Event()
    publish = module._publish_image
    def blocked(*args):
        publish(*args)
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(module, "_publish_image", blocked)
    async def commit(_target):
        pytest.fail("cancelled before metadata commit")
    task = asyncio.create_task(commit_image_file(source, target, commit))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.exists() and not target.exists()


@pytest.mark.asyncio
async def test_cancelled_database_commit_never_leaves_a_missing_asset(tmp_path, monkeypatch):
    processor, plugin = processor_fixture(tmp_path)
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    committed, release = threading.Event(), threading.Event()
    original = plugin.db_service._sync_index_sync
    def delayed(index):
        result = original(index)
        committed.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(plugin.db_service, "_sync_index_sync", delayed)
    task = asyncio.create_task(processor._store_and_index_image(str(source), True, "happy", "new", {}))
    assert await asyncio.to_thread(committed.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    path, _ = plugin.db_service.get_emoji_by_hash("new")
    assert Path(path).is_file()
    assert not source.exists()


def processor_fixture(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    categories = tmp_path / "categories"
    pending = tmp_path / "pending"
    def category_dir(category):
        directory = categories / category
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    def pending_dir():
        pending.mkdir(exist_ok=True)
        return pending
    cfg = SimpleNamespace(raw_dir=tmp_path / "raw", categories_dir=categories, pending_dir=pending,
                          categories=["happy"], content_filtration=False, vision_provider_id="",
                          ensure_category_dir=category_dir, ensure_pending_dir=pending_dir)
    plugin = SimpleNamespace(db_service=db, plugin_config=cfg)
    return ImageProcessorService(plugin), plugin


@pytest.mark.asyncio
async def test_normal_ingestion_commits_only_new_entry_and_preserves_metadata(tmp_path):
    processor, plugin = processor_fixture(tmp_path)
    await plugin.db_service.insert_batch([{"path": "old", "hash": "old", "category": "happy", "is_favorite": 1}])
    stale = {"old": {"hash": "old", "category": "happy", "is_favorite": 0}}
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    success, delta = await processor._store_and_index_image(
        str(source), True, "happy", "new", stale,
        extra_meta={"character": "fish", "scope_mode": "local", "origin_target": "group:1"},
        tags=["hi"], scenes=["hello"], desc="image",
    )
    assert success and len(delta) == 1 and "old" not in delta
    entry = plugin.db_service.get_emoji(next(iter(delta)))
    assert entry["character"] == "fish" and entry["scope_mode"] == "local"
    assert entry["tags"] == ["hi"] and entry["width"] == 32
    assert plugin.db_service.get_emoji("old")["is_favorite"] == 1
    assert not source.exists()


@pytest.mark.asyncio
async def test_ingestion_database_failure_retains_source(tmp_path, monkeypatch):
    processor, plugin = processor_fixture(tmp_path)
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    with plugin.db_service._get_connection() as conn:
        conn.execute("CREATE TRIGGER fail_emoji BEFORE INSERT ON emoji BEGIN SELECT RAISE(ABORT, 'failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        await processor._store_and_index_image(str(source), True, "happy", "new", {})
    assert source.exists()
    assert plugin.db_service.count_total() == 0
    assert not list((tmp_path / "categories" / "happy").iterdir())


@pytest.mark.asyncio
async def test_pending_insert_and_approval_failures_are_atomic(tmp_path):
    processor, plugin = processor_fixture(tmp_path)
    db = plugin.db_service
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "blue").save(source)
    with db._get_connection() as conn:
        conn.execute("CREATE TRIGGER fail_tag BEFORE INSERT ON emoji_pending_tag BEGIN SELECT RAISE(ABORT, 'failure'); END")
    assert (await processor._store_to_pending(str(source), True, "happy", "new", tags=["hi"]))[0] is False
    assert source.exists() and db.get_pending_by_hash("new") is None
    with db._get_connection() as conn:
        conn.execute("DROP TRIGGER fail_tag")
    assert (await processor._store_to_pending(
        str(source), True, "happy", "new", tags=["hi"],
        extra_meta={"character": "fish", "scope_mode": "local", "origin_target": "group:1", "retention_class": "external"},
    ))[0]
    row = db.get_pending_by_hash("new")
    await db.upsert_source({"source_id": "s", "source_type": "http_json", "name": "test"})
    await db.link_source_item({"source_id": "s", "external_id": "i", "metadata": {"pending_path": row["path"]}})
    with db._get_connection() as conn:
        conn.execute("CREATE TRIGGER fail_approval BEFORE DELETE ON emoji_pending BEGIN SELECT RAISE(ABORT, 'failure'); END")
    api = PluginAPI(plugin)
    assert (await api._approve_one(db, row["id"]))[0] is False
    assert db.get_pending(row["id"]) and db.get_emoji_by_hash("new") is None
    assert db.get_source_item("s", "i")["path"] is None
    assert not list((tmp_path / "categories" / "happy").iterdir())
    with db._get_connection() as conn:
        conn.execute("DROP TRIGGER fail_approval")
    assert (await api._approve_one(db, row["id"]))[0]
    path, entry = db.get_emoji_by_hash("new")
    assert entry["character"] == "fish" and entry["retention_class"] == "external"
    assert entry["scope_mode"] == "local" and entry["tags"] == ["hi"]
    assert db.get_source_item("s", "i")["path"] == path
    assert db.get_pending(row["id"]) is None
