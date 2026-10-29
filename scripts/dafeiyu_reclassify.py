"""用当前识图提示词把整个图库重新识别一遍（大肥鱼分支运维脚本）。

两步走，先看后写：
  preview：逐张调用插件的 /analyze 接口（只预览，不写库），把新旧结果写进
           <out>/reclassify.jsonl，并生成带缩略图的对照页 <out>/reclassify.html。
  apply：  读取 preview 的结果，调用 /images/update 把新的分类、描述、图上文字、
           标签和适合回应的句子写回。--skip 列出不想改的编号（对照页里的 #编号）。
           被判定“不是表情包”的图不会自动处理，只在对照页里列出来。

插件接口走 AstrBot 面板，需要面板登录后的 token：
  python dafeiyu_reclassify.py preview --db .../cache/emoji.db --base http://127.0.0.1:6185/api/plug/astrbot_plugin_stealer --token XXX --out /tmp/reclass
  python dafeiyu_reclassify.py apply   --base ... --token XXX --out /tmp/reclass [--skip 3,17] [--keep-moves angry>troll,happy>troll]
--keep-moves 列出的分类变动不改分类（保留原分类和情绪），描述等照样更新。
--keep-nos 同上，但按对照页 #编号指定，例如 --keep-nos 31,75。
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path


def _post(base: str, token: str, route: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/{route.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # AstrBot 面板可能把插件返回包在 {"status","data"} 里
    if isinstance(body, dict) and "success" not in body and isinstance(body.get("data"), dict):
        body = body["data"]
    return body


def _load_library(db_path: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT path, hash, category, desc, overlay_text, emotions_json FROM emoji ORDER BY created_at"
    )]
    tags: dict[str, list[str]] = {}
    for r in conn.execute("SELECT path, tag FROM emoji_tag"):
        tags.setdefault(r["path"], []).append(r["tag"])
    scenes: dict[str, list[str]] = {}
    for r in conn.execute("SELECT path, scene FROM emoji_scene"):
        scenes.setdefault(r["path"], []).append(r["scene"])
    conn.close()
    for row in rows:
        try:
            row["emotions"] = json.loads(row.pop("emotions_json") or "[]")
        except ValueError:
            row["emotions"] = []
        row["tags"] = tags.get(row["path"], [])
        row["scenes"] = scenes.get(row["path"], [])
    return rows


def _thumbnail(path: str) -> str:
    try:
        from PIL import Image
    except ImportError:
        return ""
    try:
        with Image.open(path) as im:
            im.seek(0)
            frame = im.convert("RGB")
            frame.thumbnail((140, 140))
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


NON_CATEGORY_RESULTS = {"非表情包", "过滤不通过"}


def _category_changed(old: dict, new: dict) -> bool:
    return old.get("category") != new.get("category")


def _applicable(record: dict) -> bool:
    new = record.get("new")
    return bool(new) and new.get("category") not in NON_CATEGORY_RESULTS


def preview(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "reclassify.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["hash"])
    library = _load_library(args.db)
    with jsonl.open("a", encoding="utf-8") as fh:
        for number, row in enumerate(library, start=1):
            if row["hash"] in done:
                continue
            try:
                result = _post(args.base, args.token, "analyze", {"hash": row["hash"]})
            except Exception as e:
                result = {"success": False, "error": str(e)}
            record = {"no": number, "hash": row["hash"], "path": row["path"], "old": row}
            if result.get("success"):
                record["new"] = {
                    "category": result.get("category", ""),
                    "emotions": result.get("emotions") or [],
                    "desc": result.get("description", ""),
                    "overlay_text": result.get("overlay_text", ""),
                    "tags": result.get("tags") or [],
                    "scenes": result.get("scenes") or [],
                }
            else:
                record["error"] = result.get("error", "unknown")
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{number}/{len(library)}] {row['category']} -> "
                  f"{record.get('new', {}).get('category', record.get('error'))}", flush=True)
            time.sleep(args.interval)
    _write_html(out)


def _write_html(out: Path) -> None:
    records = [json.loads(line) for line in (out / "reclassify.jsonl").read_text(encoding="utf-8").splitlines()]
    records.sort(key=lambda r: r["no"])
    usable = [r for r in records if _applicable(r)]
    changed = [r for r in usable if _category_changed(r["old"], r["new"])]
    same = [r for r in usable if not _category_changed(r["old"], r["new"])]
    not_emoji = [r for r in records if r.get("new") and not _applicable(r)]
    failed = [r for r in records if "new" not in r]
    moves: dict[str, int] = {}
    for r in changed:
        key = f"{r['old']['category']} → {r['new']['category']}"
        moves[key] = moves.get(key, 0) + 1

    def cell(d: dict) -> str:
        return (
            f"<b>{html.escape(d.get('category', ''))}</b> {html.escape(','.join(d.get('emotions') or []))}<br>"
            f"图上文字：{html.escape(d.get('overlay_text') or '无')}<br>"
            f"描述：{html.escape(d.get('desc') or '')}<br>"
            f"适合回应：{html.escape(' / '.join(d.get('scenes') or []))}"
        )

    def table(items: list[dict]) -> str:
        rows = []
        for r in items:
            thumb = _thumbnail(r["path"])
            img = f'<img src="{thumb}">' if thumb else ""
            new = cell(r["new"]) if r.get("new") else html.escape(r.get("error", ""))
            rows.append(f"<tr><td>#{r['no']}</td><td>{img}</td><td>{cell(r['old'])}</td><td>{new}</td></tr>")
        return ("<table><tr><th>编号</th><th>图</th><th>现在</th><th>重新识别</th></tr>"
                + "".join(rows) + "</table>")
    summary = "".join(f"<li>{html.escape(k)}：{v} 张</li>" for k, v in sorted(moves.items(), key=lambda x: -x[1]))
    page = f"""<!doctype html><meta charset="utf-8"><title>表情包重新识别对照</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse}}td{{border:1px solid #ccc;padding:6px;vertical-align:top;max-width:360px}}img{{max-width:140px}}</style>
<h1>表情包重新识别对照</h1>
<p>共 {len(records)} 张：分类改变 {len(changed)} 张，分类不变 {len(same)} 张（描述和适合回应的句子会更新），
判定不是表情包 {len(not_emoji)} 张（不自动处理），识别失败 {len(failed)} 张（保持原样）。
左边是现在库里的，右边是新提示词的结果。不想改的记下 # 编号。</p>
<h2>分类变动统计</h2><ul>{summary}</ul>
<h2>分类改变（{len(changed)}）</h2>{table(changed)}
<h2>判定不是表情包（{len(not_emoji)}）</h2>{table(not_emoji)}
<h2>识别失败（{len(failed)}）</h2>{table(failed)}
<h2>分类不变，只更新描述等（{len(same)}）</h2>{table(same)}"""
    (out / "reclassify.html").write_text(page, encoding="utf-8")
    print(f"对照页: {out / 'reclassify.html'}，分类改变 {len(changed)}，不是表情包 {len(not_emoji)}，失败 {len(failed)}")


def apply(args) -> None:
    out = Path(args.out)
    skip = {int(x) for x in args.skip.split(",") if x.strip()} if args.skip else set()
    keep_moves = {tuple(m.split(">", 1)) for m in args.keep_moves.split(",") if ">" in m}
    keep_nos = {int(x) for x in args.keep_nos.split(",") if x.strip()}
    kept = 0
    records = [json.loads(line) for line in (out / "reclassify.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = fail = 0
    for r in sorted(records, key=lambda r: r["no"]):
        if not _applicable(r) or r["no"] in skip:
            continue
        new = r["new"]
        category, emotions = new["category"], new["emotions"]
        if (r["old"]["category"], category) in keep_moves or r["no"] in keep_nos:
            category, emotions = r["old"]["category"], r["old"]["emotions"]
            kept += 1
        payload = {
            "hash": r["hash"],
            "category": category,
            "emotions": emotions,
            "desc": new["desc"],
            "overlay_text": new["overlay_text"],
            "tags": new["tags"],
            "scenes": new["scenes"],
        }
        try:
            result = _post(args.base, args.token, "images/update", payload)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        if result.get("success"):
            ok += 1
        else:
            fail += 1
            print(f"#{r['no']} 写回失败: {result.get('error')}", flush=True)
    print(f"写回完成：成功 {ok}，失败 {fail}，跳过 {len(skip)}，保留原分类 {kept}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--db", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    a = sub.add_parser("apply")
    a.add_argument("--skip", default="")
    a.add_argument("--keep-moves", default="")
    a.add_argument("--keep-nos", default="")
    for sp in (p, a):
        sp.add_argument("--base", required=True)
        sp.add_argument("--token", required=True)
        sp.add_argument("--out", required=True)
    args = parser.parse_args()
    preview(args) if args.cmd == "preview" else apply(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
