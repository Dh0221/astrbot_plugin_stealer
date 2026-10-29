"""
自然语言情绪分析器
使用小模型对LLM回复进行语义分析，识别隐含情绪
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..search.text_similarity import calculate_hybrid_similarity, _has_negation_prefix

# 标记：模型明确选择不发送，用于区分"LLM 异常"和"模型说 none"
_EMOTION_ABSTAIN = object()

_LEGACY_EMOTION_TEMPLATE_HASH = '1914107e8c9b676649a0e92af223db633a7d56e0468dac44148df009e14d4201'
_EMOTION_ANALYSIS_DEFAULT_TEMPLATE = """根据本轮对话，先判断助手这句回复后面适不适合跟一张表情包，适合的话再提炼检索词。表情由助手发给当前用户。
可选情绪分类：{emotion_list}

第一步，判断要不要发。像真人群友一样克制：多数回复不需要表情包。下面这些情况一律不发，输出 {"send":false}：
- 用户消息为空，或者是系统、定时任务、“主动分享/冲浪/关心”之类的指令，而不是群友在聊天；
- 助手在认真回答问题、科普、讲道理、给建议、讲正事；
- 助手在道歉、认错、拒绝、解释误会；
- 对方在难过、倾诉、生病或遇到真麻烦，而助手在安慰；
- 回复很平淡，没有明显的情绪、梗或语气可以接；
- 想不出一张贴合这句话的表情包。
只有在闲聊、接梗、调侃、吐槽、起哄、撒娇、开心、惊讶、无语这类情绪明显的时候才发。

第二步，适合发时：
1. 以助手回复的立场和语气为准，结合用户消息判断语境，区分安慰与嘲笑、关心式催促与敌意反击。
2. query 用一条简短中文检索句表达助手想怎样回应，包含具体台词、语气或场景，并保留原句中的否定含义。画面和角色以对话中明确提到的内容为准。
3. emotions 从给定分类中按相关度选 1~3 项。只有助手明确在搞怪、发癫、阴阳怪气时才选 troll。

用户消息：{user_message}
助手回复：{llm_reply}

只输出合法 JSON，不发时输出 {"send":false}；发时例如：
{"send":true,"query":"安慰加班 辛苦了早点休息","emotions":["tired","sigh"]}
"""

_MEME_PICK_TEMPLATE = """你在帮一个 QQ 群聊机器人挑表情包。它刚回复了一句话，准备在后面跟一张表情包。下面是候选图，请挑出最贴合的一张；都不合适就选 0，宁可不发也不要乱发。

判断标准：
- 图上的文字和图的情绪要和助手这句回复的语气一致，不能相反（比如助手在道晚安，图上却写“起来嗨”）；
- 图上的文字不能和当前话题对不上、让人看不懂；
- 图要像是助手接着自己那句话发的，而不是在回应别的事。

用户消息：{user_message}
助手回复：{llm_reply}

候选：
{candidates}

只输出合法 JSON，例如 {"pick":2} 或 {"pick":0}。
"""


@dataclass
class EmotionQuery:
    """查询改写结果：检索句和用于辅助评分的情绪先验。"""

    should_send: bool = True
    search_query: str = ""
    emotion_priors: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str | None:
        return self.emotion_priors[0] if self.emotion_priors else None

    def to_cache(self) -> dict[str, Any]:
        return {
            "s": self.should_send,
            "q": self.search_query,
            "e": list(self.emotion_priors),
        }

    @classmethod
    def from_cache(cls, value: Any) -> EmotionQuery | None:
        if isinstance(value, cls):
            return value
        if isinstance(value, str) and value:
            return cls(True, value, [value])
        if isinstance(value, dict):
            priors = [str(item) for item in (value.get("e") or []) if item]
            query = str(value.get("q") or "").strip()
            should = bool(value.get("s", True))
            if not should:
                return cls(False, "", [])
            if query or priors:
                return cls(should, query, priors)
        return None


class NaturalEmotionAnalyzer:
    """自然语言情绪分析器 - 使用小模型理解LLM回复的真实情绪"""

    # 常量定义
    CACHE_MAX_SIZE = 1000  # 缓存最大容量
    TEXT_MAX_LENGTH = 200  # 文本最大长度

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        # 标记：上一次分析是否为模型主动 abstain
        self.last_analysis_abstained: bool = False
        # 通过插件配置读取模型和分类设置。
        self.plugin_config = plugin_instance.plugin_config
        self.categories: list[str] = self.plugin_config.get_categories()

        # 缓存机制
        self.analysis_cache: dict[str, Any] = {}
        self.cache_max_size: int = self.CACHE_MAX_SIZE
        self._cache_lock = asyncio.Lock()

        # 性能统计
        self.stats: dict[str, float | int] = {
            "total_analyses": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "successful_analyses": 0,
        }

        # 小模型提示词模板（从配置加载）
        self._emotion_analysis_template = self._load_emotion_analysis_template()

    def _load_emotion_analysis_template(self) -> str:
        """从配置加载情绪分析提示词；为空时回退到内置模板。"""
        custom = getattr(self.plugin_config, "emotion_analysis_prompt", "") or ""
        if custom.strip() and hashlib.sha256(custom.strip().encode("utf-8")).hexdigest() != _LEGACY_EMOTION_TEMPLATE_HASH:
            return str(custom).strip()
        return _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def _build_emotion_list_text(self) -> str:
        """构建分类描述文本（紧凑格式，供模板中 {emotion_list} 替换用）"""
        info_map = self.plugin_config.category_info or {}
        parts = []
        for key in self.categories:
            info = info_map.get(key, {})
            if isinstance(info, dict):
                name = str(info.get("name", "")).strip()
                desc = str(info.get("desc", "")).strip()
                desc_text = desc or name or key
            else:
                desc_text = key
            parts.append(f"{key}({desc_text})")
        return ", ".join(parts) if parts else ", ".join(self.categories)

    async def analyze_emotion(
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        *,
        user_message: str = "",
    ) -> EmotionQuery | None:
        """分析文本并生成表情检索查询。

        Returns:
            EmotionQuery；失败或不该发送时返回 None（abstain 通过 last_analysis_abstained 区分）
        """
        if not llm_reply or len(llm_reply.strip()) < 3:
            return None

        # 清理文本
        cleaned_reply = self._clean_text(llm_reply)
        if not cleaned_reply:
            return None

        # 清理用户消息（用于缓存 key 和 prompt）
        cleaned_msg = self._clean_text(user_message) if user_message else ""

        # 检查缓存（缓存 key 同时包含用户消息和回复）
        cache_key = self._get_cache_key(cleaned_msg + "|||" + cleaned_reply)
        async with self._cache_lock:
            if cache_key in self.analysis_cache:
                self.stats["cache_hits"] += 1
                logger.debug(f"[情绪分析] 缓存命中: {cleaned_reply[:30]}...")
                cached = EmotionQuery.from_cache(self.analysis_cache[cache_key])
                if cached:
                    return cached

        # 不再走本地关键词快速路径：要不要发需要模型结合语境判断，
        # 本地匹配只在模型调用失败时兜底。

        # 执行 LLM 分析（传入用户消息作为上下文）
        start_time = time.time()
        parsed = await self._analyze_with_llm(
            event,
            cleaned_reply,
            user_message=cleaned_msg,
        )
        end_time = time.time()

        if parsed is _EMOTION_ABSTAIN:
            self.last_analysis_abstained = True
            return None
        self.last_analysis_abstained = False

        self.stats["total_analyses"] += 1
        response_time = (end_time - start_time) * 1000
        query = parsed if isinstance(parsed, EmotionQuery) else None
        if query is None and isinstance(parsed, str):
            query = EmotionQuery(True, cleaned_reply, [parsed])
        if query is None:
            fallback = self._local_keyword_match(cleaned_reply, fallback=True)
            if fallback:
                query = EmotionQuery(True, cleaned_reply, [fallback])
                logger.debug(f"[情绪分析] LLM失败，降级匹配: {cleaned_reply[:30]}... → {fallback}")

        self._update_stats(response_time, query is not None)

        if query:
            if not query.search_query:
                query.search_query = cleaned_reply
            async with self._cache_lock:
                self._cache_result(cache_key, query.to_cache())
            logger.info(
                f"[情绪分析] {cleaned_reply[:30]}... → {query.emotion_priors} "
                f"q='{query.search_query[:40]}' ({response_time:.0f}ms)"
            )
        else:
            logger.warning(f"[情绪分析] 分析失败: {cleaned_reply[:30]}...")

        return query

    def _local_keyword_match(self, text: str, fallback: bool = False) -> str | None:
        """本地关键词匹配（快速路径/降级方案）

        Args:
            text: 要分析的文本
            fallback: 是否为降级模式（降级模式阈值更低）

        Returns:
            匹配的分类，无则返回 None
        """
        if not text:
            return None

        cfg = self.plugin.plugin_config
        if not cfg:
            return None

        # 获取关键词映射
        keyword_map = cfg.get_keyword_map() if hasattr(cfg, "get_keyword_map") else {}
        if not keyword_map:
            return None

        text_lower = text.lower()

        # 1. 精确匹配关键词（带否定词检测）
        for keyword, category in keyword_map.items():
            if keyword in text_lower and category:
                # 检查关键词前是否有否定词
                if _has_negation_prefix(text_lower, keyword):
                    continue  # 跳过被否定的关键词
                return category

        # 2. 分词匹配（降级模式下执行）
        if fallback:
            # 检查文本中是否有否定词+关键词的组合（语义反转）
            has_negated_emotion = any(
                _has_negation_prefix(text_lower, keyword) for keyword in keyword_map.keys()
            )

            # 如果存在否定词反转语义，不进行降级匹配
            if has_negated_emotion:
                return None

            best_match = None
            best_score = 0.0
            threshold = 0.3  # 降级模式阈值更低

            for category in self.categories:
                # 与分类名比较
                score = calculate_hybrid_similarity(text, category)
                # 与分类中文描述比较
                info = cfg.DEFAULT_CATEGORY_INFO.get(category, {})
                desc = info.get("desc", "") or info.get("name", "")
                if desc:
                    score = max(score, calculate_hybrid_similarity(text, desc))

                if score > best_score and score > threshold:
                    best_score = score
                    best_match = category

            return best_match

        return None

    @staticmethod
    def _render_emotion_analysis_template(
        template: str,
        *,
        emotion_list: str,
        llm_reply: str,
        user_message: str,
    ) -> str:
        """仅替换受支持的占位符，保留 JSON 花括号和未知占位符原样输出。"""
        mapping = {
            "emotion_list": emotion_list,
            "llm_reply": llm_reply,
            "user_message": user_message,
        }

        # 先保护 Python format 风格的转义花括号，避免替换到 {{placeholder}}。
        template = template.replace("{{", "\x00").replace("}}", "\x01")

        def _replace(match: re.Match[str]) -> str:
            return mapping.get(match.group(1), match.group(0))

        prompt = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _replace, template)
        return prompt.replace("\x00", "{").replace("\x01", "}")

    async def _analyze_with_llm(
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        *,
        user_message: str = "",
    ) -> EmotionQuery | object | None:
        """使用小模型生成检索查询。兼容旧的单分类名输出。"""
        try:
            # 获取文本模型提供商（优先使用配置的小模型）
            provider_id = await self._get_text_provider(event)
            if not provider_id:
                logger.warning("[情绪分析] 未找到可用的文本模型")
                return None

            # 构建提示词：用分类列表替换 {emotion_list}，再填入具体文本
            emotion_list = self._build_emotion_list_text()
            prompt = self._render_emotion_analysis_template(
                self._emotion_analysis_template,
                emotion_list=emotion_list,
                llm_reply=llm_reply,
                user_message=user_message if user_message else "",
            )

            # 调用LLM（限制 max_tokens 提升速度）
            logger.debug(f"[情绪分析] 调用LLM，provider_id={provider_id}")
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                max_tokens=120,
            )

            # 安全获取响应文本
            if not response:
                logger.warning("[情绪分析] LLM返回空响应")
                return None
            result_text = response.completion_text
            if not result_text:
                logger.warning("[情绪分析] LLM返回空文本")
                return None

            return self._parse_emotion_query(result_text, fallback_query=llm_reply)

        except Exception as e:
            error_msg = str(e)
            if "Provider" in error_msg or "提供商" in error_msg:
                logger.error(
                    f"[情绪分析] 模型提供商错误: {e}\n"
                    f"  配置的provider_id: {provider_id}\n"
                    f"  提示: 请检查插件配置中的'语境理解模型'是否有效，"
                    f"  或尝试清空该配置使用默认模型"
                )
            else:
                logger.error(f"[情绪分析] LLM调用失败: {e}")
            return None

    async def _get_text_provider(self, event: AstrMessageEvent) -> str | None:
        """获取文本模型提供商ID"""
        # 1. 优先使用插件配置的语境理解模型
        configured_provider = self.plugin.plugin_config.emotion_analysis_provider_id
        if configured_provider:
            logger.debug(f"[情绪分析] 尝试使用配置的提供商: {configured_provider}")
            return configured_provider

        # 2. 使用当前会话的模型
        try:
            current_provider = await self.plugin.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            logger.debug(f"[情绪分析] 使用当前会话模型: {current_provider}")
            return current_provider
        except Exception as e:
            logger.error(f"[情绪分析] 获取当前会话模型失败: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 仅压缩空白，保留回复原文的语义结构。
        cleaned = re.sub(r"\s+", " ", text.strip())

        # 限制长度（小模型处理能力有限）
        if len(cleaned) > self.TEXT_MAX_LENGTH:
            cleaned = cleaned[: self.TEXT_MAX_LENGTH] + "..."

        return cleaned

    def _parse_emotion_query(self, result_text: str, *, fallback_query: str) -> EmotionQuery | object | None:
        """解析 LLM 输出：优先 JSON 查询改写，兼容旧的单分类名。"""
        if not result_text:
            return None
        stripped = result_text.strip()
        lowered = stripped.lower()
        if re.match(r"^none[\s:：,，.。!！?？]*$", lowered):
            logger.info("[情绪分析] 模型输出 none，跳过发送")
            return _EMOTION_ABSTAIN

        data = self._extract_json_object(stripped)
        if isinstance(data, dict):
            send_flag = data.get("send", data.get("should_send"))
            if send_flag is False or str(send_flag).strip().lower() in {"false", "0", "no"}:
                logger.info("[情绪分析] 模型判断这句不适合配表情包，跳过发送")
                return _EMOTION_ABSTAIN
            # 提取检索句和情绪先验；发送时机由调用方控制。
            query = str(data.get("query") or data.get("search_query") or "").strip()
            raw_emotions = data.get("emotions") or data.get("emotion") or []
            if isinstance(raw_emotions, str):
                raw_emotions = [raw_emotions]
            priors: list[str] = []
            seen: set[str] = set()
            for item in raw_emotions:
                mapped = self._map_category(str(item))
                if mapped and mapped not in seen:
                    seen.add(mapped)
                    priors.append(mapped)
                if len(priors) >= 3:
                    break
            if not priors:
                mapped = self._parse_emotion_result(query or lowered)
                if mapped and mapped is not _EMOTION_ABSTAIN:
                    priors = [mapped]
            return EmotionQuery(True, query or fallback_query, priors)

        emotion = self._parse_emotion_result(lowered)
        if emotion is _EMOTION_ABSTAIN or emotion is None:
            return emotion
        return EmotionQuery(True, fallback_query, [emotion])

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _map_category(self, raw: str) -> str | None:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        if text in self.categories:
            return text
        cfg = self.plugin.plugin_config
        if cfg:
            try:
                return cfg.normalize_category_strict(text)
            except Exception:
                return None
        return None

    def _parse_emotion_result(self, result_text: str) -> str | None:
        """解析LLM返回的情绪结果

        支持从带解释的文字中提取分类名，如：
        - "happy" -> happy
        - "这个文本表达的是 happy 情绪" -> happy
        - "分类：sad" -> sad
        - "我觉得是 angry" -> angry
        """
        if not result_text:
            return None

        # 清理结果
        result = result_text.strip().lower()

        # 模型选择不发送
        if re.match(r"^none[\s:：,，.。!！?？]*$", result):
            logger.info("[情绪分析] 模型输出 none，跳过发送")
            return _EMOTION_ABSTAIN  # type: ignore

        cfg = self.plugin.plugin_config

        # 尝试从文本中提取已知的分类名
        # 优先匹配完整的单词（避免部分匹配如 "sad" 匹配到 "sadness"）
        for category in self.categories:
            # 检查是否是完整单词匹配（前后是边界或标点）
            pattern = (
                r"(?:^|[\s:：,，.。!！?？])" + re.escape(category) + r"(?:$|[\s:：,，.。!！?？])"
            )
            if re.search(pattern, result, re.IGNORECASE):
                logger.debug(f"[情绪分析] 从文本中提取分类: '{result}' -> '{category}'")
                return category

        # 尝试严格归一化（处理直接返回分类名的情况）
        if cfg:
            try:
                normalized = cfg.normalize_category_strict(result)
                logger.debug(f"[情绪分析] 解析结果: '{result}' -> '{normalized}'")
                if normalized:
                    return normalized
            except Exception as e:
                logger.error(f"[情绪分析] 解析异常: {e}")

        # Fallback: 直接匹配分类名
        if result in self.categories:
            logger.debug(f"[情绪分析] Fallback 匹配: '{result}'")
            return result

        logger.debug(f"[情绪分析] 无法从回复中解析分类: '{result_text}'")
        return None

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_result(self, cache_key: str, emotion: Any):
        """缓存分析结果"""
        # 清理过期缓存
        if len(self.analysis_cache) >= self.cache_max_size:
            # 移除最旧的一半缓存
            items = list(self.analysis_cache.items())
            self.analysis_cache = dict(items[len(items) // 2 :])

        self.analysis_cache[cache_key] = emotion

    def _update_stats(self, response_time: float, success: bool):
        """更新性能统计"""
        # 更新平均响应时间
        total = self.stats["total_analyses"]
        current_avg = self.stats["avg_response_time"]
        self.stats["avg_response_time"] = (current_avg * (total - 1) + response_time) / total

        if success:
            self.stats["successful_analyses"] += 1

    def get_stats(self) -> dict:
        """获取性能统计"""
        total = self.stats["total_analyses"]
        cache_hits = self.stats["cache_hits"]
        grand_total = total + cache_hits  # 总请求数 = LLM调用 + 缓存命中
        if grand_total == 0:
            return {"message": "暂无分析数据"}

        cache_hit_rate = (cache_hits / grand_total) * 100
        success_rate = (self.stats["successful_analyses"] / total) * 100 if total > 0 else 0.0

        return {
            "total_analyses": grand_total,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{self.stats['avg_response_time']:.0f}ms",
            "cache_size": len(self.analysis_cache),
        }

    async def clear_cache(self):
        """清空缓存"""
        async with self._cache_lock:
            self.analysis_cache.clear()
        logger.info("[情绪分析] 缓存已清空")

    async def pick_candidate(
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        candidates: list[dict[str, Any]],
        *,
        user_message: str = "",
    ) -> int | None:
        """让模型从候选中挑一张最贴合的；返回候选下标，都不合适返回 None。

        模型调用失败时返回 -1，调用方按原有排序结果处理。
        """
        if not candidates:
            return None
        lines = []
        for number, item in enumerate(candidates, start=1):
            overlay = str(item.get("overlay_text") or "").strip() or "无"
            desc = str(item.get("desc") or "").strip() or "无"
            category = str(item.get("category") or "").strip()
            lines.append(f"{number}. 分类={category}；图上文字：{overlay}；画面：{desc}")
        prompt = (
            _MEME_PICK_TEMPLATE.replace("{user_message}", self._clean_text(user_message) or "（无）")
            .replace("{llm_reply}", self._clean_text(llm_reply))
            .replace("{candidates}", "\n".join(lines))
        )
        try:
            provider_id = await self._get_text_provider(event)
            if not provider_id:
                return -1
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                max_tokens=60,
            )
            text = str(getattr(response, "completion_text", "") or "")
        except Exception as e:
            logger.warning(f"[挑图] 模型调用失败，按排序结果发送: {e}")
            return -1
        data = self._extract_json_object(text) or {}
        raw = data.get("pick", data.get("index"))
        if raw is None:
            match = re.search(r"\d+", text)
            raw = match.group(0) if match else None
        try:
            number = int(raw)
        except (TypeError, ValueError):
            logger.warning(f"[挑图] 无法解析模型输出，按排序结果发送: {text[:60]}")
            return -1
        if number <= 0 or number > len(candidates):
            logger.info("[挑图] 模型认为候选都不合适，这次不发")
            return None
        logger.info(f"[挑图] 选中第 {number} 张: {candidates[number - 1].get('path', '')}")
        return number - 1

    async def analyze_for_reply(
        self, event: AstrMessageEvent, llm_reply: str, *, user_message: str = ""
    ) -> EmotionQuery | None:
        """自动发送入口：关闭分析时直接跳过，其余逻辑由分析器统一处理。"""
        if not self.plugin_config.enable_natural_emotion_analysis:
            return None
        return await self.analyze_emotion(event, llm_reply, user_message=user_message)
