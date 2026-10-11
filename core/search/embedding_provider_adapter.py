"""Keep SDK vector storage while pacing requests and reporting observed dimensions."""

import asyncio
import numpy as np

from ..util.model_requests import is_rate_limited, model_request_gate


class EmbeddingProviderAdapter:
    def __init__(self, service, provider):
        self.service = service
        self.provider = provider

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def get_dim(self):
        return self.service._effective_dim(self.provider)

    async def _request(self, operation):
        key = self.service._extract_provider_id(self.provider)
        gate = model_request_gate(self.service.plugin)
        for attempt in range(3):
            try:
                return await gate.call(key, operation)
            except Exception as exc:
                if not is_rate_limited(exc) or attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    def _validate(self, vectors):
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1] or not np.isfinite(array).all():
            raise ValueError('嵌入模型返回了无效向量')
        self.service._record_dimension(self.provider, array.shape[1])
        return array.tolist()

    async def get_embedding(self, text):
        identity = self.service._provider_identity(self.provider)
        vector = await self._request(lambda: self.provider.get_embedding(text))
        if identity != self.service._provider_identity(self.provider):
            raise ValueError('嵌入模型在请求期间发生变化')
        return self._validate([vector])[0]

    async def get_embeddings(self, texts):
        if callable(getattr(self.provider, 'get_embeddings', None)):
            identity = self.service._provider_identity(self.provider)
            vectors = await self._request(lambda: self.provider.get_embeddings(texts))
            if identity != self.service._provider_identity(self.provider):
                raise ValueError('嵌入模型在请求期间发生变化')
            if len(vectors) != len(texts):
                raise ValueError('嵌入模型返回的向量数量与请求不一致')
            return self._validate(vectors)
        return [await self.get_embedding(text) for text in texts]

    async def get_embeddings_batch(self, texts, batch_size=16, tasks_limit=1, max_retries=1, progress_callback=None):
        # Reuse the framework's batch implementation with one task and one outer
        # attempt: shared pacing/retries already happen inside get_embeddings.
        from astrbot.core.provider.provider import EmbeddingProvider
        batch = getattr(EmbeddingProvider, 'get_embeddings_batch', None)
        if callable(batch):
            return await batch(self, texts, batch_size=batch_size, tasks_limit=1,
                               max_retries=1, progress_callback=progress_callback)
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self.get_embeddings(texts[start:start + batch_size]))
        return vectors
