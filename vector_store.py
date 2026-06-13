"""
向量存储和检索模块
基于 ChromaDB，支持按 ID 增删改查，embedding 支持本地 BGE 和 API 两种模式。
"""

import os
import hashlib
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np


class EmbeddingProvider:
    """Embedding 提供者：支持本地 BGE 模型和 API 两种方式"""

    def __init__(self, mode: str = "local"):
        self.mode = mode
        self._model = None  # 延迟加载

    def _ensure_local_model(self):
        if self._model is not None:
            return
        from config import LOCAL_EMBEDDING_MODEL

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

        print(f"  加载本地 embedding 模型: {LOCAL_EMBEDDING_MODEL} ...")
        self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print("  ✓ 模型加载完成")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对一批文本生成 embedding 向量"""
        if self.mode == "api":
            return self._embed_api(texts)
        else:
            return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        self._ensure_local_model()
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        from config import EMBEDDING_API_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_API_MODEL
        import requests

        if not EMBEDDING_API_KEY:
            raise ValueError("API 模式下必须设置 EMBEDDING_API_KEY")

        url = f"{EMBEDDING_API_BASE_URL.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": EMBEDDING_API_MODEL,
            "input": texts,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 按输入顺序排列
        return [item["embedding"] for item in data["data"]]


class VectorStore:
    """基于 ChromaDB 的向量存储，支持按 ID 增删改查"""

    COLLECTION_NAME = "knowledge_gap_chunks"

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )
        self._embedding_provider: Optional[EmbeddingProvider] = None

    @property
    def embedding(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            from config import EMBEDDING_MODE
            self._embedding_provider = EmbeddingProvider(mode=EMBEDDING_MODE)
        return self._embedding_provider

    # ---- 数据操作 ----

    def add_chunks(
        self, file_id: str, scenario: str, chunks: List[Dict]
    ) -> int:
        """
        添加一个文件的所有 chunk。
        chunks: [{"text": str, "index": int}, ...]
        返回添加的 chunk 数量。
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        ids = [
            self._make_chunk_id(file_id, c["index"]) for c in chunks
        ]
        metadatas = [
            {"file_id": file_id, "scenario": scenario, "chunk_index": c["index"]}
            for c in chunks
        ]

        embeddings = self.embedding.embed_texts(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_by_file(self, file_id: str) -> int:
        """按文件 ID 删除该文件的所有 chunk，返回删除数量"""
        # 先查有哪些
        existing = self.collection.get(
            where={"file_id": file_id}
        )
        if not existing["ids"]:
            return 0
        self.collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict]:
        """
        向量检索，返回 top_k 条最相似结果。
        返回: [{"score": float, "text": str, "file_id": str, "scenario": str, "chunk_index": int}, ...]
        """
        query_embedding = self.embedding.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB 默认用 L2 distance 或 cosine distance，
                # 我们统一存的是 cosine distance（1 - cosine_similarity），
                # 转为相似度分数
                dist = results["distances"][0][i]
                # ChromaDB 的 distance 范围取决于配置，这里做归一化处理
                # 对于 cosine distance: 0 = 完全相同, 2 = 完全相反
                # 转为相似度: 1 - distance/2 → 映射到 [0, 1]
                similarity = max(0.0, min(1.0, 1.0 - dist / 2.0))
                # 更精确地：如果 distance 看起来已经是 (1-cos)，直接用 1-dist
                # ChromaDB cosine 模式下 distance = 1 - cosine_similarity
                # 直接用 1 - dist 即可
                _sim = 1.0 - dist
                # 用更合理的那个（取大值）
                similarity = max(similarity, _sim)

                items.append({
                    "score": round(similarity, 6),
                    "text": results["documents"][0][i],
                    "file_id": results["metadatas"][0][i].get("file_id", ""),
                    "scenario": results["metadatas"][0][i].get("scenario", ""),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                })

        return items

    def count(self) -> int:
        """返回向量库中 chunk 总数"""
        return self.collection.count()

    # ---- 辅助 ----

    @staticmethod
    def _make_chunk_id(file_id: str, chunk_index: int) -> str:
        raw = f"{file_id}#{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:24]


# ============================================================
# 获取向量存储单例
# ============================================================

_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store_instance
    if _store_instance is None:
        from config import VECTOR_STORE_DIR
        _store_instance = VectorStore(VECTOR_STORE_DIR)
    return _store_instance
