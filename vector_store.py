"""
向量存储和检索模块
基于 ChromaDB，支持按 ID 增删改查。

Embedding 模式：
- local：本地 BGE 中文模型（默认，支持 HF 镜像加速下载）
- api：OpenAI 兼容 API 的 embedding 接口

核心设计：判定只看相似度分数，不看返回条数。
"""

import os
import hashlib
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


class EmbeddingProvider:
    """Embedding 提供者：支持本地 BGE 模型和 API 两种方式"""

    def __init__(self, mode: str = "local"):
        self.mode = mode
        self._model = None  # 延迟加载

    def _ensure_local_model(self):
        """加载本地 BGE 模型，支持 HF 镜像"""
        if self._model is not None:
            return

        from config import LOCAL_EMBEDDING_MODEL, HF_ENDPOINT

        # 设置 HuggingFace 镜像（国内用户加速下载）
        if HF_ENDPOINT:
            os.environ["HF_ENDPOINT"] = HF_ENDPOINT
            print(f"  🌐 使用 HuggingFace 镜像: {HF_ENDPOINT}")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

        print(f"  🔽 加载本地 embedding 模型: {LOCAL_EMBEDDING_MODEL}")
        print(f"     （首次运行会自动下载，约 1.3GB，请耐心等待）")

        self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"  ✅ 模型加载完成")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对一批文本生成 embedding 向量（L2 归一化）"""
        if self.mode == "api":
            return self._embed_api(texts)
        else:
            return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        self._ensure_local_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,   # L2 归一化，cosine similarity = dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        from config import EMBEDDING_API_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_API_MODEL
        import requests

        if not EMBEDDING_API_KEY:
            raise ValueError("API 模式下必须设置 EMBEDDING_API_KEY 环境变量")

        url = f"{EMBEDDING_API_BASE_URL.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": EMBEDDING_API_MODEL,
            "input": texts,
            "encoding_format": "float",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 按输入顺序排列，排序字段可能是 index 或直接有序
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]


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

        # 使用 cosine 距离度量：distance = 1 - cosine_similarity
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedding_provider: Optional[EmbeddingProvider] = None

    @property
    def embedding(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            from config import EMBEDDING_MODE
            self._embedding_provider = EmbeddingProvider(mode=EMBEDDING_MODE)
        return self._embedding_provider

    # ================================================================
    # 数据操作
    # ================================================================

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
        ids = [self._make_chunk_id(file_id, c["index"]) for c in chunks]
        metadatas = [
            {
                "file_id": file_id,
                "scenario": scenario,
                "chunk_index": c["index"],
            }
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
        existing = self.collection.get(where={"file_id": file_id})
        if not existing or not existing["ids"]:
            return 0
        self.collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        向量检索，返回 top_k 条最相似结果。

        返回格式：
        [
            {
                "score": float,       # 相似度分数 [0, 1]，越高越相关
                "text": str,           # chunk 原文
                "file_id": str,        # 所属文件
                "scenario": str,       # 所属场景
                "chunk_index": int,    # chunk 序号
            },
            ...
        ]
        """
        query_embedding = self.embedding.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items: List[Dict] = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]

                # ChromaDB cosine 空间: distance = 1 - cosine_similarity
                # 转换为相似度: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - dist))

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

    def clear_all(self):
        """清空向量库所有数据（危险操作）"""
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
        return count

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _make_chunk_id(file_id: str, chunk_index: int) -> str:
        """生成稳定的 chunk ID（基于 file_id + chunk_index 的 MD5）"""
        raw = f"{file_id}#{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:24]


# ============================================================
# 向量存储单例（懒加载，避免重复初始化 ChromaDB + 模型）
# ============================================================

_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取向量存储单例"""
    global _store_instance
    if _store_instance is None:
        from config import VECTOR_STORE_DIR
        _store_instance = VectorStore(VECTOR_STORE_DIR)
    return _store_instance
