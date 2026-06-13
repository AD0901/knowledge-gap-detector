"""
增量更新模块
基于文件指纹（MD5）实现增量：只处理新增/修改/删除的文件，不动没变的。
"""

import os
import json
import hashlib
from typing import Dict, Set
from document_processor import process_document, chunk_text


# ============================================================
# 文件指纹
# ============================================================


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ============================================================
# 文件索引管理
# ============================================================


def load_file_index(index_path: str) -> Dict[str, str]:
    """加载文件索引 {相对路径: MD5 hash}"""
    if not os.path.exists(index_path):
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_file_index(index_path: str, index: Dict[str, str]):
    """保存文件索引"""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def scan_document_files(documents_dir: str) -> Dict[str, str]:
    """
    扫描文档目录，返回 {相对路径: 绝对路径}。
    只扫描场景子目录下的文件。
    """
    files = {}
    if not os.path.isdir(documents_dir):
        return files

    for scenario in sorted(os.listdir(documents_dir)):
        scenario_dir = os.path.join(documents_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for fname in sorted(os.listdir(scenario_dir)):
            fpath = os.path.join(scenario_dir, fname)
            if fname.startswith(".") or not os.path.isfile(fpath):
                continue
            rel_path = os.path.join(scenario, fname)
            files[rel_path] = fpath
    return files


# ============================================================
# 差异检测
# ============================================================


def detect_changes(
    documents_dir: str, index: Dict[str, str]
) -> Dict[str, list]:
    """
    和文件索引比对，检测新增、修改、删除的文件。

    返回:
    {
        "added":   [(rel_path, abs_path), ...],
        "modified": [(rel_path, abs_path), ...],
        "deleted":  [rel_path, ...],
        "unchanged": [rel_path, ...],
    }
    """
    current_files = scan_document_files(documents_dir)
    current_paths = set(current_files.keys())
    indexed_paths = set(index.keys())

    added = []
    modified = []
    unchanged = []

    for rel_path in sorted(current_paths):
        abs_path = current_files[rel_path]
        current_hash = compute_file_hash(abs_path)

        if rel_path not in indexed_paths:
            added.append((rel_path, abs_path))
        elif current_hash != index[rel_path]:
            modified.append((rel_path, abs_path))
        else:
            unchanged.append(rel_path)

    deleted = sorted(indexed_paths - current_paths)

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "unchanged": unchanged,
    }


# ============================================================
# 增量更新主流程
# ============================================================


def incremental_update(
    documents_dir: str,
    index_path: str,
    vector_store,
    chunk_size: int = 400,
    overlap: int = 80,
) -> dict:
    """
    增量更新知识库：
    1. 扫描文档目录，和索引比对
    2. 新增文件：清洗切片入库
    3. 修改文件：先删旧再入库
    4. 删除文件：从向量库中移除
    5. 更新索引

    返回操作统计。
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    index = load_file_index(index_path)
    changes = detect_changes(documents_dir, index)

    stats = {
        "added": 0,
        "modified": 0,
        "deleted": 0,
        "unchanged": len(changes["unchanged"]),
        "chunks_added": 0,
        "chunks_deleted": 0,
    }

    # 处理删除
    for rel_path in changes["deleted"]:
        n = vector_store.delete_by_file(rel_path)
        print(f"  🗑 删除: {rel_path} ({n} chunks)")
        del index[rel_path]
        stats["deleted"] += 1
        stats["chunks_deleted"] += n

    # 处理修改（先删后增）
    for rel_path, abs_path in changes["modified"]:
        scenario = os.path.dirname(rel_path)
        n_deleted = vector_store.delete_by_file(rel_path)
        stats["chunks_deleted"] += n_deleted

        text = process_document(abs_path)
        chunks = chunk_text(text, chunk_size, overlap)
        chunk_dicts = [{"text": c, "index": i} for i, c in enumerate(chunks)]
        n_added = vector_store.add_chunks(rel_path, scenario, chunk_dicts)

        new_hash = compute_file_hash(abs_path)
        index[rel_path] = new_hash

        print(f"  ✏ 修改: {rel_path} (删 {n_deleted} 旧 chunk, 加 {n_added} 新 chunk)")
        stats["modified"] += 1
        stats["chunks_added"] += n_added

    # 处理新增
    for rel_path, abs_path in changes["added"]:
        scenario = os.path.dirname(rel_path)
        text = process_document(abs_path)
        chunks = chunk_text(text, chunk_size, overlap)
        chunk_dicts = [{"text": c, "index": i} for i, c in enumerate(chunks)]
        n_added = vector_store.add_chunks(rel_path, scenario, chunk_dicts)

        new_hash = compute_file_hash(abs_path)
        index[rel_path] = new_hash

        print(f"  ➕ 新增: {rel_path} ({n_added} chunks)")
        stats["added"] += 1
        stats["chunks_added"] += n_added

    # 保存索引
    save_file_index(index_path, index)

    total_chunks = vector_store.count()
    print(f"\n  总计: {total_chunks} 个 chunk 在向量库中")
    return stats
