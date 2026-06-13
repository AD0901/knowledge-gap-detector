"""
增量更新模块
基于 MD5 文件指纹实现增量：只重算变动的文件，未变文件零开销跳过。

核心逻辑：
1. 扫描 data/documents/ 所有文件，计算 MD5
2. 与 file_index.json 比对，发现新增/修改/删除
3. 新增文件 → 清洗切片入库
4. 修改文件 → 先按文件 ID 从向量库删旧 chunk，再入库新内容
5. 删除文件 → 按文件 ID 从向量库移除全部 chunk
6. 未变文件 → 跳过
7. 更新 file_index.json

效果：几十个文档改 1 个，重建从约 30 分钟降到约 1 分钟。
"""

import os
import json
import hashlib
from typing import Dict, Tuple, List

from document_processor import process_document, chunk_text


# ============================================================
# 文件指纹
# ============================================================


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希（分块读取，大文件友好）"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
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
    """保存文件索引到磁盘，确保父目录存在"""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def scan_document_files(documents_dir: str) -> Dict[str, str]:
    """
    递归扫描文档目录，返回 {相对路径: 绝对路径}。

    只扫描场景子目录下的文件，跳过隐藏文件和目录。
    """
    files: Dict[str, str] = {}
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
    documents_dir: str,
    index: Dict[str, str],
) -> Dict:
    """
    和文件索引比对，检测新增、修改、删除、未变的文件。

    返回:
    {
        "added":    [(rel_path, abs_path), ...],
        "modified": [(rel_path, abs_path), ...],
        "deleted":  [rel_path, ...],
        "unchanged":[rel_path, ...],
    }
    """
    current_files = scan_document_files(documents_dir)
    current_paths = set(current_files.keys())
    indexed_paths = set(index.keys())

    added: List[Tuple[str, str]] = []
    modified: List[Tuple[str, str]] = []
    unchanged: List[str] = []

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
    增量更新知识库的核心流程。

    步骤：
    1. 加载文件索引
    2. 扫描文档目录，检测变化
    3. 逐个处理：删除 → 修改（先删后增） → 新增
    4. 保存更新后的文件索引

    返回操作统计字典。
    """
    index = load_file_index(index_path)
    changes = detect_changes(documents_dir, index)

    stats = {
        "added": 0,
        "modified": 0,
        "deleted": 0,
        "unchanged": len(changes.get("unchanged", [])),
        "chunks_added": 0,
        "chunks_deleted": 0,
        "errors": 0,
    }

    # 预览变更
    if not any([changes["added"], changes["modified"], changes["deleted"]]):
        print("  ✅ 所有文档无变化，无需更新")
        return stats

    print(f"\n  📋 变更预览:")
    if changes["added"]:
        print(f"     新增: {len(changes['added'])} 个文件")
    if changes["modified"]:
        print(f"     修改: {len(changes['modified'])} 个文件")
    if changes["deleted"]:
        print(f"     删除: {len(changes['deleted'])} 个文件")
    if changes["unchanged"]:
        print(f"     不变: {len(changes['unchanged'])} 个文件")
    print()

    # ---- 1. 处理删除 ----
    for rel_path in changes["deleted"]:
        try:
            n = vector_store.delete_by_file(rel_path)
            del index[rel_path]
            print(f"  🗑  删除: {rel_path} ({n} 个旧 chunk 已移除)")
            stats["deleted"] += 1
            stats["chunks_deleted"] += n
        except Exception as e:
            print(f"  ❌ 删除失败 {rel_path}: {e}")
            stats["errors"] += 1

    # ---- 2. 处理修改（先删旧 chunk，再入库新内容） ----
    for rel_path, abs_path in changes["modified"]:
        try:
            scenario = os.path.dirname(rel_path)

            # 先删旧
            n_deleted = vector_store.delete_by_file(rel_path)
            stats["chunks_deleted"] += n_deleted

            # 再增新
            text = process_document(abs_path)
            if not text:
                print(f"  ⚠  文档清洗后为空: {rel_path}")
                continue

            chunks = chunk_text(text, chunk_size, overlap)
            chunk_dicts = [{"text": c, "index": i} for i, c in enumerate(chunks)]
            n_added = vector_store.add_chunks(rel_path, scenario, chunk_dicts)

            new_hash = compute_file_hash(abs_path)
            index[rel_path] = new_hash

            print(f"  ✏  修改: {rel_path} (删 {n_deleted} → 加 {n_added} 新 chunk)")
            stats["modified"] += 1
            stats["chunks_added"] += n_added
        except Exception as e:
            print(f"  ❌ 处理失败 {rel_path}: {e}")
            stats["errors"] += 1

    # ---- 3. 处理新增 ----
    for rel_path, abs_path in changes["added"]:
        try:
            scenario = os.path.dirname(rel_path)
            text = process_document(abs_path)
            if not text:
                print(f"  ⚠  文档清洗后为空: {rel_path}")
                continue

            chunks = chunk_text(text, chunk_size, overlap)
            chunk_dicts = [{"text": c, "index": i} for i, c in enumerate(chunks)]
            n_added = vector_store.add_chunks(rel_path, scenario, chunk_dicts)

            new_hash = compute_file_hash(abs_path)
            index[rel_path] = new_hash

            print(f"  ➕ 新增: {rel_path} ({n_added} 个 chunk)")
            stats["added"] += 1
            stats["chunks_added"] += n_added
        except Exception as e:
            print(f"  ❌ 处理失败 {rel_path}: {e}")
            stats["errors"] += 1

    # ---- 4. 保存索引 ----
    save_file_index(index_path, index)

    # 总结
    total_chunks = vector_store.count()
    print(f"\n  📊 向量库: {total_chunks} 个 chunk")
    return stats
