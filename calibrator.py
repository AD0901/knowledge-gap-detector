"""
校准模式模块
读取 calibration.csv，逐题检索，推荐最佳相似度阈值。
"""

import csv
import os
from typing import List, Dict, Tuple
import numpy as np


def load_calibration(file_path: str) -> List[Dict]:
    """加载校准题集 CSV。格式: scenario, question, known_in_kb"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"校准文件不存在: {file_path}")

    questions = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "scenario": row.get("scenario", "").strip(),
                "question": row.get("question", "").strip(),
                "known_in_kb": int(row.get("known_in_kb", "0").strip()),
            })
    return questions


def recommend_threshold(scores_known: List[float], scores_unknown: List[float]) -> Tuple[float, Dict]:
    """
    根据「库里有」和「库里没有」两组的分数分布，推荐最佳阈值。

    算法：在 [min_score, max_score] 范围内以 0.01 步长搜索，
    找到使 (误报 + 漏报) 最少的分数线。

    返回: (推荐阈值, 诊断信息)
    """
    if not scores_known or not scores_unknown:
        # 如果有一组为空，用中位数策略
        all_scores = scores_known + scores_unknown
        if not all_scores:
            return 0.55, {"error": "无有效分数数据"}
        return round(np.median(all_scores), 2), {
            "method": "median_fallback",
            "reason": "一组数据为空，使用全体中位数",
        }

    known_arr = np.array(scores_known)
    unknown_arr = np.array(scores_unknown)

    min_score = min(min(known_arr), min(unknown_arr))
    max_score = max(max(known_arr), max(unknown_arr))

    best_threshold = 0.5
    best_errors = len(scores_known) + len(scores_unknown)  # 初始化为最坏情况

    # 以 0.005 步长搜索
    step = 0.005
    results_by_threshold = []

    t = min_score - step
    while t <= max_score + step:
        # 误报：库里没有但判定为有（分数 >= 阈值）
        false_positives = np.sum(unknown_arr >= t)
        # 漏报：库里有但判定为无（分数 < 阈值）
        false_negatives = np.sum(known_arr < t)
        total_errors = false_positives + false_negatives

        results_by_threshold.append({
            "threshold": round(t, 3),
            "fp": int(false_positives),
            "fn": int(false_negatives),
            "total_errors": int(total_errors),
        })

        if total_errors < best_errors:
            best_errors = total_errors
            best_threshold = round(t, 3)
        elif total_errors == best_errors:
            # 平局时取中间值
            best_threshold = round((best_threshold + t) / 2, 3)

        t += step

    # 诊断信息
    diagnosis = {
        "scores_known": {
            "count": len(scores_known),
            "min": round(float(known_arr.min()), 4),
            "max": round(float(known_arr.max()), 4),
            "mean": round(float(known_arr.mean()), 4),
            "median": round(float(np.median(known_arr)), 4),
            "std": round(float(known_arr.std()), 4),
        },
        "scores_unknown": {
            "count": len(scores_unknown),
            "min": round(float(unknown_arr.min()), 4),
            "max": round(float(unknown_arr.max()), 4),
            "mean": round(float(unknown_arr.mean()), 4),
            "median": round(float(np.median(unknown_arr)), 4),
            "std": round(float(unknown_arr.std()), 4),
        },
        "best_threshold": best_threshold,
        "best_errors": int(best_errors),
        "false_positives": 0,
        "false_negatives": 0,
        "separability_ok": False,
    }

    # 补充最佳阈值下的详细误判数
    fn = int(np.sum(known_arr < best_threshold))
    fp = int(np.sum(unknown_arr >= best_threshold))
    diagnosis["false_positives"] = fp
    diagnosis["false_negatives"] = fn

    # 可分离性判断：两组均值差距是否足够大
    gap = abs(known_arr.mean() - unknown_arr.mean())
    pooled_std = np.sqrt((known_arr.std() ** 2 + unknown_arr.std() ** 2) / 2)
    if pooled_std > 0:
        diagnosis["separability_ok"] = gap > pooled_std * 0.5
    else:
        diagnosis["separability_ok"] = gap > 0.05

    return best_threshold, diagnosis


def run_calibration(
    calibration_file: str,
    vector_store,
    top_k: int = 5,
) -> Dict:
    """
    执行校准流程：
    1. 加载校准题集
    2. 逐题检索，记录最高相似度分数
    3. 按 known_in_kb 分两组
    4. 推荐最佳阈值

    返回完整校准报告。
    """
    questions = load_calibration(calibration_file)
    if not questions:
        return {"error": "校准题集为空"}

    print(f"\n{'='*60}")
    print(f"  校准模式：共 {len(questions)} 道校准题")
    print(f"{'='*60}\n")

    known_count = sum(1 for q in questions if q["known_in_kb"] == 1)
    unknown_count = len(questions) - known_count
    print(f"  库里有: {known_count} 题, 库里没有: {unknown_count} 题\n")

    results = []
    scores_known = []
    scores_unknown = []

    for i, q in enumerate(questions):
        search_results = vector_store.search(q["question"], top_k=top_k)

        max_score = search_results[0]["score"] if search_results else 0.0
        top_text = search_results[0]["text"] if search_results else ""

        result = {
            "scenario": q["scenario"],
            "question": q["question"],
            "known_in_kb": q["known_in_kb"],
            "max_similarity": round(max_score, 6),
            "top_chunk_preview": top_text[:200] + ("..." if len(top_text) > 200 else ""),
        }
        results.append(result)

        if q["known_in_kb"] == 1:
            scores_known.append(max_score)
        else:
            scores_unknown.append(max_score)

        label = "库里有" if q["known_in_kb"] == 1 else "库里没有"
        print(f"  [{i+1}/{len(questions)}] [{label}] {q['question'][:40]}... → 最高分: {max_score:.4f}")

    # 推荐阈值
    print(f"\n{'='*60}")
    print(f"  分数分布分析")
    print(f"{'='*60}")

    if scores_known:
        arr = np.array(scores_known)
        print(f"  库里有 (n={len(scores_known)}): "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}, median={np.median(arr):.4f}")
    else:
        print(f"  库里有: (无数据)")

    if scores_unknown:
        arr = np.array(scores_unknown)
        print(f"  库里没有 (n={len(scores_unknown)}): "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}, median={np.median(arr):.4f}")
    else:
        print(f"  库里没有: (无数据)")

    threshold, diagnosis = recommend_threshold(scores_known, scores_unknown)

    print(f"\n  📊 推荐阈值: {threshold}")
    print(f"  📊 该阈值下 — 误报(库里没有判为有): {diagnosis['false_positives']}, "
          f"漏报(库里有判为无): {diagnosis['false_negatives']}")
    if diagnosis.get("separability_ok"):
        print(f"  ✅ 两组分数可分离性良好")
    else:
        print(f"  ⚠️ 两组分数可分离性不佳，建议检查文档内容和检索质量")
    print()

    return {
        "questions": questions,
        "results": results,
        "diagnosis": diagnosis,
        "recommended_threshold": threshold,
    }
