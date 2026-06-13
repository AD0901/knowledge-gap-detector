"""
校准模式模块
读取 calibration.csv，逐题检索，分析两组分数分布，推荐最佳相似度阈值。

核心算法：
- 将校准题按 known_in_kb 分为"库里有(1)"和"库里没有(0)"两组
- 在可配置步长内搜索，找使误判（FP + FN）总数最少的阈值
- 输出完整的分数分布诊断信息，供人工判断
"""

import csv
import os
from typing import List, Dict, Tuple

import numpy as np


def load_calibration(file_path: str) -> List[Dict]:
    """加载校准题集 CSV。格式: scenario, question, known_in_kb"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"校准文件不存在: {file_path}")

    questions: List[Dict] = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = {
                "scenario": row.get("scenario", "").strip(),
                "question": row.get("question", "").strip(),
                "known_in_kb": int(row.get("known_in_kb", "0").strip()),
            }
            if q["question"]:  # 跳过空行
                questions.append(q)
    return questions


def recommend_threshold(
    scores_known: List[float],
    scores_unknown: List[float],
    step: float = 0.005,
) -> Tuple[float, Dict]:
    """
    根据「库里有」和「库里没有」两组的分数分布，搜索最佳阈值。

    算法：
    在 [min_score, max_score] 范围内以 step 步长搜索，
    对每个候选阈值计算：
    - 误报 FP：库里没有但分数 ≥ 阈值（误将该题判为有覆盖）
    - 漏报 FN：库里有但分数 < 阈值（漏判为缺口）
    选总误判数最少的阈值。

    返回: (推荐阈值, 诊断信息)
    """
    # 边界情况
    if not scores_known and not scores_unknown:
        return 0.55, {"error": "两组数据均为空，无法校准，使用默认阈值 0.55"}
    if not scores_known:
        return 0.60, {
            "warning": "校准题集中没有 known_in_kb=1 的题目",
            "suggestion": "请添加正面样例（库里确实有覆盖的题目）",
        }
    if not scores_unknown:
        return 0.40, {
            "warning": "校准题集中没有 known_in_kb=0 的题目",
            "suggestion": "请添加负面样例（库里确实没有的题目）",
        }

    known_arr = np.array(scores_known)
    unknown_arr = np.array(scores_unknown)

    search_min = min(known_arr.min(), unknown_arr.min())
    search_max = max(known_arr.max(), unknown_arr.max())

    best_threshold = 0.5
    best_errors = len(scores_known) + len(scores_unknown)

    # 以 step 步长遍历所有候选阈值
    t = search_min - step
    while t <= search_max + step:
        fp = int(np.sum(unknown_arr >= t))  # 库里没有但判有
        fn = int(np.sum(known_arr < t))     # 库里有但判无
        total_errors = fp + fn

        if total_errors < best_errors:
            best_errors = total_errors
            best_threshold = round(t, 3)
        elif total_errors == best_errors:
            # 平局时取中间值
            best_threshold = round((best_threshold + t) / 2, 3)

        t += step

    # 在最佳阈值下的详细误判数
    best_fp = int(np.sum(unknown_arr >= best_threshold))
    best_fn = int(np.sum(known_arr < best_threshold))

    # 可分离性判断
    gap = abs(known_arr.mean() - unknown_arr.mean())
    pooled_std = np.sqrt((known_arr.var() + unknown_arr.var()) / 2) if len(known_arr) > 0 else 0
    separability_ok = gap > max(pooled_std * 0.5, 0.05) if pooled_std > 0 else gap > 0.05

    diagnosis = {
        "best_threshold": best_threshold,
        "best_errors": int(best_errors),
        "false_positives": best_fp,
        "false_negatives": best_fn,
        "separability_ok": separability_ok,
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
    }

    return best_threshold, diagnosis


def run_calibration(
    calibration_file: str,
    vector_store,
    top_k: int = 5,
) -> Dict:
    """
    执行校准完整流程：

    1. 加载校准题集
    2. 逐题检索，记录最高相似度分数
    3. 按 known_in_kb 分组
    4. 自动推荐最佳阈值
    5. 输出完整诊断报告

    返回包含 results、diagnosis、recommended_threshold 的字典。
    """
    questions = load_calibration(calibration_file)
    if not questions:
        return {"error": "校准题集为空，请检查 data/calibration.csv"}

    known_count = sum(1 for q in questions if q["known_in_kb"] == 1)
    unknown_count = len(questions) - known_count

    print(f"\n{'='*60}")
    print(f"  🔧 校准模式")
    print(f"{'='*60}")
    print(f"  校准题数: {len(questions)}  (库里有: {known_count}, 库里没有: {unknown_count})")
    print()

    results: List[Dict] = []
    scores_known: List[float] = []
    scores_unknown: List[float] = []

    for i, q in enumerate(questions):
        search_results = vector_store.search(q["question"], top_k=top_k)

        max_score = search_results[0]["score"] if search_results else 0.0
        top_text = search_results[0]["text"] if search_results else ""

        if q["known_in_kb"] == 1:
            scores_known.append(max_score)
        else:
            scores_unknown.append(max_score)

        label = "✅ 库里有" if q["known_in_kb"] == 1 else "❌ 库里没有"
        preview = q["question"][:45] + ("..." if len(q["question"]) > 45 else "")
        chunk_preview = top_text[:80].replace("\n", " ") if top_text else "(无)"
        print(f"  [{i+1:>2}/{len(questions)}] {label} | {preview}")
        print(f"           最高分: {max_score:.4f} | 最佳匹配: {chunk_preview}")

    # ---- 阈值推荐 ----
    threshold, diagnosis = recommend_threshold(scores_known, scores_unknown)

    # ---- 打印诊断报告 ----
    print(f"\n{'='*60}")
    print(f"  📊 分数分布诊断")
    print(f"{'='*60}")

    if "error" in diagnosis:
        print(f"  ❌ {diagnosis['error']}")
        return {"questions": questions, "results": results, "diagnosis": diagnosis}
    if "warning" in diagnosis:
        print(f"  ⚠️  {diagnosis['warning']}")
        print(f"  💡 {diagnosis.get('suggestion', '')}")

    # 库里有组
    sk = diagnosis.get("scores_known", {})
    if sk.get("count", 0) > 0:
        print(f"  📗 库里有   (n={sk['count']})")
        print(f"     分数范围: {sk['min']:.4f} ~ {sk['max']:.4f}")
        print(f"     均值: {sk['mean']:.4f}  中位数: {sk['median']:.4f}  标准差: {sk['std']:.4f}")

    # 库里没有组
    su = diagnosis.get("scores_unknown", {})
    if su.get("count", 0) > 0:
        print(f"  📕 库里没有 (n={su['count']})")
        print(f"     分数范围: {su['min']:.4f} ~ {su['max']:.4f}")
        print(f"     均值: {su['mean']:.4f}  中位数: {su['median']:.4f}  标准差: {su['std']:.4f}")

    # 推荐阈值
    print(f"\n  {'─' * 56}")
    print(f"  🎯 推荐阈值: {threshold}")
    print(f"     该阈值下 — 误报: {diagnosis['false_positives']} 题, "
          f"漏报: {diagnosis['false_negatives']} 题")
    if diagnosis.get("separability_ok"):
        print(f"  ✅ 两组分数可分离性良好，阈值可信")
    else:
        print(f"  ⚠️  两组分数可分离性不佳！")
        print(f"      建议检查: 1) 文档内容是否与题目场景匹配")
        print(f"               2) 校准题标注是否正确")
        print(f"               3) embedding 模型是否适合中文文本")

    print(f"\n  💡 确认后请将阈值写入 config.py 或设置环境变量:")
    print(f"     export SIMILARITY_THRESHOLD={threshold}")
    print(f"{'='*60}\n")

    return {
        "questions": questions,
        "results": results,
        "diagnosis": diagnosis,
        "recommended_threshold": threshold,
    }
