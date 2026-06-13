"""
批量测试和报告生成模块
读取 questions.csv，逐题检索判定，生成 CSV 结果和 HTML 可视化报告。
"""

import csv
import os
import json
from typing import List, Dict
from collections import defaultdict

# ============================================================
# 问题加载
# ============================================================


def load_questions(file_path: str) -> List[Dict]:
    """加载测试问题 CSV。格式: scenario, question"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试问题文件不存在: {file_path}")

    questions = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "scenario": row.get("scenario", "").strip(),
                "question": row.get("question", "").strip(),
            })
    return questions


# ============================================================
# 判定逻辑
# ============================================================


def judge(similarity_score: float, threshold: float) -> str:
    """
    判定逻辑（核心！）：只看最高相似度分数。
    - >= 阈值 → covered
    - < 阈值  → gap
    """
    if similarity_score >= threshold:
        return "covered"
    return "gap"


# ============================================================
# 答案生成（可选）
# ============================================================


def generate_answer(question: str, chunks: List[Dict]) -> str:
    """
    用 LLM 根据检索到的 chunk 生成答案。
    强制约束：只能基于提供的资料回答，没有依据就明说。
    """
    from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    import requests

    if not LLM_API_KEY:
        return "（未配置 LLM API Key，无法生成答案）"

    context = "\n\n---\n\n".join(
        f"[来源: {c['file_id']}]\n{c['text']}" for c in chunks[:3]
    )

    system_prompt = (
        "你是中国电信业务知识库助手。请严格根据下面提供的资料回答问题。\n"
        "规则：\n"
        "1. 只能使用提供的资料内容，禁止使用你自己的常识或外部知识。\n"
        '2. 如果提供的资料中没有相关信息，请直接回答："提供的资料中没有相关信息。"\n'
        "3. 回答要简洁、准确。"
    )

    user_prompt = f"资料：\n{context}\n\n问题：{question}\n\n请根据以上资料回答："

    try:
        url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（答案生成失败: {e}）"


# ============================================================
# 批量测试
# ============================================================


def run_batch_test(
    questions_file: str,
    vector_store,
    threshold: float = 0.55,
    top_k: int = 5,
    enable_answer_gen: bool = False,
) -> List[Dict]:
    """
    批量测试：
    1. 加载问题
    2. 逐题检索 + 判定
    3. 可选生成答案
    返回逐题结果列表。
    """
    questions = load_questions(questions_file)
    if not questions:
        return []

    total = len(questions)
    print(f"\n{'='*60}")
    print(f"  批量测试：共 {total} 道题，阈值 = {threshold}")
    print(f"{'='*60}\n")

    results = []
    for i, q in enumerate(questions):
        search_results = vector_store.search(q["question"], top_k=top_k)

        max_score = search_results[0]["score"] if search_results else 0.0
        top_chunk = search_results[0] if search_results else None
        verdict = judge(max_score, threshold)

        result = {
            "scenario": q["scenario"],
            "question": q["question"],
            "max_similarity": round(max_score, 6),
            "verdict": verdict,
            "top_chunk_text": top_chunk["text"] if top_chunk else "",
            "top_chunk_file": top_chunk["file_id"] if top_chunk else "",
            "top_chunk_scenario": top_chunk["scenario"] if top_chunk else "",
            "answer": "",
        }

        # 可选答案生成
        if enable_answer_gen and verdict == "covered" and search_results:
            print(f"  [{i+1}/{total}] ✅ [{q['scenario']}] {q['question'][:50]}... → {max_score:.4f} (生成答案...)")
            result["answer"] = generate_answer(q["question"], search_results)
        elif verdict == "gap":
            print(f"  [{i+1}/{total}] ❌ [{q['scenario']}] {q['question'][:50]}... → {max_score:.4f} (缺口)")
        else:
            print(f"  [{i+1}/{total}] ✅ [{q['scenario']}] {q['question'][:50]}... → {max_score:.4f}")

        results.append(result)

    return results


# ============================================================
# 场景汇总
# ============================================================


def summarize_by_scenario(results: List[Dict], documents_dir: str) -> Dict:
    """
    按场景汇总测试结果。
    返回每个场景的：
    - total: 总题数
    - covered: 覆盖数
    - gaps: 缺口数（及缺口详情）
    - completeness: 完善度百分比
    - has_documents: 该场景目录下是否有文档
    """
    # 获取有文档的场景列表
    doc_scenarios = set()
    if os.path.isdir(documents_dir):
        for d in os.listdir(documents_dir):
            if os.path.isdir(os.path.join(documents_dir, d)):
                doc_scenarios.add(d)

    scenarios = defaultdict(lambda: {
        "total": 0,
        "covered": 0,
        "gaps": 0,
        "gap_items": [],
    })

    for r in results:
        s = r["scenario"]
        scenarios[s]["total"] += 1
        if r["verdict"] == "covered":
            scenarios[s]["covered"] += 1
        else:
            scenarios[s]["gaps"] += 1
            scenarios[s]["gap_items"].append({
                "question": r["question"],
                "max_similarity": r["max_similarity"],
                "top_chunk_text": r["top_chunk_text"][:300] if r["top_chunk_text"] else "",
                "top_chunk_file": r["top_chunk_file"],
            })

    # 补充计算
    summary = {}
    for scenario_name, data in scenarios.items():
        data["completeness"] = (
            round(data["covered"] / data["total"] * 100, 1)
            if data["total"] > 0 else 0
        )
        data["has_documents"] = scenario_name in doc_scenarios
        # 整场景缺失判定
        data["is_whole_missing"] = (
            not data["has_documents"]
            or (data["total"] > 0 and data["gaps"] == data["total"])
        )
        summary[scenario_name] = dict(data)

    return summary


# ============================================================
# CSV 输出
# ============================================================


def save_batch_csv(results: List[Dict], output_path: str):
    """保存逐题结果到 CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "scenario", "question", "max_similarity", "verdict",
        "top_chunk_text", "top_chunk_file", "answer",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  📄 逐题结果已保存: {output_path}")


# ============================================================
# HTML 可视化报告
# ============================================================


def generate_html_report(
    results: List[Dict],
    scenario_summary: Dict,
    threshold: float,
    top_k: int,
    output_path: str,
):
    """生成 HTML 可视化报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_questions = len(results)
    total_covered = sum(1 for r in results if r["verdict"] == "covered")
    total_gaps = total_questions - total_covered
    overall_completeness = (
        round(total_covered / total_questions * 100, 1) if total_questions > 0 else 0
    )

    # 整场景缺失列表
    whole_missing = [
        name for name, s in scenario_summary.items() if s.get("is_whole_missing")
    ]

    # 场景列表按完善度排序
    sorted_scenarios = sorted(
        scenario_summary.items(),
        key=lambda x: x[1]["completeness"],
        reverse=True,
    )

    # 构建场景热力图
    heatmap_rows = ""
    for name, s in sorted_scenarios:
        pct = s["completeness"]
        if s.get("is_whole_missing"):
            color = "#ff4444"
            bg = "#ffe0e0"
        elif pct >= 90:
            color = "#2e7d32"
            bg = "#e8f5e9"
        elif pct >= 70:
            color = "#f9a825"
            bg = "#fff8e1"
        elif pct >= 50:
            color = "#ef6c00"
            bg = "#fff3e0"
        else:
            color = "#c62828"
            bg = "#ffebee"

        bar_width = min(pct, 100)
        doc_status = "📁" if s.get("has_documents") else "⚠️ 无文档"

        heatmap_rows += f"""
        <tr>
            <td class="scenario-name">{name}</td>
            <td>{doc_status}</td>
            <td class="num">{s['total']}</td>
            <td class="num">{s['covered']}</td>
            <td class="num" style="color:{'#c62828' if s['gaps'] > 0 else '#2e7d32'}">{s['gaps']}</td>
            <td>
                <div class="bar-container">
                    <div class="bar-fill" style="width:{bar_width}%; background:{color}"></div>
                    <span class="bar-label">{pct}%</span>
                </div>
            </td>
        </tr>"""

    # 缺口详情
    gap_sections = ""
    for name, s in sorted_scenarios:
        if not s["gap_items"]:
            continue
        is_missing = s.get("is_whole_missing")
        gap_sections += f"""
        <div class="scenario-section {'whole-missing' if is_missing else ''}">
            <h3>
                {name}
                <span class="badge {'badge-danger' if is_missing else 'badge-warning'}">
                    {'⚠ 整场景缺失' if is_missing else f'{s["gaps"]} 个缺口'}
                </span>
            </h3>
            <table>
                <thead>
                    <tr>
                        <th style="width:35%">问题</th>
                        <th style="width:8%">最高分</th>
                        <th style="width:15%">来源文件</th>
                        <th style="width:42%">最相似片段</th>
                    </tr>
                </thead>
                <tbody>"""

        for item in s["gap_items"]:
            gap_sections += f"""
                    <tr>
                        <td>{item['question']}</td>
                        <td class="num">{item['max_similarity']:.4f}</td>
                        <td class="file-path">{item['top_chunk_file']}</td>
                        <td class="chunk-preview">{item['top_chunk_text'][:200]}</td>
                    </tr>"""

        gap_sections += """
                </tbody>
            </table>
        </div>"""

    # 完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识库缺口探测报告</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                     "Microsoft YaHei", sans-serif;
        background: #f5f7fa; color: #333; line-height:1.6;
    }}
    .container {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .header {{
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px;
    }}
    .header h1 {{ font-size:28px; margin-bottom:8px; }}
    .header .meta {{ opacity:0.85; font-size:14px; }}
    .stats {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
    .stat-card {{
        flex:1; min-width:180px; background:white; padding:20px;
        border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.08);
        text-align:center;
    }}
    .stat-card .value {{ font-size:32px; font-weight:700; }}
    .stat-card .label {{ font-size:13px; color:#666; margin-top:4px; }}
    .stat-card.gap .value {{ color:#c62828; }}
    .stat-card.covered .value {{ color:#2e7d32; }}
    .stat-card.completeness .value {{ color:#1565c0; }}

    .alert {{
        background:#fff3e0; border-left:4px solid #ef6c00;
        padding:16px 20px; border-radius:8px; margin-bottom:20px;
        font-size:15px;
    }}
    .alert-danger {{
        background:#ffebee; border-left-color:#c62828; color:#b71c1c;
    }}
    .alert strong {{ color:#c62828; }}

    .section {{ background:white; border-radius:10px; padding:24px; margin-bottom:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
    .section h2 {{ font-size:20px; margin-bottom:16px; color:#1a237e; border-bottom:2px solid #e8eaf6; padding-bottom:8px; }}

    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th {{ background:#f5f7fa; padding:10px 12px; text-align:left; font-weight:600;
          border-bottom:2px solid #e0e0e0; white-space:nowrap; }}
    td {{ padding:10px 12px; border-bottom:1px solid #eee; vertical-align:top; }}
    tr:hover {{ background:#fafafa; }}
    .num {{ text-align:center; font-variant-numeric:tabular-nums; }}
    .file-path {{ font-size:12px; color:#888; max-width:150px; overflow:hidden;
                  text-overflow:ellipsis; white-space:nowrap; }}
    .chunk-preview {{ font-size:13px; color:#555; max-width:400px; }}

    .bar-container {{
        display:flex; align-items:center; gap:8px; min-width:120px;
    }}
    .bar-fill {{
        height:20px; border-radius:4px; min-width:4px; transition:width 0.3s;
    }}
    .bar-label {{ font-size:12px; font-weight:600; white-space:nowrap; }}

    .scenario-section {{ margin-bottom:24px; }}
    .scenario-section h3 {{
        font-size:17px; margin-bottom:10px; display:flex; align-items:center; gap:10px;
    }}
    .badge {{
        font-size:12px; padding:3px 10px; border-radius:12px; font-weight:500;
    }}
    .badge-danger {{ background:#ffcdd2; color:#b71c1c; }}
    .badge-warning {{ background:#fff3e0; color:#e65100; }}

    .whole-missing h3 {{ color:#c62828; }}

    .footer {{
        text-align:center; color:#999; font-size:12px; padding:20px; margin-top:20px;
    }}

    @media print {{
        body {{ background:white; }}
        .container {{ max-width:100%; }}
        .stat-card, .section {{ box-shadow:none; border:1px solid #ddd; }}
    }}
</style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>📊 知识库缺口探测报告</h1>
        <div class="meta">
            判定阈值: {threshold} | 检索 Top-K: {top_k} | 共 {total_questions} 道测试题
        </div>
    </div>

    <div class="stats">
        <div class="stat-card completeness">
            <div class="value">{overall_completeness}%</div>
            <div class="label">整体完善度</div>
        </div>
        <div class="stat-card covered">
            <div class="value">{total_covered}</div>
            <div class="label">已覆盖</div>
        </div>
        <div class="stat-card gap">
            <div class="value">{total_gaps}</div>
            <div class="label">真缺口</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(sorted_scenarios)}</div>
            <div class="label">业务场景</div>
        </div>
    </div>

    {f'''<div class="alert alert-danger">
        <strong>⚠️ 整场景缺失警告：</strong> 以下场景可能完全缺少文档或全部题目均为缺口 —
        {', '.join(whole_missing)}
    </div>''' if whole_missing else ''}

    <div class="section">
        <h2>📋 场景 × 完善度总览</h2>
        <table>
            <thead>
                <tr>
                    <th>业务场景</th>
                    <th>文档状态</th>
                    <th>总题数</th>
                    <th>覆盖数</th>
                    <th>缺口数</th>
                    <th>完善度</th>
                </tr>
            </thead>
            <tbody>
                {heatmap_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🔍 真缺口详情（附最高分 & 最相似片段）</h2>
        {gap_sections if gap_sections else '<p style="color:#2e7d32; font-size:15px;">🎉 所有题目均有覆盖，未发现知识缺口！</p>'}
    </div>

    <div class="footer">
        知识库缺口探测系统 · 基于向量检索 + 相似度阈值判定
    </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📊 可视化报告已保存: {output_path}")


# ============================================================
# 主流程：run_test
# ============================================================


def run_test(
    questions_file: str,
    documents_dir: str,
    vector_store,
    threshold: float = 0.55,
    top_k: int = 5,
    enable_answer_gen: bool = False,
    batch_csv_path: str = "",
    report_html_path: str = "",
):
    """执行完整测试流程并生成报告"""
    # 1. 批量测试
    results = run_batch_test(
        questions_file, vector_store, threshold, top_k, enable_answer_gen
    )

    # 2. 保存逐题 CSV
    if batch_csv_path:
        save_batch_csv(results, batch_csv_path)

    # 3. 场景汇总
    scenario_summary = summarize_by_scenario(results, documents_dir)

    # 4. 打印摘要
    total = len(results)
    covered = sum(1 for r in results if r["verdict"] == "covered")
    gaps = total - covered

    print(f"\n{'='*60}")
    print(f"  测试完成")
    print(f"{'='*60}")
    print(f"  总题数: {total}")
    print(f"  已覆盖: {covered} ({round(covered/total*100,1) if total else 0}%)")
    print(f"  真缺口: {gaps}")
    print(f"  涉及场景: {len(scenario_summary)}")

    whole_missing = [
        name for name, s in scenario_summary.items() if s.get("is_whole_missing")
    ]
    if whole_missing:
        print(f"\n  ⚠️ 整场景缺失: {', '.join(whole_missing)}")

    print()

    # 5. 生成 HTML 报告
    if report_html_path:
        generate_html_report(
            results, scenario_summary, threshold, top_k, report_html_path
        )

    return results, scenario_summary
