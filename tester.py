"""
批量测试和报告生成模块

流程：
1. 读取 questions.csv（scenario, question）
2. 逐题向量检索 → 取最高相似度分数
3. 判定：最高分 ≥ 阈值 → covered，< 阈值 → gap
4. 输出 batch_results.csv（逐题明细）
5. 生成 report.html（场景维度可视化报告）

核心约束：
- 判定只看相似度分数，绝不看"检索返回了几条"
- 因为向量检索永远会返回 top-K 条最近邻，哪怕库里没有内容也会硬凑
- 用返回条数判断会导致缺口系统漏报、完善度虚高
"""

import csv
import os
from typing import List, Dict
from collections import defaultdict


# ============================================================
# 问题加载
# ============================================================


def load_questions(file_path: str) -> List[Dict]:
    """加载测试问题 CSV。格式: scenario, question"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"测试问题文件不存在: {file_path}")

    questions: List[Dict] = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = {
                "scenario": row.get("scenario", "").strip(),
                "question": row.get("question", "").strip(),
            }
            if q["question"]:
                questions.append(q)
    return questions


# ============================================================
# 判定逻辑（核心！）
# ============================================================


def judge(similarity_score: float, threshold: float) -> str:
    """
    判定逻辑：
    - 分数 ≥ 阈值 → "covered"（有覆盖）
    - 分数 < 阈值 → "gap"（真缺口）

    注意：判定只看相似度分数，绝不看检索返回条数。
    原因：向量检索永远返回 top-K 条结果，哪怕库里没有内容也会硬凑 K 条回来。
    """
    return "covered" if similarity_score >= threshold else "gap"


# ============================================================
# 答案生成（可选，默认关闭）
# ============================================================


def generate_answer(question: str, chunks: List[Dict]) -> str:
    """
    用 LLM 根据检索到的 chunk 生成答案。

    强制约束：
    - 只能基于提供的资料回答
    - 若资料中没有相关信息，必须明确说明
    """
    from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    import requests

    if not LLM_API_KEY:
        return "（未配置 LLM API Key，无法生成答案）"

    context = "\n\n---\n\n".join(
        f"[来源: {c.get('file_id', '')}]\n{c.get('text', '')}"
        for c in chunks[:3]
    )

    system_prompt = (
        "你是中国电信业务知识库助手。请严格根据下面提供的资料回答问题。\n"
        "规则：\n"
        "1. 只能使用提供的资料内容，禁止使用你自己的常识或外部知识。\n"
        '2. 如果提供的资料中没有相关信息，请直接回答："提供的资料中没有相关信息。"\n'
        "3. 回答要简洁、准确。"
    )

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
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}\n\n请根据以上资料回答："},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
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
    批量测试：逐题检索 + 判定 + 可选答案生成。

    返回: 逐题结果列表，每项包含 scenario/question/score/verdict/chunk 等内容。
    """
    questions = load_questions(questions_file)
    if not questions:
        print("  ⚠️ 测试问题列表为空")
        return []

    total = len(questions)
    print(f"\n{'='*60}")
    print(f"  📋 批量测试：{total} 道题 | 阈值: {threshold} | Top-K: {top_k}")
    print(f"{'='*60}\n")

    results: List[Dict] = []
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

        preview = q["question"][:50] + ("..." if len(q["question"]) > 50 else "")
        if enable_answer_gen and verdict == "covered" and search_results:
            print(f"  [{i+1:>2}/{total}] ✅ [{q['scenario']}] {preview} → {max_score:.4f} (生成答案...)")
            result["answer"] = generate_answer(q["question"], search_results)
        elif verdict == "gap":
            print(f"  [{i+1:>2}/{total}] ❌ [{q['scenario']}] {preview} → {max_score:.4f} 🔴缺口")
        else:
            print(f"  [{i+1:>2}/{total}] ✅ [{q['scenario']}] {preview} → {max_score:.4f}")

        results.append(result)

    return results


# ============================================================
# 场景维度汇总
# ============================================================


def summarize_by_scenario(results: List[Dict], documents_dir: str) -> Dict:
    """
    按业务场景汇总测试结果。

    每个场景包含：
    - total: 该场景总题数
    - covered: 覆盖数
    - gaps: 缺口数
    - completeness: 完善度百分比
    - gap_items: 缺口详情列表
    - has_documents: 该场景目录下是否有文档
    - is_whole_missing: 是否整场景缺失
    """
    # 获取有文档的场景
    doc_scenarios = set()
    if os.path.isdir(documents_dir):
        for d in os.listdir(documents_dir):
            if os.path.isdir(os.path.join(documents_dir, d)):
                doc_scenarios.add(d)

    scenarios = defaultdict(lambda: {
        "total": 0, "covered": 0, "gaps": 0, "gap_items": [],
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

    summary = {}
    for name, data in scenarios.items():
        if data["total"] > 0:
            data["completeness"] = round(data["covered"] / data["total"] * 100, 1)
        else:
            data["completeness"] = 0
        data["has_documents"] = name in doc_scenarios
        # 整场景缺失：没文档，或全部题目都是缺口
        data["is_whole_missing"] = (
            not data["has_documents"]
            or (data["total"] > 0 and data["gaps"] == data["total"])
        )
        summary[name] = dict(data)

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
    """生成场景维度的可视化 HTML 报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_questions = len(results)
    total_covered = sum(1 for r in results if r["verdict"] == "covered")
    total_gaps = total_questions - total_covered
    overall_completeness = (
        round(total_covered / total_questions * 100, 1)
        if total_questions > 0 else 0
    )

    # 整场景缺失列表
    whole_missing = sorted(
        name for name, s in scenario_summary.items() if s.get("is_whole_missing")
    )

    # 场景按完善度排序
    sorted_scenarios = sorted(
        scenario_summary.items(),
        key=lambda x: x[1]["completeness"],
        reverse=True,
    )

    # ---- 场景热力图 ----
    heatmap_rows = ""
    for name, s in sorted_scenarios:
        pct = s["completeness"]
        if s.get("is_whole_missing"):
            color = "#c62828"
            bg = "#ffebee"
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
        doc_status = "📁 有文档" if s.get("has_documents") else "⚠️ 无文档"

        heatmap_rows += f"""
        <tr>
            <td class="scenario-name">{name}</td>
            <td>{doc_status}</td>
            <td class="num">{s['total']}</td>
            <td class="num">{s['covered']}</td>
            <td class="num" style="color:{'#c62828' if s['gaps'] > 0 else '#2e7d32'}; font-weight:700;">{s['gaps']}</td>
            <td>
                <div class="bar-wrap">
                    <div class="bar" style="width:{bar_width}%; background:{color};"></div>
                    <span class="bar-text">{pct}%</span>
                </div>
            </td>
        </tr>"""

    # ---- 缺口详情 ----
    gap_sections = ""
    for name, s in sorted_scenarios:
        if not s["gap_items"]:
            continue
        is_missing = s.get("is_whole_missing")

        gap_sections += f"""
        <div class="gap-section {'whole-missing' if is_missing else ''}">
            <h3>
                {name}
                <span class="tag {'tag-red' if is_missing else 'tag-orange'}">
                    {'⚠ 整场景缺失' if is_missing else f'缺口 {s["gaps"]} 个'}
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
            chunk_preview = item["top_chunk_text"][:200].replace("<", "&lt;").replace(">", "&gt;")
            gap_sections += f"""
                    <tr>
                        <td>{item['question']}</td>
                        <td class="num">{item['max_similarity']:.4f}</td>
                        <td class="file-col">{item['top_chunk_file']}</td>
                        <td class="chunk-col">{chunk_preview}</td>
                    </tr>"""

        gap_sections += """
                </tbody>
            </table>
        </div>"""

    # ---- 完整 HTML ----
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
                     "Microsoft YaHei", "Noto Sans SC", sans-serif;
        background: #f0f2f5; color: #333; line-height:1.6;
    }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:24px; }}
    .hero {{
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white; padding: 36px 32px; border-radius: 14px; margin-bottom: 24px;
    }}
    .hero h1 {{ font-size:28px; margin-bottom:8px; letter-spacing:1px; }}
    .hero .sub {{ opacity:0.85; font-size:14px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px; }}
    .card {{
        background:white; padding:20px; border-radius:10px;
        box-shadow:0 2px 8px rgba(0,0,0,0.06); text-align:center;
    }}
    .card .val {{ font-size:36px; font-weight:700; }}
    .card .lbl {{ font-size:13px; color:#888; margin-top:6px; }}
    .card.gap .val {{ color:#c62828; }}
    .card.ok .val {{ color:#2e7d32; }}
    .card.pct .val {{ color:#1565c0; }}

    .alert {{
        padding:16px 20px; border-radius:10px; margin-bottom:20px; font-size:15px;
    }}
    .alert-red {{ background:#ffebee; border-left:4px solid #c62828; color:#b71c1c; }}
    .alert-red strong {{ color:#c62828; }}

    .block {{
        background:white; border-radius:10px; padding:24px; margin-bottom:20px;
        box-shadow:0 2px 8px rgba(0,0,0,0.04);
    }}
    .block h2 {{
        font-size:19px; margin-bottom:16px; color:#1a237e;
        border-bottom:2px solid #e8eaf6; padding-bottom:10px;
    }}

    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th {{ background:#f5f6fa; padding:10px 12px; text-align:left; font-weight:600;
          border-bottom:2px solid #ddd; white-space:nowrap; }}
    td {{ padding:10px 12px; border-bottom:1px solid #eee; vertical-align:top; }}
    tr:hover td {{ background:#fafbfc; }}
    .num {{ text-align:center; font-variant-numeric:tabular-nums; }}
    .file-col {{ font-size:12px; color:#888; max-width:140px; overflow:hidden;
                  text-overflow:ellipsis; white-space:nowrap; }}
    .chunk-col {{ font-size:13px; color:#555; max-width:400px; }}

    .bar-wrap {{ display:flex; align-items:center; gap:8px; min-width:140px; }}
    .bar {{
        height:22px; border-radius:4px; min-width:4px;
        transition:width 0.4s ease;
    }}
    .bar-text {{ font-size:13px; font-weight:600; white-space:nowrap; }}

    .gap-section {{ margin-bottom:28px; }}
    .gap-section h3 {{
        font-size:17px; margin-bottom:12px; display:flex; align-items:center; gap:10px;
    }}
    .tag {{
        font-size:11px; padding:3px 10px; border-radius:12px; font-weight:500;
    }}
    .tag-red {{ background:#ffcdd2; color:#b71c1c; }}
    .tag-orange {{ background:#fff3e0; color:#e65100; }}
    .whole-missing h3 {{ color:#c62828; }}

    .footer {{
        text-align:center; color:#aaa; font-size:12px; padding:24px; margin-top:8px;
    }}

    @media print {{
        body {{ background:white; }}
        .wrap {{ max-width:100%; }}
        .card, .block {{ box-shadow:none; border:1px solid #ddd; }}
    }}
</style>
</head>
<body>
<div class="wrap">

    <div class="hero">
        <h1>📊 知识库缺口探测报告</h1>
        <div class="sub">
            判定阈值: {threshold} &nbsp;|&nbsp;
            检索 Top-K: {top_k} &nbsp;|&nbsp;
            共 {total_questions} 道测试题
        </div>
    </div>

    <div class="cards">
        <div class="card pct">
            <div class="val">{overall_completeness}%</div>
            <div class="lbl">整体完善度</div>
        </div>
        <div class="card ok">
            <div class="val">{total_covered}</div>
            <div class="lbl">已覆盖</div>
        </div>
        <div class="card gap">
            <div class="val">{total_gaps}</div>
            <div class="lbl">真缺口</div>
        </div>
        <div class="card">
            <div class="val">{len(sorted_scenarios)}</div>
            <div class="lbl">业务场景</div>
        </div>
    </div>

    {f'''<div class="alert alert-red">
        <strong>⚠️ 整场景缺失警告：</strong>
        以下场景可能完全缺少文档或全部题目均为缺口 —
        {', '.join(whole_missing)}
    </div>''' if whole_missing else ''}

    <div class="block">
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

    <div class="block">
        <h2>🔍 真缺口详情（附最高分 &amp; 最相似片段，供人工复核）</h2>
        {gap_sections if gap_sections else '<p style="color:#2e7d32; font-size:15px;">🎉 所有题目均有覆盖，未发现知识缺口！</p>'}
    </div>

    <div class="footer">
        知识库缺口探测系统 &mdash; 用问答当探针，自动发现知识库的内容缺口
    </div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📊 可视化报告已保存: {output_path}")


# ============================================================
# 主流程
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
    """执行完整测试流程：批量测试 → CSV → 场景汇总 → HTML 报告"""
    # 1. 批量测试
    results = run_batch_test(
        questions_file, vector_store, threshold, top_k, enable_answer_gen
    )

    if not results:
        return [], {}

    # 2. 保存逐题 CSV
    if batch_csv_path:
        save_batch_csv(results, batch_csv_path)

    # 3. 场景维度汇总
    scenario_summary = summarize_by_scenario(results, documents_dir)

    # 4. 打印摘要
    total = len(results)
    covered = sum(1 for r in results if r["verdict"] == "covered")
    gaps = total - covered

    print(f"\n{'='*60}")
    print(f"  ✅ 测试完成")
    print(f"{'='*60}")
    print(f"  总题数:     {total}")
    print(f"  已覆盖:     {covered}  ({round(covered/total*100,1) if total else 0}%)")
    print(f"  真缺口:     {gaps}")
    print(f"  涉及场景:   {len(scenario_summary)}")

    whole_missing_list = sorted(
        name for name, s in scenario_summary.items() if s.get("is_whole_missing")
    )
    if whole_missing_list:
        print(f"  ⚠️ 整场景缺失: {', '.join(whole_missing_list)}")

    print()

    # 5. 生成 HTML 报告
    if report_html_path:
        generate_html_report(
            results, scenario_summary, threshold, top_k, report_html_path
        )

    return results, scenario_summary
