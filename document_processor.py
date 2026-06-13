"""
文档清洗和切片模块
支持 HTML（去标签、去噪声）、PDF（文本提取）、Excel（转 Markdown 表格）
"""

import re
import os
from typing import List, Dict
from html.parser import HTMLParser

# ============================================================
# HTML 清洗器
# ============================================================


class HTMLTextExtractor(HTMLParser):
    """提取 HTML 正文文本，跳过脚本和样式"""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "noscript", "nav", "footer", "header"}

        # 噪声类名/ID 关键词（导航栏、页脚、推荐等模板噪声）
        self.noise_keywords = [
            "nav", "menu", "footer", "header", "sidebar", "breadcrumb",
            "related", "recommend", "推荐", "相关", "上一篇", "下一篇",
            "copyright", "版权", "banner", "advertisement", "广告",
            "pagination", "分页", "comment", "评论", "share", "分享",
        ]
        self.in_skip = 0

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        # 检查标签名
        if tag_lower in self.skip_tags:
            self.in_skip += 1
            return
        # 检查 class/id 是否包含噪声关键词
        attr_str = " ".join(f"{k}={v}" for k, v in attrs if v).lower()
        if tag_lower == "div" and any(kw in attr_str for kw in self.noise_keywords):
            self.in_skip += 1

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in self.skip_tags:
            self.in_skip = max(0, self.in_skip - 1)
            return

    def handle_data(self, data):
        if self.in_skip > 0:
            return
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


def clean_html(file_path: str) -> str:
    """清洗 HTML 文件，去标签去噪声，返回纯文本"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    # 先用正则去掉 <script> <style> 整块
    html_content = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE
    )

    extractor = HTMLTextExtractor()
    extractor.feed(html_content)
    text = extractor.get_text()

    # 后处理：合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# PDF 文本提取
# ============================================================


def extract_pdf_text(file_path: str) -> str:
    """提取 PDF 文本（文本型 PDF；扫描件需预先 OCR）"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")

    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            texts.append(page_text.strip())
    doc.close()
    return "\n\n".join(texts)


# ============================================================
# Excel 转 Markdown 表格
# ============================================================


def excel_to_markdown(file_path: str) -> str:
    """将 Excel 文件转为 Markdown 表格文本"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("请安装 openpyxl: pip install openpyxl")

    wb = openpyxl.load_workbook(file_path, data_only=True)
    results = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        results.append(f"## 工作表: {sheet_name}\n")

        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            results.append("（空表）\n")
            continue

        # 过滤全空行
        rows = [r for r in rows if any(c is not None and str(c).strip() != "" for c in r)]
        if not rows:
            results.append("（空表）\n")
            continue

        # 生成 Markdown 表格
        max_cols = max(len(r) for r in rows)
        md_rows = []
        for i, row in enumerate(rows):
            cells = [str(c) if c is not None else "" for c in row]
            # 补齐列
            while len(cells) < max_cols:
                cells.append("")
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * max_cols) + " |")

        results.append("\n".join(md_rows) + "\n")

    wb.close()
    return "\n".join(results)


# ============================================================
# 文档分发器：根据扩展名调用对应清洗函数
# ============================================================


def process_document(file_path: str) -> str:
    """根据文件扩展名调用对应的清洗函数"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".html", ".htm"):
        return clean_html(file_path)
    elif ext == ".pdf":
        return extract_pdf_text(file_path)
    elif ext in (".xlsx", ".xls"):
        return excel_to_markdown(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    else:
        # 未知类型当纯文本尝试
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()


# ============================================================
# 文本切片
# ============================================================


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    将文本按字数切片，带重叠。

    优先按段落边界切，如果某段太长则按句子切，
    还不够则硬切。
    """
    if not text or not text.strip():
        return []

    chunks = []
    paragraphs = text.split("\n")

    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            if current:
                # 空行视为段落边界，优先在此处切
                if len(current) >= chunk_size * 0.6:
                    chunks.append(current)
                    # 重叠：保留末尾
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    current = overlap_text + "\n" if overlap_text else ""
                else:
                    current += "\n"
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current += ("\n" if current else "") + para
        else:
            # 当前段太长，先输出之前累积的
            if current:
                chunks.append(current)
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + "\n" if overlap_text else ""

            # 如果单段就超过 chunk_size，按句子切
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[。！？；.!?;])", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) <= chunk_size:
                        current += sent
                    else:
                        if current:
                            chunks.append(current)
                            overlap_text = current[-overlap:] if len(current) > overlap else current
                            current = overlap_text + sent if overlap_text else sent
                        else:
                            # 单个句子超过 chunk_size，硬切
                            for i in range(0, len(sent), chunk_size - overlap):
                                piece = sent[i:i + chunk_size]
                                if piece.strip():
                                    chunks.append(piece.strip())
                            current = ""
            else:
                current = para

    if current.strip():
        chunks.append(current.strip())

    # 过滤过短的 chunk
    chunks = [c for c in chunks if len(c) >= 20]
    return chunks


# ============================================================
# 完整流程：扫描文档目录，清洗+切片
# ============================================================


def scan_and_process_documents(
    documents_dir: str, chunk_size: int = 400, overlap: int = 80
) -> Dict[str, List[Dict]]:
    """
    扫描 documents_dir，按场景子目录处理所有文档。
    返回: {文件相对路径: [{"text": chunk文本, "index": 序号}, ...]}
    """
    results = {}
    if not os.path.isdir(documents_dir):
        return results

    for scenario in sorted(os.listdir(documents_dir)):
        scenario_dir = os.path.join(documents_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        for fname in sorted(os.listdir(scenario_dir)):
            fpath = os.path.join(scenario_dir, fname)
            if fname.startswith("."):
                continue
            if not os.path.isfile(fpath):
                continue

            rel_path = os.path.join(scenario, fname)
            try:
                text = process_document(fpath)
                if not text:
                    print(f"  ⚠ 文档为空，跳过: {rel_path}")
                    continue
                chunks = chunk_text(text, chunk_size, overlap)
                results[rel_path] = [
                    {"text": c, "index": i} for i, c in enumerate(chunks)
                ]
                print(f"  ✓ {rel_path} → {len(chunks)} 个 chunk")
            except Exception as e:
                print(f"  ✗ 处理失败 {rel_path}: {e}")

    return results
