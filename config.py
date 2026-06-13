"""
知识库缺口探测系统 —— 配置文件
支持环境变量覆盖：同名环境变量优先级高于此文件中的值。
"""

import os

# ============================================================
# LLM / API 配置（可选：仅在开启答案生成时需要）
# ============================================================
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ============================================================
# Embedding 配置
#   方式一 local：使用本地 BGE 中文模型（免费，推荐中文电信文本）
#   方式二 api：使用 OpenAI 兼容 API 的 embedding 接口
# ============================================================
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")

# --- API 模式 ---
EMBEDDING_API_BASE_URL = os.getenv("EMBEDDING_API_BASE_URL", LLM_BASE_URL)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "text-embedding-3-small")

# --- 本地模式 ---
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"
)

# HuggingFace 镜像（国内用户必备！）
# 可选值: "https://hf-mirror.com" 、 "https://huggingface.co" 、 空字符串 = 不设代理
# 用法: export HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")

# ============================================================
# 向量库配置（ChromaDB 持久化目录）
# ============================================================
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")

# ============================================================
# 切片配置
# ============================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))        # 每片字数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))   # 片间重叠字数

# ============================================================
# 检索 & 判定配置
# ============================================================
TOP_K = int(os.getenv("TOP_K", "5"))                    # 检索返回条数

# 核心阈值（必须通过校准确定，不能拍脑袋！）
# 用法: export SIMILARITY_THRESHOLD=0.61
SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.605")
)

# ============================================================
# 答案生成开关（默认关闭）
# ============================================================
ENABLE_ANSWER_GEN = os.getenv("ENABLE_ANSWER_GEN", "false").lower() in (
    "true", "1", "yes"
)

# ============================================================
# 数据文件路径
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
QUESTIONS_FILE = os.path.join(DATA_DIR, "questions.csv")
CALIBRATION_FILE = os.path.join(DATA_DIR, "calibration.csv")
FILE_INDEX_FILE = os.path.join(DATA_DIR, "file_index.json")

# ============================================================
# 输出路径
# ============================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BATCH_RESULTS_FILE = os.path.join(OUTPUT_DIR, "batch_results.csv")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
REPORT_FILE = os.path.join(REPORT_DIR, "report.html")

# ============================================================
# 路径校验：确保必要目录存在
# ============================================================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
