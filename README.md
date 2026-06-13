# 🔍 knowledge-gap-detector

**用问答当探针，自动找出知识库里答不出来的内容缺口。**

这是一个"知识库质量检测器"——批量跑一批测试问题，逐题向量检索，按相似度阈值判定哪些题知识库能覆盖、哪些是真缺口，最后生成按业务场景分类的可视化完善度报告。它不是问答机器人。

---

## 💡 核心思路

### 为什么做"检测器"而非"问答机器人"

问答机器人的目标是"给出答案"，哪怕答错了也会硬凑。而本工具的定位是**审计**——告诉你知识库的薄弱环节在哪里、哪个场景缺得多。报告里每条缺口都附带「最高相似度分数 + 最相似的那段 chunk」，方便你人工复核是不是误判。

### 判定逻辑：相似度阈值二分法

```
每道测试题 → 向量检索 → 取最高相似度分数
  ├─ 最高分 ≥ 阈值 → ✅ 有覆盖
  └─ 最高分 < 阈值 → ❌ 真缺口
```

### ⚠️ 关键硬约束：不看返回条数，只看分数

这是一个踩过的坑，必须强调：

**向量检索永远会返回 top-K 条最近邻**——哪怕库里根本没有相关内容，也会硬凑 K 条回来（那是向量空间里距离最近的无意义噪声）。如果用「返回条数」来判断库里有没有，真缺口会被系统性漏报，完善度虚高。

因此，本工具**判定依据只有最高相似度分数，绝不看检索返回了几条**。

### 阈值不能拍脑袋

阈值必须通过**校准模式**确定：用一个手动标注的校准题集（每题标注「库里确实有=1 / 确实没有=0」），系统跑完分析两组分数分布，自动推荐一个能最好区分两者的阈值。默认阈值为 `0.605`（已校准）。

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 相似度阈值二分判定 | 有覆盖 / 真缺口，纯分数驱动，不看返回条数 |
| 阈值自动校准 | 跑校准集 → 分析分数分布 → 推荐最佳阈值，给你完整的统计诊断 |
| 多业务场景支持 | 文档按场景分子目录，报告以场景为维度，一眼看出哪个场景缺得多 |
| 整场景缺失高亮 | 某个场景无文档或全部题目为缺口时，报告顶部标红警告 |
| MD5 指纹增量更新 | 只重算变动的文件，未变文件零开销跳过 |
| 可视化 HTML 报告 | 完善度总览、场景热力图、真缺口清单（附分数+原文片段） |
| 多种文档格式 | HTML（BeautifulSoup 去标签去噪声）、PDF（pdfplumber）、Excel（openpyxl）、TXT |
| 中文智能切片 | 300–500 字/片，片间 80 字重叠，段落优先 + jieba 辅助 |
| LLM 答案生成（可选） | 默认关闭，开启后对已覆盖的题用 LLM 基于检索内容生成答案 |

---

## 🛠 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言 | Python 3.9+ |
| 向量库 | ChromaDB（持久化，支持按 ID 增删改） |
| Embedding | 本地 BGE 中文模型 `BAAI/bge-large-zh-v1.5`（默认），或 OpenAI 兼容 API |
| HTML 清洗 | BeautifulSoup4 + lxml |
| PDF 提取 | pdfplumber |
| Excel | openpyxl |
| 中文分词 | jieba（辅助切片） |
| LLM（可选） | DeepSeek / 任意 OpenAI 兼容 API |

---

## 📁 项目结构

```
knowledge-gap-detector/
├── config.py              # 配置中心（阈值、LLM、嵌入方式、检索参数）
├── document_processor.py  # HTML/PDF/Excel 清洗 + jieba 辅助智能切片
├── vector_store.py        # ChromaDB + BGE embedding（HF 镜像支持）
├── updater.py             # MD5 指纹增量更新（只处理变动文件）
├── calibrator.py          # 校准模式 → 跑校准集 → 推荐阈值 + 分数分布诊断
├── tester.py              # 批量测试 + HTML 报告生成
├── main.py                # CLI 入口（update/calibrate/test/full 四模式）
├── requirements.txt
├── data/
│   ├── documents/         # 按场景分子目录放文档（如 套餐升值/、携号转网/）
│   ├── questions.csv      # 测试问题（scenario, question）
│   ├── calibration.csv    # 校准题集（scenario, question, known_in_kb）
│   └── file_index.json    # 文件指纹（自动维护，勿手动编辑）
├── output/
│   ├── batch_results.csv  # 逐题明细
│   └── reports/
│       └── report.html    # 可视化报告
└── vector_store/          # 持久化向量库（自动生成）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd knowledge-gap-detector
python -m venv .venv
source .venv/bin/activate

# 国内用户建议先设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

pip install -r requirements.txt
```

### 2. 准备知识库文档

把文档按业务场景放入 `data/documents/`，一个场景一个子目录：

```
data/documents/
├── 套餐升值/
│   ├── 套餐升值业务指南.html
│   └── 5G升档FAQ.xlsx
├── 携号转网/
│   ├── 携号转网服务指南.html
│   └── 携号转网办理流程.pdf
└── 宽带新装/
    └── 宽带业务手册.txt
```

### 3. 准备测试问题

编辑 `data/questions.csv`，格式：

```csv
scenario,question
套餐升值,套餐升值后原有合约期是否保持不变？
套餐升值,副卡是否可以升档？
携号转网,携号转网需要满足哪些条件？
宽带新装,电信宽带新装需要准备哪些材料？
```

### 4. 准备校准题集

编辑 `data/calibration.csv`，格式：

```csv
scenario,question,known_in_kb
套餐升值,套餐升值后原有合约期是否保持不变？,1
套餐升值,巴西足球联赛的冠军是谁？,0
```

- `known_in_kb=1` → 库里**确实有**这道题的答案
- `known_in_kb=0` → 库里**确实没有**这道题的答案

> 💡 校准题集建议每组至少 5 题，且覆盖不同场景。

### 5. 入库 & 校准

```bash
# 第一步：增量更新知识库
python main.py --mode update

# 第二步：校准阈值
python main.py --mode calibrate
```

校准输出示例：

```
📗 库里有   (n=7)
   分数范围: 0.6104 ~ 0.7429
   均值: 0.6776
📕 库里没有 (n=5)
   分数范围: 0.1597 ~ 0.4701
   均值: 0.3203

🎯 推荐阈值: 0.605    该阈值下误报 0 题、漏报 0 题
✅ 两组分数可分离性良好
```

如果可分离性不佳，系统会明确警告并给出排查建议。

### 6. 写入阈值并跑测试

```bash
# 方式一：环境变量（推荐）
export SIMILARITY_THRESHOLD=0.605

# 方式二：直接改 config.py
# SIMILARITY_THRESHOLD = 0.605

python main.py --mode test
```

### 7. 查看报告

打开 `output/reports/report.html`，你会看到：

- **总览卡片**：整体完善度、覆盖数、缺口数、场景数
- **场景热力图**：每个场景的题数 / 覆盖 / 缺口 / 完善度百分比
- **整场景缺失警告**：没有文档或全部为缺口的场景红色高亮
- **真缺口清单**：每条缺口附最高分 + 最相似 chunk 原文，方便人工复核

---

## 📋 CLI 模式一览

| 命令 | 作用 | 何时用 |
|------|------|--------|
| `python main.py --mode update` | 增量更新知识库 | 增删改文档后 |
| `python main.py --mode calibrate` | 校准阈值 | 首次使用 / 文档大变后 |
| `python main.py --mode test` | 批量测试 + 生成报告 | 日常探测缺口 |
| `python main.py --mode full` | update + test 一键完成 | 快速全流程 |

---

## 📊 报告解读

### 完善度怎么算

```
某场景完善度 = 该场景「已覆盖题数」/ 该场景「总题数」× 100%
整体完善度   = 全部「已覆盖题数」/「全部题数」× 100%
```

### 真缺口清单怎么看

每条缺口都标注了：
- **问题原文**：你出的测试题
- **最高分**：向量检索返回的最高相似度分数（可用于判断是「接近覆盖但差点」还是「完全无关」）
- **最相似片段**：库里和这道题最接近的一段 chunk 原文

这让你可以逐条复核：是**真的缺内容**，还是**内容有但没检索到**？后者可能意味着文档切片、embedding 模型或阈值需要调优。

### 场景维度怎么定位

报告的场景热力图一眼就能看出：
- 🟢 95%+ → 该场景文档完备
- 🟡 70-90% → 有少量缺口，需补充
- 🔴 <70% → 缺口较多，优先补充
- ⚠️ 整场景缺失 → 报告顶部红色警告

---

## ⚙️ 配置说明

所有配置均可通过环境变量覆盖。以下是完整环境变量列表：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HF_ENDPOINT` | (空) | HuggingFace 镜像（国内用户设为 `https://hf-mirror.com`） |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 本地 embedding 模型名称 |
| `EMBEDDING_MODE` | `local` | embedding 模式：`local` 或 `api` |
| `EMBEDDING_API_BASE_URL` | (同 LLM) | API embedding 接口地址 |
| `EMBEDDING_API_KEY` | (空) | API embedding 密钥 |
| `EMBEDDING_API_MODEL` | `text-embedding-3-small` | API embedding 模型名 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM API 地址（仅答案生成时用） |
| `LLM_API_KEY` | (空) | LLM API 密钥 |
| `LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `SIMILARITY_THRESHOLD` | `0.605` | 核心判定阈值（必须校准后设定） |
| `TOP_K` | `5` | 检索返回条数 |
| `CHUNK_SIZE` | `400` | 切片字数 |
| `CHUNK_OVERLAP` | `80` | 切片重叠字数 |
| `ENABLE_ANSWER_GEN` | `false` | 是否开启 LLM 答案生成 |

---

## ❓ 常见问题

### BGE 模型下载太慢 / 卡住？

设置 HuggingFace 国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新运行即可。首次下载约 1.3GB，请耐心等待。

### 真缺口比例过高，是不是出错了？

请逐一检查：

1. **文档放对了吗？** 文档必须在 `data/documents/` 下按场景分子目录
2. **场景标签对上了吗？** `questions.csv` 中的 `scenario` 列必须和子目录名一致
3. **文档清洗是否有问题？** 运行 `update` 时看是否有「文档为空」的警告
4. **阈值设对了吗？** 不同的 embedding 模型、不同的文档语料需要不同的阈值，务必重新校准

### 阈值怎么重新校准？

修改文档后、或换了 embedding 模型 / 文档语料后，都需要重新校准：

```bash
python main.py --mode calibrate
```

把新输出的推荐阈值写入 `SIMILARITY_THRESHOLD`。

### 增量更新和全量重建的区别？

- **增量更新**（`--mode update`）：基于 MD5 指纹，只处理新增/修改/删除的文件。改 1 个文件就只重算那 1 个。
- **全量重建**：如果你需要（比如换了 embedding 模型），可以删除 `data/file_index.json` 和 `vector_store/` 后重新 `update`。

### 能直接用 OpenAI 的 embedding 吗？

可以。设置环境变量：

```bash
export EMBEDDING_MODE=api
export EMBEDDING_API_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=sk-xxx
export EMBEDDING_API_MODEL=text-embedding-3-small
```

注意：换 embedding 后阈值会变，需要重新校准。

---

## ⚠️ 诚实声明

### 关于阈值二分的局限

本工具只区分「有覆盖 / 真缺口」**两类**，不细分「检索没找到」还是「模型没答好」。这是设计取舍：对探测缺口够用，但**它不是一个诊断检索质量的工具**。如果你需要区分「真缺口 / 检索问题 / 模型问题」，这本工具目前不支持。

### 关于"证明不了全"

这工具能告诉你**「测到的地方缺什么」**，但证明不了知识库「全了」。

覆盖率取决于你的测试题出得全不全。比如你只测了"套餐升值"和"携号转网"，那"宽带新装"有什么缺口它不会告诉你——因为你没出题。它是**探照灯**，不是**裁判**。"应该有哪些内容"需要业务方自己定基线。

---

## 📝 设计取舍

本项目是 Vibe Coding 的产物，设计上保留了相当的简洁性：

- 不做 Web UI / API Server，纯 CLI + 文件输入输出，适合脚本化接入 CI
- 不做增量校准（校准始终跑完整校准集），保证结果可复现
- 不做自动写入阈值（让你在 calibrate → 看分布 → 确认 → 写配置之间有个判断环节）

---

## 📄 License

MIT
