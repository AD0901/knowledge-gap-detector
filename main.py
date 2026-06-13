"""
知识库缺口探测系统 —— 主入口

用法:
  python main.py --mode update      # 增量更新知识库（增删改文档后跑）
  python main.py --mode calibrate   # 校准阈值（首次必跑）
  python main.py --mode test        # 批量测试 + 生成报告
  python main.py --mode full        # update + test 一键完成
"""

import argparse
import sys

from config import (
    CALIBRATION_FILE, QUESTIONS_FILE, DOCUMENTS_DIR, FILE_INDEX_FILE,
    SIMILARITY_THRESHOLD, TOP_K, ENABLE_ANSWER_GEN,
    BATCH_RESULTS_FILE, REPORT_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP,
)
from vector_store import get_vector_store
from updater import incremental_update
from calibrator import run_calibration
from tester import run_test


def _ensure_store_not_empty(store):
    """确保向量库中有数据，否则提示先执行 update"""
    if store.count() == 0:
        print("\n⚠️  向量库为空！")
        print("   请先运行: python main.py --mode update")
        print("   将文档放入 data/documents/ 按场景分目录后执行。\n")
        return False
    return True


def cmd_update():
    """增量更新模式"""
    print(f"\n{'='*60}")
    print(f"  📦 增量更新知识库")
    print(f"{'='*60}")

    store = get_vector_store()
    stats = incremental_update(
        DOCUMENTS_DIR, FILE_INDEX_FILE, store,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(f"\n{'='*60}")
    print(f"  ✅ 更新完成")
    print(f"     新增: {stats['added']} | 修改: {stats['modified']} | "
          f"删除: {stats['deleted']} | 不变: {stats['unchanged']}")
    if stats["errors"]:
        print(f"     ⚠️ 错误: {stats['errors']} 个文件处理失败")
    print(f"     Chunk 变动: +{stats['chunks_added']} / -{stats['chunks_deleted']}")
    print(f"{'='*60}\n")


def cmd_calibrate():
    """校准模式"""
    store = get_vector_store()
    if not _ensure_store_not_empty(store):
        return

    result = run_calibration(CALIBRATION_FILE, store, top_k=TOP_K)
    if "error" in result:
        print(f"\n❌ 校准失败: {result['error']}\n")
        return

    threshold = result["recommended_threshold"]
    print(f"\n  请将阈值写入环境变量或 config.py:")
    print(f"  > export SIMILARITY_THRESHOLD={threshold}")
    print()


def cmd_test():
    """批量测试模式"""
    store = get_vector_store()
    if not _ensure_store_not_empty(store):
        return

    run_test(
        questions_file=QUESTIONS_FILE,
        documents_dir=DOCUMENTS_DIR,
        vector_store=store,
        threshold=SIMILARITY_THRESHOLD,
        top_k=TOP_K,
        enable_answer_gen=ENABLE_ANSWER_GEN,
        batch_csv_path=BATCH_RESULTS_FILE,
        report_html_path=REPORT_FILE,
    )


def cmd_full():
    """全流程：先更新知识库，再跑测试"""
    cmd_update()
    cmd_test()


def main():
    parser = argparse.ArgumentParser(
        description="🔍 知识库缺口探测系统 — 用问答当探针，找出知识库答不出的内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode update      # 增量更新知识库
  python main.py --mode calibrate   # 校准阈值（首次必跑）
  python main.py --mode test        # 批量测试 + 生成报告
  python main.py --mode full        # 一键 update + test

首次使用流程:
  1. 把文档放入 data/documents/ 按场景分目录
  2. 编辑 data/calibration.csv 标注校准题
  3. 编辑 data/questions.csv 放测试问题
  4. python main.py --mode update       # 入库
  5. python main.py --mode calibrate    # 校准阈值
  6. 将阈值写入 config.py 或 export SIMILARITY_THRESHOLD=0.xx
  7. python main.py --mode test         # 跑测试 → 看报告
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["update", "calibrate", "test", "full"],
        help="运行模式: update(增量更新), calibrate(校准阈值), test(批量测试), full(全流程)",
    )
    args = parser.parse_args()

    handlers = {
        "update": cmd_update,
        "calibrate": cmd_calibrate,
        "test": cmd_test,
        "full": cmd_full,
    }

    try:
        handlers[args.mode]()
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("   请检查 data/ 目录下的文件是否存在。\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断\n")
        sys.exit(0)
    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("   请运行: pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}\n")
        raise


if __name__ == "__main__":
    main()
