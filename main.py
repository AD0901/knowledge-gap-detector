"""
知识库缺口探测系统 —— 主入口
用法:
  python main.py --mode calibrate   # 校准阈值
  python main.py --mode update      # 增量更新知识库
  python main.py --mode test        # 批量测试 + 生成报告
  python main.py --mode full        # update + test 全流程
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


def cmd_calibrate():
    """校准模式"""
    store = get_vector_store()
    if store.count() == 0:
        print("⚠️ 向量库为空！请先运行 --mode update 导入文档。")
        return

    result = run_calibration(CALIBRATION_FILE, store, top_k=TOP_K)
    if "error" in result:
        print(f"❌ 校准失败: {result['error']}")
        return

    threshold = result["recommended_threshold"]
    print(f"\n{'='*60}")
    print(f"  推荐阈值: {threshold}")
    print(f"  请将 config.py 中的 SIMILARITY_THRESHOLD 修改为 {threshold}")
    print(f"  或设置环境变量: export SIMILARITY_THRESHOLD={threshold}")
    print(f"{'='*60}")


def cmd_update():
    """增量更新模式"""
    print(f"\n{'='*60}")
    print(f"  增量更新知识库")
    print(f"{'='*60}\n")

    store = get_vector_store()
    stats = incremental_update(
        DOCUMENTS_DIR, FILE_INDEX_FILE, store, CHUNK_SIZE, CHUNK_OVERLAP
    )

    print(f"\n{'='*60}")
    print(f"  更新完成")
    print(f"  新增: {stats['added']}, 修改: {stats['modified']}, "
          f"删除: {stats['deleted']}, 不变: {stats['unchanged']}")
    print(f"  Chunk 变化: +{stats['chunks_added']} / -{stats['chunks_deleted']}")
    print(f"{'='*60}")


def cmd_test():
    """批量测试模式"""
    store = get_vector_store()
    if store.count() == 0:
        print("⚠️ 向量库为空！请先运行 --mode update 导入文档。")
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
    """全流程：update + test"""
    cmd_update()
    cmd_test()


def main():
    parser = argparse.ArgumentParser(
        description="知识库缺口探测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode calibrate   # 校准阈值（首次必跑）
  python main.py --mode update      # 增量更新知识库
  python main.py --mode test        # 批量测试 + 生成报告
  python main.py --mode full        # 全流程
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["calibrate", "update", "test", "full"],
        help="运行模式",
    )
    args = parser.parse_args()

    mode_handlers = {
        "calibrate": cmd_calibrate,
        "update": cmd_update,
        "test": cmd_test,
        "full": cmd_full,
    }

    handler = mode_handlers[args.mode]
    try:
        handler()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        raise


if __name__ == "__main__":
    main()
