import os
import sys
from src.core.config import settings
from src.core.logger import get_logger

# 初始化日志
logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("🚀 O2O 营销智能体启动（基于天池数据集）")
    logger.info(f"📌 运行环境: {settings.ENV}")

    try:
        # ==========================================
        # 1. 检查数据集是否存在
        # ==========================================
        raw_data_path = os.path.join(settings.DATA_RAW_DIR, "tianchi_o2o")
        offline_train_file = os.path.join(raw_data_path, "offline_train.csv.zip")

        if not os.path.exists(offline_train_file):
            logger.error("❌ 未找到天池数据集！")
            logger.error(f"   请先下载数据集到: {raw_data_path}")
            logger.error(f"   下载地址: https://tianchi.aliyun.com/dataset/137322")
            logger.error("   需要下载: offline_train.csv.zip (必选) + online_train.csv.zip (可选)")
            sys.exit(1)

        # ==========================================
        # 2. 初始化核心组件
        # ==========================================
        logger.info("🔧 初始化核心组件...")

        # 2.1 初始化 Embeddings（用于 RAG 向量库）
        logger.info("   - 加载 Embedding 模型...")
        from src.rag.embeddings import QwenEmbeddings
        embeddings = QwenEmbeddings()

        # 2.2 初始化天池数据加载器
        logger.info("   - 初始化数据加载器...")
        from src.data.tianchi_loader import TianchiDataLoader
        data_loader = TianchiDataLoader()
        data_loader.count_user_segments()

        # 2.3 初始化向量库（如果有知识库文件）
        logger.info("   - 初始化向量库...")
        from src.rag.vectorstore import VectorStoreManager
        vs_manager = VectorStoreManager(embeddings)

        # 可选：如果有知识库 Markdown 文件，自动加载
        kb_dir = settings.KNOWLEDGE_BASE_DIR
        if os.path.exists(kb_dir) and len([f for f in os.listdir(kb_dir) if f.endswith(".md")]) > 0:
            logger.info("   - 检测到知识库文件，正在构建/更新向量库...")
            from src.rag.document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            docs = doc_processor.load_and_split_documents()
            vs_manager.add_documents(docs)

        # 2.4 构建 RAG 检索器
        logger.info("   - 构建 RAG 检索器...")
        from src.rag.retriever import build_retriever
        retriever = build_retriever(vs_manager)

        # 2.5 初始化 Uplift 模型（基于天池数据训练）
        logger.info("   - 加载/训练 Uplift 模型...")
        from src.uplift.model import UpliftModel
        uplift_model = UpliftModel()

        # 2.6 初始化营销智能体
        logger.info("   - 初始化营销智能体...")
        from src.agent.marketing_agent import MarketingAgent
        agent = MarketingAgent(
            retriever=retriever,
            uplift_model=uplift_model,
            data_loader=data_loader
        )

        # ==========================================
        # 3. 执行策略生成（示例）
        # ==========================================
        logger.info("=" * 60)
        logger.info("🎯 开始生成营销策略（示例）")

        # 示例：使用天池数据集中的真实用户 ID（可自行修改）
        example_user_id = "14394087"  # 这是数据集中的一个真实用户 ID
        example_distance = 2.5  # 单位：km

        logger.info(f"👤 示例用户: ID={example_user_id}, 距离={example_distance}km")

        # 生成策略
        strategy = agent.generate_strategy(
            user_id=example_user_id,
            distance=example_distance
        )

        # ==========================================
        # 4. 输出结果
        # ==========================================
        logger.info("✅ 策略生成完成！")
        logger.info("📋 最终决策:")

        # 美化输出
        import json
        logger.info(json.dumps(strategy, ensure_ascii=False, indent=4))

        logger.info("=" * 60)
        logger.info("🎉 程序运行结束")

    except Exception as e:
        logger.error(f"❌ 程序运行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()