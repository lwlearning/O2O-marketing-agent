# ========== 终极路径修复（100%生效）==========
import sys
import os
from pathlib import Path

# 自动获取根目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 打印路径（你能看到系统已经识别根目录了）
print("项目根目录：", BASE_DIR)
# =============================================

from src.data_process import run_data_pipeline
from src.rag_service import UpliftRAGService
from src.marketing_agent import MarketingAgent

if __name__ == "__main__":
    run_data_pipeline()
    rag = UpliftRAGService()
    agent = MarketingAgent(rag)
    strategy = agent.generate_marketing_strategy(user_id=1001, distance=2)
    print(strategy)