import json
import dashscope
from dashscope import Generation
from src.config import settings

class MarketingAgent:
    def __init__(self, rag_service):
        self.rag = rag_service
        dashscope.api_key = settings.QWEN_API_KEY

    def _llm(self, prompt):
        resp = Generation.call(
            model=settings.QWEN_CHAT_MODEL,
            prompt=prompt,
            result_format="json"
        )
        return resp.output.text

    def generate_marketing_strategy(self, user_id, distance):
        query = f"用户{user_id} 距离{distance}km 最优优惠券策略"
        context = self.rag.retrieve(query)

        prompt = f"""
你是专业电商营销智能Agent，基于Uplift因果纠偏数据决策。

参考数据：
{context}

用户：ID={user_id}，距离={distance}km

输出JSON：
send_coupon: bool, coupon_type: str, coupon_value: str, reason: str, user_segment: str
"""
        return json.loads(self._llm(prompt))