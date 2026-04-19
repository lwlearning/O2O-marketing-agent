# src/agent/marketing_agent.py
import json
import re
import dashscope
from dashscope import Generation
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)
# Uplift最低盈利阈值：天池行业通用10%
UPLIFT_MIN_PROFIT = 0.1

class MarketingAgent:
    def __init__(self, retriever, uplift_model, data_loader):
        self.retriever = retriever
        self.uplift_model = uplift_model
        self.data_loader = data_loader
        dashscope.api_key = settings.QWEN_API_KEY

    def _llm(self, prompt):
        try:
            resp = Generation.call(
                model=settings.QWEN_CHAT_MODEL,
                prompt=prompt,
                result_format="message"
            )
            if resp.status_code == 200 and resp.output.choices:
                return resp.output.choices[0].message.content
            logger.error(f"LLM调用失败: {resp}")
            return None
        except Exception as e:
            logger.error(f"LLM异常: {e}")
            return None

    def _extract_json(self, text):
        if not text:
            return None
        try:
            text = re.sub(r'```json|```', '', text).strip()
            return json.loads(text)
        except:
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                return json.loads(match.group()) if match else None
            except:
                return None

    def generate_strategy(self, user_id: str, distance: float):
        user_features = self.data_loader.get_user_features(user_id)
        historical_spend = user_features.get("historical_spend", 0)
        user_segment = user_features.get("user_segment", "平台新用户")

        uplift_score = self.uplift_model.predict_uplift(user_features)
        logger.info(f"用户分群结果: {user_segment}，Uplift增量: {uplift_score:.2%}")

        # ======================
        # 🔥 Uplift硬决策：增量不足，无论什么人群都不发券
        # ======================
        if uplift_score < UPLIFT_MIN_PROFIT:
            send_coupon = False
        else:
            send_coupon = True

        context = self.retriever.invoke(f"{user_segment} 用户 Uplift={uplift_score:.2%} 优惠券投放策略")

        prompt = f"""你是O2O因果营销智能Agent
        用户ID：{user_id}
        距离：{distance}km
        用户分层：{user_segment}
        历史核销：{historical_spend}
        Uplift因果增量转化：{uplift_score:.2%}
        风控规则：**Uplift＜10% 禁止发放任何优惠券**

        天池数据集券类型：满减券、折扣券
        新用户可用大额拉新满减券，老用户常规小额券

        严格输出JSON：
        {{
            "send_coupon": {send_coupon},
            "coupon_type": "大额拉新满减券/常规满减券/折扣券/不发放",
            "coupon_value": "满20减5/满10减2/8折/无",
            "reason": "依据Uplift因果增量收益+用户空间分层判断是否补贴",
            "user_segment": "{user_segment}"
        }}"""

        # 5. LLM生成
        llm_out = self._llm(prompt)
        result = self._extract_json(llm_out)

        # 6. 兜底策略
        if not result:
            result = {
                "send_coupon": False,
                "coupon_type": "无",
                "coupon_value": "无",
                "reason": "兜底策略",
                "user_segment": user_segment
            }

        return result