import json
import re
import dashscope
from dashscope import Generation
from src.config import settings


class MarketingAgent:
    def __init__(self, rag_service):
        self.rag = rag_service
        # ✅ 全局设置 API Key（兼容新旧版 SDK）
        dashscope.api_key = settings.QWEN_API_KEY

    def _llm(self, prompt):
        """调用通义千问 LLM，修复了 SDK 调用和输出解析"""
        try:
            # ✅ 新版 DashScope SDK 标准调用方式
            resp = Generation.call(
                model=settings.QWEN_CHAT_MODEL,
                prompt=prompt,
                result_format="message"  # ✅ 推荐用 message 格式，更稳定
            )

            # ✅ 安全获取输出文本
            if resp.status_code == 200 and resp.output and resp.output.choices:
                return resp.output.choices[0].message.content
            else:
                raise Exception(f"LLM 调用失败: {resp}")

        except Exception as e:
            raise Exception(f"LLM 服务异常: {str(e)}")

    def _extract_json(self, text):
        """✨ 核心修复：从 LLM 输出中提取纯 JSON（去除 Markdown 代码块）"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 尝试提取 ```json ... ``` 代码块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 尝试提取第一个 {...}
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        raise ValueError(f"无法从 LLM 输出中解析 JSON: {text}")

    def generate_marketing_strategy(self, user_id, distance):
        query = f"用户{user_id} 距离{distance}km 最优优惠券策略"
        context = self.rag.retrieve(query)

        # ✅ 优化 Prompt，明确要求输出纯 JSON
        prompt = f"""你是专业电商营销智能Agent，基于Uplift因果纠偏数据决策。

【参考数据】
{context}

【当前用户】
ID={user_id}，距离={distance}km

【输出要求】
1. 只输出一个纯 JSON 对象，不要任何其他文字
2. JSON 结构：
{{
    "send_coupon": true/false,
    "coupon_type": "券类型",
    "coupon_value": "券金额",
    "reason": "决策原因",
    "user_segment": "用户分群"
}}
3. 不要用 Markdown 代码块包裹"""

        llm_output = self._llm(prompt)
        return self._extract_json(llm_output)