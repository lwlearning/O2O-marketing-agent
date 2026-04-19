# src/uplift/model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class UpliftModel:
    def __init__(self):
        self.model = self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(settings.UPLIFT_MODEL_PATH):
            logger.info("✅ 加载已训练的 Uplift 模型")
            return joblib.load(settings.UPLIFT_MODEL_PATH)

        logger.info("🔧 训练新的 Uplift 模型（基于天池数据集）")
        return self._train_model()

    def _train_model(self):
        from src.data.tianchi_loader import TianchiDataLoader
        loader = TianchiDataLoader()
        df = loader.load_raw_data()

        # 特征处理
        def distance_to_km(x):
            if pd.isna(x): return 5.0
            x = float(x)
            if x == 0: return 0.3
            if x == 10: return 6.0
            return x * 0.5

        df["distance_km"] = df["Distance"].apply(distance_to_km)
        df["is_used"] = df["Date"].notna().astype(int)
        df["is_treated"] = df["Coupon_id"].notna().astype(int)

        # 双模型 Uplift 架构
        treated = df[df["is_treated"] == 1]
        control = df[df["is_treated"] == 0]

        model_t = RandomForestClassifier(n_estimators=50, random_state=42)
        model_c = RandomForestClassifier(n_estimators=50, random_state=42)

        if not treated.empty:
            model_t.fit(treated[["distance_km"]], treated["is_used"])
        if not control.empty:
            model_c.fit(control[["distance_km"]], control["is_used"])

        model = {
            "model_treated": model_t,
            "model_control": model_c,
            "features": ["distance_km"]
        }

        os.makedirs(os.path.dirname(settings.UPLIFT_MODEL_PATH), exist_ok=True)
        joblib.dump(model, settings.UPLIFT_MODEL_PATH)
        logger.info("✅ Uplift 模型训练完成并保存")
        return model

    def predict_uplift(self, user_features):
        """✅ 修复：安全获取概率，永不越界"""
        mt = self.model["model_treated"]
        mc = self.model["model_control"]

        distance = user_features.get("avg_distance", 5.0)
        df = pd.DataFrame([{"distance_km": distance}])

        # 安全获取正类概率
        def get_prob(model, data):
            if not hasattr(model, "predict_proba"):
                return 0.0
            p = model.predict_proba(data)[0]
            return p[1] if len(p) > 1 else 0.0

        p_treated = get_prob(mt, df)
        p_control = get_prob(mc, df)

        uplift = p_treated - p_control
        return max(0.0, uplift)